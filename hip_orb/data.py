"""Worker-local, bounded-memory loading of packed OrbMol teacher labels."""
from collections import OrderedDict
from pathlib import Path
import os
from itertools import islice

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

FIELDS = ("atomic_numbers", "coords", "forces", "energy", "charge", "spin", "hessian_upper")


def build_index(paths, output):
    """Build metadata once on CPUs, never during a training epoch."""
    paths = [Path(path).resolve() for path in paths]
    counts, sizes, sources = [], [], []
    for path in paths:
        with h5py.File(path) as f:
            if not f.attrs.get("complete", False) or f.attrs.get("model") != "orbmol_v2":
                raise ValueError("Require complete OrbMol teacher shards")
            n = f["natoms"][:]
            if not len(n) or not np.all(n == n[0]):
                raise ValueError("Training shards must have one atom count")
            if "hessian_upper" not in f:
                raise ValueError("Require packed upper-triangle labels")
            dimension = 3 * int(n[0])
            if not np.array_equal(f["hessian_ptr"][:], np.arange(len(n) + 1) * dimension * (dimension + 1) // 2):
                raise ValueError("Packed pointers differ")
            counts.append(len(n))
            sizes.append(int(n[0]))
            sources.append(str(f.attrs["source_dataset"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, paths=np.asarray([str(path) for path in paths]),
             ptr=np.r_[0, np.cumsum(counts)], natoms=np.asarray(sizes, dtype=np.int32),
             sources=np.asarray(sources))


class OrbMolDataset(Dataset):
    """Index immutable shards; retain file handles or shard arrays per worker.

    cache_mode='arrays' amortizes decompression across an entire shard.
    cache_mode='chunks' retains HDF5 datasets and their raw chunk caches.
    No HDF5 handles are serialized into spawned loader workers.
    """
    def __init__(self, index, *, backend="hdf5", cache_mode="arrays", cache_bytes=256 << 20,
                 max_open=8, chunk_cache_bytes=4 << 20, training_only=False):
        with np.load(index, allow_pickle=False) as source:
            self.paths = source["paths"].tolist()
            self.ptr = source["ptr"].copy()
            self.natoms = source["natoms"].copy()
            self.sources = source["sources"].tolist()
        if backend not in ("hdf5", "mmap") or cache_mode not in ("arrays", "chunks", "none"):
            raise ValueError("Unknown backend or cache mode")
        if cache_bytes < 0 or max_open < 1:
            raise ValueError("Invalid cache limits")
        self.backend, self.cache_mode = backend, cache_mode
        self.fields = tuple(key for key in FIELDS if not training_only or key not in ("energy", "forces"))
        self.cache_bytes, self.max_open, self.chunk_cache_bytes = cache_bytes, max_open, chunk_cache_bytes
        self._reset()

    def _reset(self):
        self._pid = os.getpid()
        self._files, self._arrays = OrderedDict(), OrderedDict()
        self._bytes = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_files=OrderedDict(), _arrays=OrderedDict(), _bytes=0, _pid=None)
        return state

    def __len__(self):
        return int(self.ptr[-1])

    def close(self):
        for handle, _ in self._files.values():
            if handle is not None:
                handle.close()
        self._reset()

    def _datasets(self, shard):
        if self._pid != os.getpid():
            self.close()
        if shard in self._files:
            self._files.move_to_end(shard)
            return self._files[shard][1]
        if self.backend == "mmap":
            datasets = {key: np.load(Path(self.paths[shard]) / f"{key}.npy", mmap_mode="r")
                        for key in (*self.fields, "omol_index")}
            handle = None
        else:
            try:
                import hdf5plugin  # noqa: F401 - register optional filters in each worker
            except ImportError:
                pass
            handle = h5py.File(self.paths[shard], "r", rdcc_nbytes=self.chunk_cache_bytes)
            # Keep dataset objects alive: their HDF5 chunk caches live here.
            datasets = {key: handle[key] for key in (*self.fields, "omol_index")}
        self._files[shard] = handle, datasets
        while len(self._files) > self.max_open:
            old, _ = self._files.popitem(last=False)[1]
            if old is not None:
                old.close()
        return datasets

    def _payload(self, shard):
        if self._pid != os.getpid():
            self.close()
        if shard in self._arrays:
            self._arrays.move_to_end(shard)
            return self._arrays[shard]
        datasets = self._datasets(shard)
        if self.cache_mode != "arrays" or self.backend == "mmap":
            return datasets
        size = sum(value.size * value.dtype.itemsize for value in datasets.values())
        if size > self.cache_bytes:
            return datasets
        while self._arrays and self._bytes + size > self.cache_bytes:
            _, old = self._arrays.popitem(last=False)
            self._bytes -= sum(value.nbytes for value in old.values())
        payload = {key: value[:] for key, value in datasets.items()}
        self._arrays[shard] = payload
        self._bytes += size
        return payload

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.ptr, index, side="right") - 1)
        row = int(index - self.ptr[shard])
        n = int(self.natoms[shard])
        width = (3 * n) * (3 * n + 1) // 2
        values = self._payload(shard)
        sample = {key: np.asarray(values[key][row * n:(row + 1) * n]).copy()
                  for key in ("atomic_numbers", "coords", "forces") if key in self.fields}
        sample.update({key: np.asarray(values[key][row]).copy()
                       for key in ("energy", "charge", "spin", "omol_index") if key in values})
        sample["hessian_upper"] = np.asarray(values["hessian_upper"][row * width:(row + 1) * width]).copy()
        sample["source_dataset"] = self.sources[shard]
        if self.cache_mode == "none":
            self.close()
        return sample

    def __getitems__(self, indices):
        if any(index < 0 or index >= len(self) for index in indices):
            raise IndexError("Sample index outside dataset")
        # Group accesses by shard, even when a random batch spans many files.
        groups = {}
        for position, index in enumerate(indices):
            shard = int(np.searchsorted(self.ptr, index, side="right") - 1)
            groups.setdefault(shard, []).append((position, index))
        result = [None] * len(indices)
        for shard, group in groups.items():
            n = int(self.natoms[shard])
            width = 3 * n * (3 * n + 1) // 2
            rows = np.asarray([index - self.ptr[shard] for _, index in group])
            payload = self._payload(shard)
            fields = {}
            for key, values in payload.items():
                block = n if key in ("atomic_numbers", "coords", "forces") else width if key == "hessian_upper" else 1
                if isinstance(values, np.ndarray):
                    selected = values.reshape(-1, block, *values.shape[1:])[rows]
                else:
                    selected = np.empty((len(rows), block, *values.shape[1:]), dtype=values.dtype)
                    unique, inverse = np.unique(rows, return_inverse=True)
                    # Read each consecutive row span once. Preserve duplicates and input order.
                    cuts = np.r_[0, np.flatnonzero(np.diff(unique) != 1) + 1, len(unique)]
                    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                        lo, hi = int(unique[a]), int(unique[b - 1]) + 1
                        chunk = np.asarray(values[lo * block:hi * block]).reshape(hi - lo, block, *values.shape[1:])
                        mask = (inverse >= a) & (inverse < b)
                        selected[mask] = chunk[rows[mask] - lo]
                fields[key] = selected if block != 1 or key in ("atomic_numbers", "coords", "forces", "hessian_upper") else selected[:, 0]
            for j, (position, _) in enumerate(group):
                result[position] = {key: value[j] for key, value in fields.items()}
                result[position]["source_dataset"] = self.sources[shard]
            if self.cache_mode == "none":
                self.close()
        return result


class AtomBatchSampler(Sampler):
    """Equal-N batches with an explicit dense-Hessian element budget.

    Shard mode shuffles shards and rows, reducing shared-filesystem reads.
    Random mode shuffles all samples within each atom-count bucket.
    set_epoch changes the deterministic order; rank partitions shards first.
    """
    def __init__(self, dataset, *, batch_size=32, max_hessian_elements=2_000_000,
                 order="shard", seed=7, rank=0, world_size=1):
        if order not in ("shard", "random") or batch_size < 1 or max_hessian_elements < 1:
            raise ValueError("Invalid batch settings")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        self.dataset, self.batch_size, self.budget = dataset, batch_size, max_hessian_elements
        self.order, self.seed, self.epoch = order, seed, 0
        self.rank, self.world_size = rank, world_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = rng.permutation(len(self.dataset.paths))[self.rank::self.world_size]
        if self.order == "shard":
            groups = [np.arange(self.dataset.ptr[s], self.dataset.ptr[s + 1]) for s in shards]
        else:
            groups = [np.concatenate([np.arange(self.dataset.ptr[s], self.dataset.ptr[s + 1])
                                      for s in shards if self.dataset.natoms[s] == n])
                      for n in rng.permutation(np.unique(self.dataset.natoms[shards]))]
        for group in groups:
            rng.shuffle(group)
            s = int(np.searchsorted(self.dataset.ptr, group[0], side="right") - 1)
            n = int(self.dataset.natoms[s])
            cap = min(self.batch_size, max(1, self.budget // (3 * n)**2))
            for start in range(0, len(group), cap):
                yield group[start:start + cap].tolist()


def collate_samples(samples):
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
            if key != "source_dataset" else [sample[key] for sample in samples]
            for key in samples[0]}


class UniformSampleBatches:
    """Choose geometries uniformly before grouping equal-size validation batches."""

    def __init__(self, dataset, count, *, batch_size, max_hessian_elements, seed):
        if count < 1 or batch_size < 1 or max_hessian_elements < 1:
            raise ValueError("Uniform validation requires positive sample and batch limits")
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
        sizes = dataset.natoms[np.searchsorted(dataset.ptr, selected, side="right") - 1]
        self.batches = []
        for n in rng.permutation(np.unique(sizes)):
            indices = np.sort(selected[sizes == n])
            cap = min(batch_size, max(1, max_hessian_elements // (3 * int(n)) ** 2))
            self.batches.extend(indices[start:start + cap].tolist()
                                for start in range(0, len(indices), cap))

    def __iter__(self):
        return iter(self.batches)

    def set_epoch(self, epoch):
        # Validation membership and order stay fixed across evaluations.
        pass


class LimitedBatches:
    """Do not prefetch work beyond a bounded optimizer test."""
    def __init__(self, sampler, count):
        if count < 1:
            raise ValueError("Batch limit must be positive")
        self.sampler, self.count = sampler, count

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return islice(iter(self.sampler), self.count)


class LimitedSamples:
    """Cap geometries exactly, including a partial final batch."""
    def __init__(self, sampler, count):
        if count < 1:
            raise ValueError("Sample limit must be positive")
        self.sampler, self.count = sampler, count

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        remaining = self.count
        for batch in self.sampler:
            selected = batch[:remaining]
            yield selected
            remaining -= len(selected)
            if remaining == 0:
                break


class SkipBatches:
    """Resume at a batch boundary without reading earlier geometries."""
    def __init__(self, sampler, count):
        self.sampler, self.count, self.first_epoch = sampler, count, None

    def set_epoch(self, epoch):
        if self.first_epoch is None:
            self.first_epoch = epoch
        elif epoch != self.first_epoch:
            self.count = 0
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return islice(iter(self.sampler), self.count, None)


class StratifiedBatches:
    """Benchmark-only equal-N round robin; not a dataset-weighted epoch."""
    def __init__(self, sampler):
        self.sampler = sampler

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        from collections import deque
        groups = {}
        for batch in self.sampler:
            shard = np.searchsorted(self.sampler.dataset.ptr, batch[0], side="right") - 1
            n = int(self.sampler.dataset.natoms[shard])
            groups.setdefault(n, deque()).append(batch)
        while groups:
            for n in sorted(list(groups)):
                yield groups[n].popleft()
                if not groups[n]:
                    del groups[n]


def make_loader(dataset, *, workers=0, order="shard", seed=7, pin_memory=True,
                batch_size=32, max_hessian_elements=2_000_000, rank=0, world_size=1,
                max_batches=None, max_samples=None, start_batch=0,
                collate_fn=collate_samples, stratified=False, batch_sampler=None,
                prefetch_factor=2, uniform_samples=False):
    if prefetch_factor < 1:
        raise ValueError("Prefetch factor must be positive")
    if uniform_samples:
        if batch_sampler is not None or max_batches is not None or stratified or rank != 0 or world_size != 1:
            raise ValueError("Uniform validation requires one process and a sample limit without batch truncation")
        sampler = UniformSampleBatches(dataset, max_samples or len(dataset),
                                       batch_size=batch_size, max_hessian_elements=max_hessian_elements, seed=seed)
    elif batch_sampler is not None:
        if rank != 0 or world_size != 1 or stratified:
            raise ValueError("Custom batch samplers require one process and stratified=False")
        if collate_fn is collate_samples:
            raise ValueError("Custom batch samplers require an explicit compatible collate_fn")
        sampler = batch_sampler
    else:
        sampler = AtomBatchSampler(dataset, order=order, seed=seed, batch_size=batch_size,
                                   max_hessian_elements=max_hessian_elements, rank=rank, world_size=world_size)
    if stratified:
        sampler = StratifiedBatches(sampler)
    if max_batches is not None:
        sampler = LimitedBatches(sampler, max_batches)
    if max_samples is not None:
        sampler = LimitedSamples(sampler, max_samples)
    if start_batch:
        if start_batch < 0:
            raise ValueError("Starting batch must be nonnegative")
        sampler = SkipBatches(sampler, start_batch)
    kwargs = dict(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn,
                  num_workers=workers, pin_memory=pin_memory)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor, multiprocessing_context="spawn")
    return DataLoader(**kwargs)


def unpack_hessian(upper, natoms):
    dimension = 3 * int(natoms)
    row, col = torch.triu_indices(dimension, dimension, device=upper.device)
    result = upper.new_empty((*upper.shape[:-1], dimension, dimension))
    result[..., row, col] = upper
    result[..., col, row] = upper
    return result
