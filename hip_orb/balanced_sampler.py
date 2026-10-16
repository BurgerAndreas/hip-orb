"""Reusable atom-balanced plans for single-GPU packed Hessian training."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Pack samples under atom and structure limits, with an optional Hessian cap.

    Read only shard metadata. Build membership once, within shuffled shard
    windows, and shuffle group, batch, and row order each epoch. Windows bound the
    number of source files touched while planning. They do not bound open files
    during loading. No sample is dropped or duplicated. This sampler supports
    one training process; it does not implement distributed rank balancing.

    ``max_atoms`` is a hard upper bound, not a guaranteed batch size. The other
    limits or shard-window boundaries can produce smaller batches. Membership
    stays fixed across epochs. Use a different seed to create a different plan.
    """

    def __init__(
        self,
        dataset,
        *,
        max_atoms,
        batch_size=64,
        max_hessian_elements=None,
        window_shards=8,
        seed=7,
        shuffle=True,
        cache_path=None,
        order="locality",
    ):
        for name, value in {
            "max_atoms": max_atoms,
            "batch_size": batch_size,
            "window_shards": window_shards,
        }.items():
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_hessian_elements is not None and (
            not isinstance(max_hessian_elements, (int, np.integer)) or max_hessian_elements < 1
        ):
            raise ValueError("max_hessian_elements must be positive or None")
        if order not in ("global", "locality"):
            raise ValueError("Order must be global or locality")
        self.epoch = 0
        self.seed, self.shuffle = int(seed), bool(shuffle)
        self.settings = {
            "version": 2,
            "max_atoms": int(max_atoms),
            "batch_size": int(batch_size),
            "max_hessian_elements": None
            if max_hessian_elements is None
            else int(max_hessian_elements),
            "window_shards": int(window_shards),
            "seed": self.seed,
            "order": order,
        }
        try:
            ptr, sizes = np.asarray(dataset.ptr), np.asarray(dataset.natoms)
        except AttributeError as error:
            raise ValueError("Require shard metadata: dataset.ptr and dataset.natoms") from error
        if (
            ptr.ndim != 1
            or sizes.ndim != 1
            or len(sizes) == 0
            or len(ptr) != len(sizes) + 1
            or not np.issubdtype(ptr.dtype, np.integer)
            or not np.issubdtype(sizes.dtype, np.integer)
            or ptr[0] != 0
            or ptr[-1] != len(dataset)
            or np.any(np.diff(ptr) <= 0)
            or np.any(sizes <= 0)
        ):
            raise ValueError("Invalid shard atom-count metadata")
        ptr, sizes = ptr.astype(np.int64), sizes.astype(np.int64)
        if np.any(sizes > max_atoms) or (
            max_hessian_elements is not None and np.any(sizes > np.sqrt(max_hessian_elements / 9))
        ):
            raise ValueError("A geometry exceeds a batch budget; increase the budget")
        digest = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode())
        digest.update(ptr.tobytes())
        digest.update(sizes.tobytes())
        digest.update(json.dumps([str(p) for p in getattr(dataset, "paths", [])]).encode())
        self.fingerprint = digest.hexdigest()
        if cache_path is not None and Path(cache_path).exists():
            with np.load(cache_path, allow_pickle=False) as plan:
                if str(plan["fingerprint"]) != self.fingerprint:
                    raise ValueError("Cached batch plan differs from metadata or settings")
                self.indices = plan["indices"]
                self.batch_ptr = plan["batch_ptr"]
                self.group_ptr = plan["group_ptr"]
            self._validate(ptr, sizes)
        else:
            self.indices, self.batch_ptr, self.group_ptr = self._build(ptr, sizes)
            if cache_path is not None:
                self.save(cache_path)
        self.indices.flags.writeable = False
        self.batch_ptr.flags.writeable = False
        self.group_ptr.flags.writeable = False

    def _build(self, ptr, sizes):
        rng = np.random.default_rng(self.seed)
        shards = rng.permutation(len(sizes))
        batches = []
        groups = [0]
        window = self.settings["window_shards"]
        for start in range(0, len(shards), window):
            selected = shards[start : start + window]
            ids = np.concatenate([rng.permutation(np.arange(ptr[s], ptr[s + 1])) for s in selected])
            ns = np.repeat(sizes[selected], np.diff(ptr)[selected])
            # Largest-first best fit fills atom budgets with smaller geometries.
            order = np.argsort(-ns, kind="stable")
            capacity = len(ids)
            atoms = np.zeros(capacity, dtype=np.int64)
            elements = np.zeros(capacity, dtype=np.int64)
            counts = np.zeros(capacity, dtype=np.int64)
            local = []
            for position in order:
                n, idx = int(ns[position]), int(ids[position])
                count = len(local)
                feasible = (atoms[:count] + n <= self.settings["max_atoms"]) & (
                    counts[:count] < self.settings["batch_size"]
                )
                if self.settings["max_hessian_elements"] is not None:
                    feasible &= (
                        elements[:count] + 9 * n * n <= self.settings["max_hessian_elements"]
                    )
                candidates = np.flatnonzero(feasible)
                if len(candidates):
                    b = int(candidates[np.argmax(atoms[candidates])])
                else:
                    b = count
                    local.append([])
                local[b].append(idx)
                atoms[b] += n
                elements[b] += 9 * n * n
                counts[b] += 1
            batches.extend(local)
            groups.append(len(batches))
        dtype = np.int32 if ptr[-1] <= np.iinfo(np.int32).max else np.int64
        return (
            np.asarray([i for batch in batches for i in batch], dtype=dtype),
            np.r_[0, np.cumsum([len(b) for b in batches])].astype(dtype),
            np.asarray(groups, dtype=dtype),
        )

    def _validate(self, ptr, sizes):
        ids, bounds = self.indices, self.batch_ptr
        if (
            ids.ndim != 1
            or bounds.ndim != 1
            or len(bounds) < 2
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(bounds.dtype, np.integer)
            or bounds[0] != 0
            or bounds[-1] != len(ids)
            or np.any(np.diff(bounds) <= 0)
            or len(ids) != ptr[-1]
            or np.any(ids < 0)
            or np.any(ids >= ptr[-1])
            or np.any(np.bincount(ids, minlength=int(ptr[-1])) != 1)
        ):
            raise ValueError("Invalid cached batch coverage")
        ns = sizes[np.searchsorted(ptr, ids, side="right") - 1]
        if (
            np.any(np.diff(bounds) > self.settings["batch_size"])
            or np.any(np.add.reduceat(ns, bounds[:-1]) > self.settings["max_atoms"])
            or (
                self.settings["max_hessian_elements"] is not None
                and np.any(
                    np.add.reduceat(9 * ns * ns, bounds[:-1])
                    > self.settings["max_hessian_elements"]
                )
            )
        ):
            raise ValueError("Cached batch plan exceeds a budget")
        groups = self.group_ptr
        if (
            groups.ndim != 1
            or not np.issubdtype(groups.dtype, np.integer)
            or len(groups) < 2
            or groups[0] != 0
            or groups[-1] != len(bounds) - 1
            or np.any(np.diff(groups) <= 0)
        ):
            raise ValueError("Invalid cached locality groups")

    def save(self, path):
        """Write a compact plan atomically. No geometry data enters this file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
                temporary = f.name
                np.savez(
                    f,
                    indices=self.indices,
                    batch_ptr=self.batch_ptr,
                    group_ptr=self.group_ptr,
                    fingerprint=self.fingerprint,
                    settings=json.dumps(self.settings, sort_keys=True),
                )
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.batch_ptr) - 1

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.shuffle:
            order = range(len(self))
        elif self.settings["order"] == "global":
            order = rng.permutation(len(self))
        else:
            order = np.concatenate(
                [
                    rng.permutation(np.arange(self.group_ptr[g], self.group_ptr[g + 1]))
                    for g in rng.permutation(len(self.group_ptr) - 1)
                ]
            )
        for b in order:
            batch = self.indices[self.batch_ptr[b] : self.batch_ptr[b + 1]]
            yield (rng.permutation(batch) if self.shuffle else batch).tolist()
