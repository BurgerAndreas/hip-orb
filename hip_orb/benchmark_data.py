"""CPU loader benchmarks on immutable real OrbMol shards; no model training."""
import argparse
import gc
import json
from pathlib import Path
import time
import tempfile

import h5py
import numpy as np
import torch

from hip_orb.data import FIELDS, OrbMolDataset, build_index, make_loader


def prepare_cache(paths, folder, backend):
    """Build isolated caches and verify bitwise equality with release arrays."""
    folder.mkdir(parents=True, exist_ok=False)
    outputs = []
    for path in paths:
        target = folder / (path.stem if backend == "mmap" else path.name)
        with h5py.File(path) as source:
            if backend == "mmap":
                target.mkdir()
                for field in (*FIELDS, "omol_index"):
                    values = source[field][:]
                    np.save(target / f"{field}.npy", values, allow_pickle=False)
                    restored = np.load(target / f"{field}.npy", mmap_mode="r")
                    if restored.tobytes() != values.tobytes():
                        raise ValueError("Memory-mapped cache differs")
            else:
                import hdf5plugin
                with h5py.File(target, "w") as output:
                    for key, value in source.attrs.items():
                        output.attrs[key] = value
                    for field in (*FIELDS, "omol_index", "natoms", "hessian_ptr"):
                        if field != "hessian_upper":
                            source.copy(field, output)
                            continue
                        values = source[field][:]
                        output.create_dataset(field, data=values, chunks=source[field].chunks,
                                              shuffle=True, **hdf5plugin.Zstd(clevel=3))
                with h5py.File(target) as output:
                    for field in (*FIELDS, "omol_index"):
                        if output[field][:].tobytes() != source[field][:].tobytes():
                            raise ValueError("Zstandard cache differs")
        outputs.append(target)
    return outputs


def timed_case(index, expected, config, epochs):
    dataset = OrbMolDataset(index, backend=config["backend"], cache_mode=config["cache"],
                            cache_bytes=256 << 20, max_open=8)
    loader = make_loader(dataset, workers=config["workers"], order=config["order"], pin_memory=False)
    results = []
    try:
        for epoch in range(epochs):
            loader.batch_sampler.set_epoch(epoch)
            started = time.perf_counter()
            ids, payload, batches, first_s = [], 0, 0, None
            for batch in loader:
                if first_s is None:
                    first_s = time.perf_counter() - started
                ids.extend(batch["omol_index"].tolist())
                payload += sum(value.numel() * value.element_size() for value in batch.values()
                               if isinstance(value, torch.Tensor))
                batches += 1
            elapsed = time.perf_counter() - started
            if not np.array_equal(np.sort(ids), expected):
                raise ValueError("Loader omitted or duplicated samples")
            results.append({"epoch": epoch, "samples": len(ids), "batches": batches,
                            "elapsed_s": elapsed, "first_batch_s": first_s,
                            "samples_per_s": len(ids) / elapsed,
                            "payload_mb_per_s": payload / elapsed / 1e6})
    finally:
        # Shut down workers between strategies, not between timed epochs.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        dataset.close()
        del loader
        gc.collect()
    warm = [row["samples_per_s"] for row in results[1:]]
    return {**config, "epochs": results, "warm_samples_per_s": float(np.median(warm))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mmap-local", type=Path, help="Optional node-local temporary cache parent")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shards-per-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 2 or args.shards_per_size < 1:
        raise ValueError("Use at least two epochs and one shard per size")
    torch.set_num_threads(1)
    with np.load(args.assignments, allow_pickle=False) as assignments:
        sizes = assignments["shard_natoms"]
    selected = set()
    for n in (2, 20, 50, 60, 70, 80, 100, 110, 160, 250, 350):
        nearest = int(sizes[np.argmin(np.abs(sizes - n))])
        eligible = np.flatnonzero(sizes == nearest)
        selected.update(int(eligible[i]) for i in np.linspace(0, len(eligible) - 1,
                                                             min(args.shards_per_size, len(eligible)), dtype=int))
    ids = sorted(selected)
    paths = [args.source / f"{i // 1000:03d}" / f"shard_{i:05d}.h5" for i in ids]
    expected_by_shard = []
    for path in paths:
        with h5py.File(path) as f:
            expected_by_shard.append(f["omol_index"][:])
    args.cache.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    zstd = prepare_cache(paths, args.cache / "zstd", "hdf5")
    zstd_s = time.perf_counter() - started
    started = time.perf_counter()
    local_cache = tempfile.TemporaryDirectory(prefix="hip-loader-", dir=args.mmap_local) if args.mmap_local else None
    mmap_root = Path(local_cache.name) / "mmap" if local_cache else args.cache / "mmap"
    mmap = prepare_cache(paths, mmap_root, "mmap")
    mmap_s = time.perf_counter() - started
    preparations = {"gzip": {"preparation_s": 0, "bytes": sum(p.stat().st_size for p in paths)},
                    "zstd": {"preparation_s": zstd_s, "bytes": sum(p.stat().st_size for p in zstd)},
                    "mmap": {"preparation_s": mmap_s, "bytes": sum(p.stat().st_size for p in mmap_root.rglob("*.npy")),
                             "location": "node_local_temporary" if local_cache else "same_cache_filesystem"}}
    configs = [dict(storage="gzip", backend="hdf5", cache="none", order="random", workers=0)]
    for cache in ("chunks", "arrays"):
        for order in ("random", "shard"):
            configs.append(dict(storage="gzip", backend="hdf5", cache=cache, order=order, workers=0))
    for workers in (2, 4, 8):
        configs.append(dict(storage="gzip", backend="hdf5", cache="arrays", order="shard", workers=workers))
    for storage, backend in (("zstd", "hdf5"), ("mmap", "mmap")):
        for workers in (0, 2, 4):
            configs.append(dict(storage=storage, backend=backend, cache="arrays", order="shard", workers=workers))
    profiles = {"50_70": (50, 70), "70_110": (70, 110), "2_350": (2, 350)}
    report = {"benchmark": "cache_preparation_only" if args.prepare_only else "cpu_loader_only", "bitwise_cache_verification": True,
              "warm_cache_note": "No OS cache eviction; warm results reflect a bounded representative working set.",
              "gpu_training_speedup_measured": False, "cache_bytes_per_worker": 256 << 20,
              "storage": preparations, "profiles": {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for name, (minimum, maximum) in profiles.items():
        positions = [i for i, shard in enumerate(ids) if minimum <= sizes[shard] <= maximum]
        expected = np.sort(np.concatenate([expected_by_shard[i] for i in positions]))
        indices = {}
        for storage, collection in (("gzip", paths), ("zstd", zstd)):
            index = args.cache / f"{name}_{storage}.npz"
            build_index([collection[i] for i in positions], index)
            indices[storage] = index
        with np.load(indices["gzip"], allow_pickle=False) as original:
            content = {key: original[key].copy() for key in original.files}
        content["paths"] = np.asarray([str(mmap[i]) for i in positions])
        indices["mmap"] = args.cache / f"{name}_mmap.npz"
        np.savez(indices["mmap"], **content)
        if args.prepare_only:
            report["profiles"][name] = {"samples": len(expected), "results": []}
            continue
        results = []
        # Complete the single-process comparisons before the longer worker sweep.
        order = sorted(rng.permutation(len(configs)), key=lambda i: configs[int(i)]["workers"])
        for i in order:
            config = configs[int(i)]
            result = timed_case(indices[config["storage"]], expected, config, args.epochs)
            results.append(result)
            report["profiles"][name] = {"samples": len(expected), "results": results}
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"profile": name, **{k:v for k,v in result.items() if k != "epochs"}}), flush=True)
    report["complete"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if local_cache:
        local_cache.cleanup()


if __name__ == "__main__":
    main()
