"""Measure balanced packed training with identical geometry coverage."""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.benchmark_speed import head_options
from hip_orb.data import OrbMolDataset, make_loader
from hip_orb.execution import configure_execution
from hip_orb.model import HIPOrb
from hip_orb.packed_training import clear_device_indices, collate_packed_compact, packed_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path)
    parser.add_argument("--shards", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = json.loads(args.config.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = HIPOrb.pretrained("cuda", weights_path=args.weights, **head_options(config))
    configure_execution(model)
    model.train()
    # Same shuffled metadata cohort for every setting, with native source files.
    source = OrbMolDataset(args.index, training_only=True)
    selected = np.random.default_rng(7).permutation(len(source.paths))[: args.shards]
    counts = np.diff(source.ptr)[selected]
    source.paths = [source.paths[s] for s in selected]
    source.sources = [source.sources[s] for s in selected]
    source.natoms = source.natoms[selected]
    source.ptr = np.r_[0, np.cumsum(counts)]
    sys.path.insert(0, str((args.reference_package / "hip_orb").resolve()))
    reference_module = importlib.import_module("data")
    reference = reference_module.OrbMolDataset(args.index, training_only=True)
    for key in ("paths", "sources", "natoms", "ptr"):
        setattr(reference, key, getattr(source, key))
    report = {
        "complete": False,
        "samples": len(source),
        "config": config,
        "cases": [],
        "gpu": torch.cuda.get_device_name(),
        "capacity": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.report.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    def reset():
        model.head.load_state_dict(saved["head"])
        model.head.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0)
        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
        return opt

    # Test the largest supported molecule before streaming each atom budget.
    full = OrbMolDataset(args.index, training_only=True)
    shard = int(np.argmax(full.natoms))
    n = int(full.natoms[shard])
    safe = set()
    if args.capacity_report:
        report["capacity"] = json.loads(args.capacity_report.read_text())["capacity"]
        safe.update(
            r["max_atoms"]
            for r in report["capacity"]
            if r.get("success")
            and r["peak_allocated_bytes"] <= 0.9 * torch.cuda.get_device_properties(0).total_memory
        )
    for atoms in () if args.capacity_report else (2048, 2560, 3072, 4096):
        opt = reset()
        count = min(64, atoms // n, int(full.ptr[shard + 1] - full.ptr[shard]))
        record = {"max_atoms": atoms, "natoms": n, "geometries": count}
        try:
            batch = collate_packed_compact(
                full.__getitems__(list(range(int(full.ptr[shard]), int(full.ptr[shard]) + count)))
            )
            torch.cuda.reset_peak_memory_stats()
            packed_update(model, opt, batch)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            record.update(success=True, peak_allocated_bytes=peak)
            if peak <= 0.9 * torch.cuda.get_device_properties(0).total_memory:
                safe.add(atoms)
        except torch.cuda.OutOfMemoryError:
            record.update(success=False, error="CUDA out of memory")
        finally:
            report["capacity"].append(record)
            print(json.dumps({"capacity": record}), flush=True)
            model.head.zero_grad(set_to_none=True)
            del opt
            if "batch" in locals():
                del batch
            clear_device_indices()
            gc.collect()
            torch.cuda.empty_cache()
            save()
    full.close()
    # A final legacy control shows timing drift. No case changes a live run.
    cases = [
        ("legacy_global", 128, 4096, 4_000_000, "global", 2, 256, 2),
        ("global64", 64, 2048, None, "global", 2, 256, 2),
        ("local64", 64, 2048, None, "locality", 2, 256, 2),
        ("prefetch64", 64, 2048, None, "locality", 4, 1024, 4),
        ("workers8", 64, 2048, None, "locality", 8, 512, 4),
        ("batch32", 32, 2048, None, "locality", 4, 1024, 4),
        ("batch128", 128, 2048, None, "locality", 4, 1024, 4),
        ("atoms2560", 64, 2560, None, "locality", 4, 1024, 4),
        ("atoms3072", 64, 3072, None, "locality", 4, 1024, 4),
        ("atoms4096", 64, 4096, None, "locality", 4, 1024, 4),
        ("legacy_final", 128, 4096, 4_000_000, "global", 2, 256, 2),
    ]
    for label, cap, atoms, budget, order, workers, cache, prefetch in cases:
        if budget is None and atoms not in safe:
            report["cases"].append({"label": label, "skipped": "capacity margin"})
            save()
            continue
        dataset = reference if label.startswith("legacy") else source
        dataset.cache_bytes, dataset.max_open = cache << 20, 8 if label.startswith("legacy") else 16
        sampler = BalancedBatchSampler(
            source, max_atoms=atoms, batch_size=cap, max_hessian_elements=budget, order=order
        )
        plan = list(sampler)
        row = {
            "label": label,
            "batch_cap": cap,
            "max_atoms": atoms,
            "hessian_budget": budget,
            "order": order,
            "workers": workers,
            "cache_mib_per_worker": cache,
            "prefetch": prefetch,
            "batches": len(plan),
            "mean_batch": len(source) / len(plan),
            "repeats": [],
            "coverage_sha256": hashlib.sha256(np.sort(sampler.indices).tobytes()).hexdigest(),
        }
        for repeat in range(args.repeats):
            opt = reset()
            loader = make_loader(
                dataset,
                batch_sampler=plan,
                collate_fn=collate_packed_compact,
                workers=workers,
                prefetch_factor=prefetch,
            )
            iterator = iter(loader)
            batch = next(iterator)
            # Exclude worker startup and CUDA warmup. Include every timed source read.
            for _ in range(2):
                packed_update(model, opt, batch)
            opt = reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            wait = 0.0
            samples = 0
            try:
                for step in range(len(plan)):
                    if step:
                        t = time.perf_counter()
                        batch = next(iterator)
                        wait += time.perf_counter() - t
                    _, count = packed_update(model, opt, batch)
                    samples += count
                torch.cuda.synchronize()
                duration = time.perf_counter() - started
                assert samples == len(source)
                row["repeats"].append(
                    {
                        "seconds": duration,
                        "geometries_per_s": samples / duration,
                        "ms_per_update": 1000 * duration / len(plan),
                        "loader_wait_percent": 100 * wait / duration,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
            finally:
                iterator._shutdown_workers()
                model.head.zero_grad(set_to_none=True)
                del loader, iterator, opt, batch
                clear_device_indices()
                gc.collect()
                torch.cuda.empty_cache()
        row["geometries_per_s"] = float(np.mean([r["geometries_per_s"] for r in row["repeats"]]))
        row["loader_wait_percent"] = float(
            np.mean([r["loader_wait_percent"] for r in row["repeats"]])
        )
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        save()
    source.close()
    reference.close()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
