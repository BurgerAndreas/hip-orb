"""Quick exact-batch training timings. Source reads stay outside step timing."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.block_layout import layout
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.model import HIPOrb


def timed(model, initial, samples, batch_size, steps, warmup):
    model.head.load_state_dict(initial)
    model.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
    # Reuse real records only if the bounded source cache has too few records.
    # This is a capacity test, not an epoch or a statistically independent batch.
    batch = collate_samples([samples[i % len(samples)] for i in range(batch_size)])
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        training_step(model, optimizer, batch)
    torch.cuda.synchronize()
    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        loss = training_step(model, optimizer, batch)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return {"status": "ok", "median_ms_per_step": float(np.median(durations)) * 1000,
            "mean_ms_per_step": float(np.mean(durations)) * 1000,
            "samples_per_s": batch_size / float(np.mean(durations)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "last_loss": float(loss), "seconds_per_step": durations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    initial = copy.deepcopy(model.head.state_dict())
    base_hash = model_hash(model.base)
    dataset = OrbMolDataset(args.index, training_only=True)
    with np.load(args.assignments, allow_pickle=False) as assignment:
        sizes = assignment["shard_natoms"]
        counts = np.diff(assignment["shard_ptr"])
    population = {int(n): int(counts[sizes == n].sum()) for n in np.unique(sizes)}
    report = {"complete": False, "full_training_started": False, "precision": "float32-high",
              "warmup_steps": args.warmup, "measured_steps": args.steps,
              "source_reads_and_collation_excluded": True, "frozen_features_cached": False,
              "timing_includes": "rotation, ASE preparation, GPU graphs, transfers, backbone, Hessian head, loss, backward, AdamW",
              "training_population_by_natoms": population, "cases": []}

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        chosen = sorted({int(dataset.natoms[np.abs(dataset.natoms - n).argmin()])
                         for n in (2, 20, 50, 70, 110, 160, 250, 350)})
        for n in chosen:
            samples = []
            started = time.perf_counter()
            for shard in np.flatnonzero(dataset.natoms == n):
                lo, hi = map(int, dataset.ptr[shard:shard + 2])
                samples.extend(dataset[i] for i in range(lo, min(hi, lo + 512 - len(samples))))
                if len(samples) >= 512:
                    break
            source_s = time.perf_counter() - started
            for batch_size in (64, 128, 256, 512):
                model.head.zero_grad(set_to_none=True)
                layout.cache_clear()
                model.head._pair_indices.clear()
                gc.collect()
                torch.cuda.empty_cache()
                row = {"natoms": n, "batch_size": batch_size,
                       "unique_records_used": min(batch_size, len(samples)),
                       "repeated_records": batch_size > len(samples), "source_pool_read_s": source_s}
                try:
                    row.update(timed(model, initial, samples, batch_size, args.steps, args.warmup))
                except torch.cuda.OutOfMemoryError:
                    row.update(status="out_of_memory", peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
            del samples
        report["frozen_backbone_unchanged"] = model_hash(model.base) == base_hash
        report["backbone_has_gradients"] = any(p.grad is not None for p in model.base.parameters())
        report["complete"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        dataset.close()
        save()


if __name__ == "__main__":
    main()
