"""Test batch capacity and matched-cohort throughput without changing live runs."""

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hip_orb.benchmark_speed import head_options
from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.execution import configure_execution
from hip_orb.model import HIPOrb
from hip_orb.profile_batch_steps import read_pool
from hip_orb.profile_head import layer_ranges, summarize_trace
from hip_orb.runner import train_update
from hip_orb.schedules import learning_rate


def capacity(n, batch_size, limit):
    if min(n, batch_size, limit) <= 0:
        raise ValueError("Atom count, batch size, and element limit must be positive")
    return min(batch_size, max(1, limit // (3 * n) ** 2))


def plan_stats(natoms, counts, batch_size, limit):
    caps = np.asarray([capacity(int(n), batch_size, limit) for n in natoms])
    counts = np.asarray(counts)
    steps = (counts + caps - 1) // caps
    full = counts // caps
    return {
        "steps_per_epoch": int(steps.sum()),
        "samples": int(counts.sum()),
        "mean_batch": float(counts.sum() / steps.sum()),
        "budget_reduced_steps_fraction": float(steps[caps < batch_size].sum() / steps.sum()),
        "full_budget_steps_fraction": float(full[caps < batch_size].sum() / steps.sum()),
        "tail_steps_fraction": float(np.count_nonzero(counts % caps) / steps.sum()),
    }


def cohort_groups(dataset, shard_count, seed=7):
    rng = np.random.default_rng(seed)
    groups = []
    for shard in rng.permutation(len(dataset.paths))[:shard_count]:
        indices = np.arange(dataset.ptr[shard], dataset.ptr[shard + 1])
        rng.shuffle(indices)
        groups.append((int(dataset.natoms[shard]), indices.tolist()))
    return groups


def cohort_batches(groups, batch_size, limit):
    batches = []
    for n, indices in groups:
        cap = capacity(n, batch_size, limit)
        batches.extend(indices[start : start + cap] for start in range(0, len(indices), cap))
    return batches


def fingerprint(batches):
    return hashlib.sha256(
        np.asarray([i for batch in batches for i in batch], dtype=np.int64).tobytes()
    ).hexdigest()


def parse_setting(value):
    try:
        batch, limit = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use batch:limit, such as 256:8000000") from error
    if batch < 1 or limit < 1:
        raise argparse.ArgumentTypeError("Batch and limit must be positive")
    return batch, limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atoms", nargs="+", type=int, default=[20, 40, 70, 150, 250, 350])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=[2_000_000, 4_000_000, 5_000_000, 6_000_000, 8_000_000, 12_000_000],
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cohort-shards", type=int, default=48)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--capacity-report",
        type=Path,
        help="Reuse a completed fixed capacity test for cohort confirmation",
    )
    parser.add_argument(
        "--settings",
        type=parse_setting,
        nargs="+",
        help="Selected batch:limit pairs in execution order",
    )
    args = parser.parse_args()
    if args.smoke:
        args.atoms, args.batch_sizes, args.limits = [20, 350], [128], [4_000_000]
        args.steps, args.repeats, args.warmup, args.cohort_shards, args.profile_steps = (
            2,
            1,
            1,
            2,
            1,
        )
    if (
        min(
            args.steps,
            args.repeats,
            args.warmup,
            args.cohort_shards,
            *args.batch_sizes,
            *args.limits,
        )
        < 1
    ):
        parser.error("Benchmark sizes and counts must be positive")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    torch.manual_seed(7)
    dataset = OrbMolDataset(args.index, training_only=True)
    saved = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    model = HIPOrb.pretrained(
        "cuda", weights_path=args.weights, precision="float32-high", **head_options(config)
    )
    if model.head.config != saved["head_config"]:
        raise ValueError("Checkpoint architecture differs")
    configure_execution(model)
    model.train()
    original_hash = model_hash(model.base)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    report = {
        "complete": False,
        "config": config,
        "gpu": torch.cuda.get_device_name(),
        "total_memory_bytes": total_memory,
        "fixed": [],
        "cohort": [],
        "profiles": [],
        "feature_cache": False,
        "precision": "float32-high",
        "checkpoint_step": saved["global_step"],
        "implementation_root": str(Path(__file__).resolve().parent),
    }
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    report["implementation_sha256"] = digest.hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.report.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    def clear():
        model.head.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def optimizer():
        model.head.load_state_dict(saved["head"])
        model.head.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(model.head.parameters(), lr=config["lr"], weight_decay=0)
        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
        return opt

    def update(opt, batch, step):
        rate = learning_rate(
            saved["global_step"] + step + 1,
            config["lr"],
            config["lr_schedule"],
            config["max_steps"],
            config["warmup_fraction"],
        )
        for group in opt.param_groups:
            group["lr"] = rate
        return train_update(
            model,
            opt,
            batch,
            "cuda",
            config.get("seed", 7),
            loss_type=config["loss_type"],
            translation_scale=config.get("translation_scale", 0.0),
            consistency_weight=config.get("consistency_weight", 0.0),
            rotation_sampling=config.get("rotation_sampling", "uniform"),
            reference_views=config.get("reference_views", 1),
        )

    def fixed_trial(batch, profile=False):
        opt = optimizer()
        for step in range(args.warmup):
            update(opt, batch, step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            for step in range(args.steps):
                update(opt, batch, args.warmup + repeat * args.steps + step)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000 / args.steps)
        row = {
            "status": "ok",
            "ms_per_step": timings,
            "mean_ms_per_step": float(np.mean(timings)),
            "samples_per_s": len(batch["coords"]) * 1000 / float(np.mean(timings)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if profile:
            with (
                layer_ranges(model.head),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as trace,
            ):
                for step in range(args.profile_steps):
                    update(opt, batch, args.warmup + args.steps * args.repeats + step)
                    trace.step()
                torch.cuda.synchronize()
            row["profile"] = summarize_trace(trace)
        return row

    def cohort_trial(batches):
        results = []
        for repeat in range(args.repeats):
            opt = optimizer()
            # Warm the exact molecular shapes before measuring a fresh cohort.
            for step in range(args.warmup):
                update(opt, warm_batch, step)
            model.head.load_state_dict(saved["head"])
            opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
            loader = DataLoader(
                dataset,
                batch_sampler=batches,
                collate_fn=collate_samples,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
                multiprocessing_context="spawn",
                prefetch_factor=2,
            )
            iterator = iter(loader)
            try:
                # Separate worker startup from the measured source-reading path.
                current = next(iterator)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                wait_s, samples, loss_sum = 0.0, 0, 0.0
                for step in range(len(batches)):
                    if step:
                        wait_started = time.perf_counter()
                        current = next(iterator)
                        wait_s += time.perf_counter() - wait_started
                    loss, size = update(opt, current, step)
                    samples += size
                    loss_sum += loss * size
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                results.append(
                    {
                        "elapsed_s": elapsed,
                        "steps": len(batches),
                        "samples": samples,
                        "samples_per_s": samples / elapsed,
                        "mean_ms_per_step": elapsed * 1000 / len(batches),
                        "loader_wait_s": wait_s,
                        "sample_weighted_loss": loss_sum / samples,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
            finally:
                iterator._shutdown_workers()
        return {
            "status": "ok",
            "repeats": results,
            "samples_per_s": float(np.mean([r["samples_per_s"] for r in results])),
            "mean_ms_per_step": float(np.mean([r["mean_ms_per_step"] for r in results])),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in results),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in results),
        }

    try:
        sizes = sorted(
            {int(dataset.natoms[np.abs(dataset.natoms - n).argmin()]) for n in args.atoms}
        )
        pools = {}
        if args.capacity_report:
            prior = json.loads(args.capacity_report.read_text())
            if (
                not prior["complete"]
                or prior["config"] != config
                or prior["checkpoint_step"] != saved["global_step"]
            ):
                raise ValueError(
                    "Capacity report must be complete and match the configuration and checkpoint"
                )
            report["fixed"] = prior["fixed"]
            report["capacity_source"] = str(args.capacity_report)
            report["capacity_source_sha256"] = prior["implementation_sha256"]
            pools[sizes[0]], _ = read_pool(dataset, sizes[0], 64)
        for n in [] if args.capacity_report else sizes:
            caps = sorted(
                {capacity(n, batch, limit) for batch in args.batch_sizes for limit in args.limits}
            )
            pools[n], actual_n = read_pool(dataset, n, max(caps))
            assert actual_n == n
            failed_cap = None
            for cap in caps:
                row = {
                    "natoms": n,
                    "batch_size": cap,
                    "hessian_elements": cap * (3 * n) ** 2,
                    "fixed_source_reads_excluded": True,
                    "repeated_records": cap > len(pools[n]),
                }
                if failed_cap is not None:
                    row.update(status="skipped_after_oom", smaller_oom_batch=failed_cap)
                else:
                    batch = collate_samples([pools[n][i % len(pools[n])] for i in range(cap)])
                    try:
                        row.update(fixed_trial(batch))
                    except torch.cuda.OutOfMemoryError as error:
                        row.update(status="oom", error=str(error))
                        failed_cap = cap
                    del batch
                    clear()
                report["fixed"].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k != "error"}), flush=True)
        groups = cohort_groups(dataset, args.cohort_shards)
        report["cohort_samples"] = sum(len(indices) for _, indices in groups)
        report["cohort_atom_counts"] = [n for n, _ in groups]
        baseline = (128, 4_000_000)
        settings = [
            baseline,
            (64, 4_000_000),
            (128, 2_000_000),
            (256, 4_000_000),
            (512, 4_000_000),
            (256, 5_000_000),
            (128, 6_000_000),
            (256, 6_000_000),
            (256, 8_000_000),
            (512, 8_000_000),
            (256, 12_000_000),
            baseline,
        ]
        if args.smoke:
            settings = [baseline]
        if args.settings:
            settings = args.settings
        warm_n = sizes[0]
        warm_batch = collate_samples(pools[warm_n][: min(64, len(pools[warm_n]))])
        expected_fingerprint = None
        for case, (batch_size, limit) in enumerate(settings):
            row = {
                "case": case,
                "batch_size": batch_size,
                "limit": limit,
                "full_dataset_plan": plan_stats(
                    dataset.natoms, np.diff(dataset.ptr), batch_size, limit
                ),
            }
            # Keep at least 15% of device capacity free at every tested size.
            checks = [
                r
                for r in report["fixed"]
                if r["batch_size"] == capacity(r["natoms"], batch_size, limit)
            ]
            unsafe = [
                r
                for r in checks
                if r["status"] != "ok" or r["peak_allocated_bytes"] > 0.85 * total_memory
            ]
            missing_shapes = sorted(set(sizes) - {r["natoms"] for r in checks})
            if unsafe or missing_shapes:
                row["unmeasured_atom_counts"] = missing_shapes
                row.update(
                    status="skipped_capacity",
                    unsafe_shapes=[(r["natoms"], r["batch_size"], r["status"]) for r in unsafe],
                )
            else:
                batches = cohort_batches(groups, batch_size, limit)
                row["cohort_sha256"] = fingerprint(batches)
                if expected_fingerprint is None:
                    expected_fingerprint = row["cohort_sha256"]
                assert row["cohort_sha256"] == expected_fingerprint
                try:
                    row.update(cohort_trial(batches))
                except torch.cuda.OutOfMemoryError as error:
                    row.update(status="oom", error=str(error))
                clear()
            report["cohort"].append(row)
            save()
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("error", "repeats")}),
                flush=True,
            )
        for n in sorted({sizes[min(2, len(sizes) - 1)], sizes[-1]}) if args.profile_steps else []:
            safe = [
                r
                for r in report["fixed"]
                if r["natoms"] == n
                and r["status"] == "ok"
                and r["peak_allocated_bytes"] <= 0.85 * total_memory
            ]
            best = max(safe, key=lambda r: r["samples_per_s"])
            for label, cap in [
                ("baseline", capacity(n, 128, 4_000_000)),
                ("best", best["batch_size"]),
            ]:
                batch = collate_samples([pools[n][i % len(pools[n])] for i in range(cap)])
                row = {"natoms": n, "batch_size": cap, "label": label}
                row.update(fixed_trial(batch, profile=True))
                report["profiles"].append(row)
                del batch
                clear()
                save()
        report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
        report["backbone_has_gradients"] = any(p.grad is not None for p in model.base.parameters())
        assert report["frozen_backbone_unchanged"] and not report["backbone_has_gradients"]
        report["complete"] = True
    finally:
        dataset.close()
        save()


if __name__ == "__main__":
    main()
