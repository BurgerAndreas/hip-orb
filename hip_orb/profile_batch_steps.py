"""Bounded exact-batch profiling and asynchronous CPU preparation comparisons."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.block_layout import layout
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.graphs import canonical_message_graph, deterministic_graph_audit, refresh_message_vectors
from hip_orb.model import HIPOrb
from hip_orb.preparation import PreparedCollator, prepare_batch
from hip_orb.profile_head import layer_ranges, summarize_trace


class RepeatedPool(Dataset):
    """Reuse a bounded real pool, as in the batch-capacity benchmark."""
    def __init__(self, samples, batch_size, batches):
        self.samples, self.batch_size, self.batches = samples, batch_size, batches

    def __len__(self):
        return self.batch_size * self.batches

    def __getitem__(self, index):
        return self.samples[(index % self.batch_size) % len(self.samples)]


def read_pool(dataset, requested, limit):
    n = int(dataset.natoms[np.abs(dataset.natoms - requested).argmin()])
    samples = []
    for shard in np.flatnonzero(dataset.natoms == n):
        lo, hi = map(int, dataset.ptr[shard:shard + 2])
        samples.extend(dataset[i] for i in range(lo, min(hi, lo + limit - len(samples))))
        if len(samples) >= limit:
            break
    if not samples:
        raise ValueError("No records for requested size")
    return samples, n


def audit_graph(model, samples):
    raw = collate_samples(samples[:4])
    cpu_batch = prepare_batch(raw, adapter=model.adapter)
    cpu = refresh_message_vectors(cpu_batch["prepared_graph"].to("cuda"))
    gpu = refresh_message_vectors(canonical_message_graph(
        model.adapter.from_ase_atoms_list(cpu_batch["molecules"], device="cuda")))
    for name in ("senders", "receivers", "n_node"):
        torch.testing.assert_close(getattr(cpu, name), getattr(gpu, name), atol=0, rtol=0)
    torch.testing.assert_close(cpu.edge_features["vectors"], gpu.edge_features["vectors"], atol=0, rtol=0)
    sizes = torch.full((len(samples[:4]),), raw["coords"].shape[1], dtype=torch.long)
    pairs = [{k: v.to("cuda") for k, v in pair.items()} for pair in cpu_batch["prepared_pairs"]]
    with deterministic_graph_audit():
        reference = model.features(gpu, sizes=sizes)
        actual = model.features(cpu, sizes=sizes, pairs=pairs)
        model.head.zero_grad(set_to_none=True)
        expected = model.head(reference)
        sum(x.square().mean() for x in expected).backward()
        grads = [p.grad.clone() for p in model.head.parameters()]
        model.head.zero_grad(set_to_none=True)
        predicted = model.head(actual)
        for a, b in zip(predicted, expected, strict=True):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        sum(x.square().mean() for x in predicted).backward()
        for p, g in zip(model.head.parameters(), grads, strict=True):
            torch.testing.assert_close(p.grad, g, atol=1e-5, rtol=1e-4)
    model.head.zero_grad(set_to_none=True)
    return {"passed": True, "samples": len(samples[:4]), "predictions_and_gradients": True,
            "sampled_check_only": True}


def run_variant(model, initial, samples, batch_size, variant, args):
    model.head.load_state_dict(initial)
    model.head.zero_grad(set_to_none=True)
    model.train()
    layout.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
    raw = collate_samples([samples[i % len(samples)] for i in range(batch_size)])
    batch_identity = hashlib.sha256(raw["omol_index"].numpy().tobytes()).hexdigest()
    workers = 4 if variant.endswith("4") else 2
    loader = None
    torch.cuda.synchronize()
    startup_start = time.perf_counter()
    if variant in ("sync", "prebuilt_cpu"):
        batch = prepare_batch(raw, adapter=model.adapter) if variant == "prebuilt_cpu" else raw
        next_batch = lambda: batch
    else:
        loader = DataLoader(RepeatedPool(samples, batch_size, args.warmup + args.steps * args.repeats),
                            batch_size=batch_size, num_workers=workers, pin_memory=True,
                            collate_fn=PreparedCollator(adapter=model.adapter if variant.startswith("cpu_graph") else None),
                            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        iterator = iter(loader)
        next_batch = lambda: next(iterator)
    try:
        for _ in range(args.warmup):
            training_step(model, optimizer, next_batch())
        torch.cuda.synchronize()
        startup_s = time.perf_counter() - startup_start
        torch.cuda.reset_peak_memory_stats()
        repeats, wait_s = [], 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(args.steps):
                wait_start = time.perf_counter()
                batch = next_batch()
                wait_s += time.perf_counter() - wait_start
                loss = training_step(model, optimizer, batch)
            torch.cuda.synchronize()
            repeats.append((time.perf_counter() - start) * 1000 / args.steps)
        row = {"variant": variant, "ms_per_step_runs": repeats,
               "median_ms_per_step": float(np.median(repeats)), "last_loss": float(loss),
               "samples_per_s": batch_size * 1000 / float(np.mean(repeats)),
               "main_loader_wait_ms_per_step": wait_s * 1000 / (args.steps * args.repeats),
               "startup_and_warmup_s": startup_s, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "batch_identity_sha256": batch_identity,
               "preparation_excluded": variant == "prebuilt_cpu",
               "collation_and_worker_transfer_included": loader is not None}
        # Profiles run separately. Reuse workers after the uninstrumented timing.
        if loader is not None:
            iterator = iter(loader)
        with layer_ranges(model.head), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as trace:
            for _ in range(args.profile_steps):
                with torch.profiler.record_function("hip_loader_wait"):
                    batch = next_batch()
                training_step(model, optimizer, batch)
                trace.step()
            torch.cuda.synchronize()
        row["profile"] = {"steps": args.profile_steps, **summarize_trace(trace)}
        row["profile"]["stage_ranges"] = [
            {"name": event.key, "device": str(event.device_type), "calls": event.count,
             "cpu_total_ms": event.cpu_time_total / 1000,
             "gpu_total_ms": event.device_time_total / 1000}
            for event in trace.key_averages() if event.key.startswith("hip_")]
        if args.export_traces:
            trace.export_chrome_trace(str(args.report.with_name(
                f"trace_{batch['coords'].shape[1]}_{batch_size}_{variant}_private.json")))
        return row
    finally:
        if loader is not None and loader._iterator is not None:
            loader._iterator._shutdown_workers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["20:128", "50:128", "110:64", "110:128", "250:64", "350:64"])
    parser.add_argument("--variants", nargs="+", choices=["sync", "prebuilt_cpu", "cpu_prepare2", "cpu_graph2", "cpu_graph4"],
                        default=["sync", "prebuilt_cpu", "cpu_prepare2", "cpu_graph2", "cpu_graph4"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--export-traces", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup, args.profile_steps) < 1:
        raise ValueError("Use positive step counts")
    if args.profile_steps > args.warmup + args.steps * args.repeats:
        raise ValueError("Profile exceeds bounded loader length")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    initial = copy.deepcopy(model.head.state_dict())
    original_hash = model_hash(model.base)
    report = {"complete": False, "full_training_started": False, "cases": [],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "precision": torch.get_float32_matmul_precision(), "feature_cache": False,
              "source_reads_excluded": True, "steps_per_repeat": args.steps, "repeats": args.repeats,
              "warmup_steps": args.warmup,
              "note": "Capacity diagnostic reuses the same real batch. Async timings include collation, IPC and loader waits. Sync matches the source-excluded benchmark. Prebuilt CPU graphs bound the benefit of preparation; they are not end-to-end training. Profiler ranges overlap; do not sum inclusive times. Worker execution is outside the main-process trace."}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    dataset = OrbMolDataset(args.index, training_only=True)
    save()
    try:
        for case in args.cases:
            requested, batch_size = map(int, case.split(":"))
            if min(requested, batch_size) < 1:
                raise ValueError("Use positive atom and batch sizes")
            samples, n = read_pool(dataset, requested, batch_size)
            row = {"natoms": n, "batch_size": batch_size, "unique_records": min(batch_size, len(samples)),
                   "repeated_records": len(samples) < batch_size, "results": []}
            try:
                row["cpu_graph_audit"] = audit_graph(model, samples)
            except (AssertionError, RuntimeError, NotImplementedError) as error:
                row["cpu_graph_audit"] = {"passed": False, "reason": str(error)}
            report["cases"].append(row)
            save()
            for variant in args.variants:
                if not row["cpu_graph_audit"]["passed"] and (variant.startswith("cpu_graph") or variant == "prebuilt_cpu"):
                    continue
                result = run_variant(model, initial, samples, batch_size, variant, args)
                row["results"].append(result)
                save()
                print(json.dumps({"natoms": n, "batch_size": batch_size,
                                  **{k: v for k, v in result.items() if k != "profile"}}), flush=True)
            identities = {r["batch_identity_sha256"] for r in row["results"]}
            if len(identities) != 1:
                raise RuntimeError("Variants used different batches")
        report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
        report["backbone_has_gradients"] = any(p.grad is not None for p in model.base.parameters())
        if not report["frozen_backbone_unchanged"] or report["backbone_has_gradients"]:
            raise RuntimeError("Frozen backbone changed")
        report["complete"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        dataset.close()
        save()


if __name__ == "__main__":
    main()
