"""Bounded real-data profiling; separate complete steps from cached-feature diagnostics."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.data import OrbMolDataset, make_loader
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.model import HIPOrb
from hip_orb.preparation import PreparedCollator


@contextmanager
def layer_ranges(head):
    handles, opened = [], {}
    names = {"node_head.0": "node_projection", "pair_adapter.0": "pair_input_projection",
             "pair_adapter.2": "pair_hidden_1", "pair_adapter.4": "pair_hidden_2",
             "edge_head": "pair_block_readout"}
    for name, module in head.named_modules():
        if name not in names:
            continue

        def before(module, inputs, label=names[name]):
            scope = torch.profiler.record_function("head_" + label)
            scope.__enter__()
            opened.setdefault(module, []).append(scope)

        def after(module, inputs, output):
            opened[module].pop().__exit__(None, None, None)

        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for scopes in opened.values():
            for scope in scopes:
                scope.__exit__(None, None, None)


def dataset_for_size(index, requested):
    dataset = OrbMolDataset(index, training_only=True)
    n = int(dataset.natoms[np.abs(dataset.natoms - requested).argmin()])
    selected = np.flatnonzero(dataset.natoms == n)
    counts = np.diff(dataset.ptr)[selected]
    dataset.paths = [dataset.paths[i] for i in selected]
    dataset.sources = [dataset.sources[i] for i in selected]
    dataset.natoms = dataset.natoms[selected]
    dataset.ptr = np.r_[0, np.cumsum(counts)]
    return dataset, n


def summarize_trace(trace):
    operations = [{"name": event.key, "device": str(event.device_type), "calls": event.count,
                   "shapes": event.input_shapes, "cpu_self_ms": event.self_cpu_time_total / 1000,
                   "gpu_self_ms": event.self_device_time_total / 1000,
                   "gpu_total_ms": event.device_time_total / 1000,
                   "self_device_memory_bytes": event.self_device_memory_usage}
                  for event in trace.key_averages(group_by_input_shape=True)]
    # Annotation ranges overlap kernels. Do not rank them as kernel self time.
    kernels = [event for event in operations if event["device"] == "DeviceType.CPU" and
               not event["name"].startswith(("head_", "hip_", "Optimizer"))]
    return {"top_gpu_operations": sorted(kernels, key=lambda event: event["gpu_self_ms"], reverse=True)[:25],
            "top_cpu_operations": sorted(operations, key=lambda event: event["cpu_self_ms"], reverse=True)[:25],
            "scopes": [event for event in operations if event["name"].startswith(("head_", "hip_"))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("Use positive measured step counts")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    original_hash = model_hash(model.base)
    initial_head = copy.deepcopy(model.head.state_dict())
    report = {"complete": False, "full_training_started": False, "cases": [],
              "note": "Complete-step timings include real loader waits, workers, graph construction and optimizer updates. Detailed traces reuse frozen features only to isolate head costs."}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for requested in (50, 70, 110, 250, 350):
        model.head.load_state_dict(initial_head)
        dataset, n = dataset_for_size(args.index, requested)
        cap = min(16, max(1, 2_000_000 // (3*n)**2))
        available = sum((int(count) + cap - 1) // cap for count in np.diff(dataset.ptr))
        warmup = min(10, max(1, available // 3))
        steps = min(args.steps, available - warmup)
        if steps < 1:
            raise ValueError("Require at least two batches per profiling size")
        loader = make_loader(dataset, workers=2, batch_size=16, max_batches=warmup + steps,
                             collate_fn=PreparedCollator(), pin_memory=True)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
        model.train()
        try:
            start = time.perf_counter()
            iterator = iter(loader)
            for _ in range(warmup):
                training_step(model, optimizer, next(iterator))
            torch.cuda.synchronize()
            startup = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            samples = 0
            for _ in range(steps):
                batch = next(iterator)
                training_step(model, optimizer, batch)
                samples += len(batch["coords"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            case = {"requested_natoms": requested, "natoms": n, "workers": 2,
                    "steps": steps, "warmup_steps": warmup, "samples": samples,
                    "complete_ms_per_step": elapsed * 1000 / steps,
                    "samples_per_s": samples / elapsed, "startup_and_warmup_s": startup,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
            try:
                next(iterator)
                raise RuntimeError("Bounded loader returned extra batches")
            except StopIteration:
                pass
            # Reuse persistent workers and their caches. Profile real training,
            # separately from the uninstrumented timing and cached-feature trace.
            iterator = iter(loader)
            diagnostic_steps = min(3, warmup + steps)
            with layer_ranges(model.head), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True) as complete_trace:
                for _ in range(diagnostic_steps):
                    with torch.profiler.record_function("hip_loader_wait"):
                        batch = next(iterator)
                    training_step(model, optimizer, batch)
                    complete_trace.step()
                torch.cuda.synchronize()
            case["complete_step_profile"] = {"steps": diagnostic_steps, "feature_cache": False,
                                              **summarize_trace(complete_trace)}
            complete_trace.export_chrome_trace(str(args.report.with_name(
                f"{args.report.stem}_{n}_complete_private_trace.json")))
            # Drain the bounded iterator before shutting down persistent workers.
            for unused_batch in iterator:
                pass
            del complete_trace
            graph = refresh_message_vectors(canonical_message_graph(
                model.adapter.from_ase_atoms_list(batch["molecules"], device="cuda")))
            features = model.features(graph, sizes=torch.full((len(batch["coords"]),), n, dtype=torch.long))
            case["profile_batch_size"] = len(batch["coords"])
            case["directed_hessian_pairs"] = len(batch["coords"]) * n * (n - 1)
            case["message_edges"] = len(features.edges)
            case["latent_dim"] = model.head.config["latent_dim"]
            case["hidden_dim"] = model.head.config["hidden"]
            # Warm head-only execution before recording a diagnostic trace.
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                sum(h.square().mean() for h in model.head(features)).backward()
                optimizer.step()
            torch.cuda.synchronize()
            with layer_ranges(model.head), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True) as trace:
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.profiler.record_function("head_forward_dense_coulomb_loss"):
                        prediction = model.head(features)
                        loss = sum(h.square().mean() for h in prediction)
                    with torch.profiler.record_function("head_backward_only"):
                        loss.backward()
                    with torch.profiler.record_function("head_optimizer_only"):
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
                        optimizer.step()
                    trace.step()
                torch.cuda.synchronize()
            case.update(summarize_trace(trace))
            trace.export_chrome_trace(str(args.report.with_name(f"{args.report.stem}_{n}_private_trace.json")))
            report["cases"].append(case)
            save()
            print(json.dumps({key: value for key, value in case.items() if key not in
                              ("top_gpu_operations", "top_cpu_operations", "scopes", "complete_step_profile")}), flush=True)
        finally:
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator._shutdown_workers()
            dataset.close()
        del features, graph, optimizer, loader, iterator, batch, prediction, loss, trace
        torch.cuda.empty_cache()
    report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
    if not report["frozen_backbone_unchanged"]:
        raise RuntimeError("Frozen backbone changed")
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
