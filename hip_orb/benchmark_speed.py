"""Measure complete updates on bounded real batches and export separate profiles.

Fixed-batch diagnostics exclude source reads. Streaming measurements use the
production sampler, include source reads, and never cache backbone features.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, collate_samples, make_loader
from hip_orb.model import HIPOrb
from hip_orb.profile_batch_steps import read_pool
from hip_orb.profile_head import summarize_trace
from hip_orb.runner import train_update
from hip_orb.schedules import learning_rate


def head_options(config):
    names = ("hidden", "radial_features", "radial_cutoff", "radial_bases",
             "pair_residual_layers", "neighbor_pooling", "extra_orb_round",
             "translation_projection", "context_rounds", "mlp_hidden", "all_to_all",
             "orb_initialization", "orb_message_graph", "orb_rounds", "orb_combine_backbone_edges",
             "orb_feature_adapters", "radial_log_range", "shared_geometry", "backbone_layer_mix",
             "checkpoint_context")
    options = {name: config[name] for name in names if name in config}
    options["cutoff"] = config.get("hessian_cutoff")
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atoms", type=int, nargs="+", default=[20, 70, 150, 350])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--candidate", default="eager")
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--precision", choices=("float32-high", "float32-highest"), default="float32-high")
    args = parser.parse_args()
    if args.head_checkpoint is not None and len(args.configs) != 1:
        parser.error("A head checkpoint requires exactly one model configuration")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    report = {"complete": False, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cases": [], "streaming": args.streaming,
              "feature_cache": False, "fixed_batch_source_reads_excluded": not args.streaming,
              "candidate": args.candidate}
    report["cuda_allocator_config"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    report["implementation_root"] = str(Path(__file__).resolve().parent)
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    report["implementation_sha256"] = digest.hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    base = adapter = None
    dataset = OrbMolDataset(args.index, training_only=True)
    candidates = ExitStack()
    try:
        for config_path in args.configs:
            config = json.loads(config_path.read_text())
            torch.manual_seed(7)
            if base is None:
                model = HIPOrb.pretrained("cuda", weights_path=args.weights, precision=args.precision,
                                          **head_options(config))
                base, adapter = model.base, model.adapter
                original_hash = model_hash(base)
            else:
                model = HIPOrb(base, adapter, **head_options(config))
            model.train()
            from hip_orb.acceleration import speed_candidate
            candidates.enter_context(speed_candidate(model, args.candidate))
            saved = None
            if args.head_checkpoint is not None:
                saved = torch.load(args.head_checkpoint, map_location="cuda", weights_only=True)
                if saved["head_config"] != model.head.config:
                    raise ValueError("Checkpoint architecture differs from the benchmark configuration")
                model.head.load_state_dict(saved["head"])
            initial = copy.deepcopy(model.head.state_dict())
            atom_cases = [None] if args.streaming else args.atoms
            for requested in atom_cases:
                model.head.load_state_dict(initial)
                optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
                if saved is not None and "optimizer" in saved:
                    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                start_step = saved.get("global_step", 0) if saved is not None else 0
                step = 0

                def update(current, start_step=start_step, config=config, optimizer=optimizer, model=model):
                    nonlocal step
                    step += 1
                    rate = learning_rate(start_step + step, config.get("lr", 1e-3),
                                         config.get("lr_schedule", "constant"),
                                         config.get("max_steps"), config.get("warmup_fraction", 0.15))
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    return train_update(model, optimizer, current, "cuda", 7,
                                        loss_type=config["loss_type"])
                loader = None
                case_start = time.perf_counter()
                if args.streaming:
                    loader = make_loader(dataset, workers=2, batch_size=config["batch_size"],
                                         max_hessian_elements=config["max_hessian_elements"],
                                         seed=7, pin_memory=True,
                                         max_batches=args.warmup + args.steps * args.repeats + args.profile_steps)
                    iterator = iter(loader)
                    next_batch = lambda iterator=iterator: next(iterator)
                    n = batch_size = None
                else:
                    samples, n = read_pool(dataset, requested, config["batch_size"])
                    batch_size = min(config["batch_size"], max(1, config["max_hessian_elements"] // (3*n)**2))
                    batch = collate_samples([samples[i % len(samples)] for i in range(batch_size)])
                    next_batch = lambda batch=batch: batch
                try:
                    for _ in range(args.warmup):
                        update(next_batch())
                    torch.cuda.synchronize()
                    startup_and_warmup = time.perf_counter() - case_start
                    torch.cuda.reset_peak_memory_stats()
                    timings, observed_shapes = [], []
                    identity = hashlib.sha256()
                    loss_sum, samples_seen = 0.0, 0
                    loader_wait_s, shape_timings = 0.0, {}
                    for repeat in range(args.repeats):
                        started = time.perf_counter()
                        for _ in range(args.steps):
                            batch_started = time.perf_counter()
                            current = next_batch()
                            waited = time.perf_counter() - batch_started
                            loader_wait_s += waited
                            loss, size = update(current)
                            update_s = time.perf_counter() - batch_started - waited
                            shape = shape_timings.setdefault(str(current["coords"].shape[1]),
                                {"steps": 0, "samples": 0, "loader_wait_s": 0., "update_s": 0.})
                            shape["steps"] += 1
                            shape["samples"] += size
                            shape["loader_wait_s"] += waited
                            shape["update_s"] += update_s
                            observed_shapes.append([size, current["coords"].shape[1]])
                            identity.update(current["omol_index"].numpy().tobytes())
                            identity.update(json.dumps(current["source_dataset"]).encode())
                            loss_sum += loss * size
                            samples_seen += size
                        torch.cuda.synchronize()
                        timings.append((time.perf_counter() - started) * 1000 / args.steps)
                    row = {"config": config_path.name, "natoms": n, "batch_size": batch_size,
                           "ms_per_step": timings, "mean_ms_per_step": sum(timings)/len(timings),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "last_loss": loss, "observed_shapes": observed_shapes,
                           "sample_sequence_sha256": identity.hexdigest(),
                           "samples": samples_seen, "sample_weighted_loss": loss_sum / samples_seen,
                           "lr_schedule": config.get("lr_schedule", "constant"),
                           "checkpoint_step": start_step}
                    row["startup_and_warmup_s"] = startup_and_warmup
                    row["main_loader_wait_ms_per_step"] = loader_wait_s * 1000 / (args.steps * args.repeats)
                    row["shape_timings"] = shape_timings
                    row["matmul_precision"] = torch.get_float32_matmul_precision()
                    row["execution_config"] = getattr(model, "execution_config", None)
                    memory = torch.cuda.memory_stats()
                    row["allocator"] = {key: memory[key] for key in (
                        "num_alloc_retries", "num_ooms", "inactive_split_bytes.all.peak") if key in memory}
                    report["cases"].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if k != "observed_shapes"}), flush=True)
                    if args.profile_steps:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True) as trace:
                            for _ in range(args.profile_steps):
                                update(next_batch())
                                trace.step()
                            torch.cuda.synchronize()
                        row["profile"] = summarize_trace(trace)
                        trace.export_chrome_trace(str(args.report.parent / f"{args.candidate}_{config_path.stem}_{n}_trace.json"))
                        save()
                finally:
                    if loader is not None and getattr(loader, "_iterator", None) is not None:
                        loader._iterator._shutdown_workers()
                    model.head.zero_grad(set_to_none=True)
                    del optimizer
                torch.cuda.empty_cache()
            candidates.close()
            del model, initial
            gc.collect()
            torch.cuda.empty_cache()
        report["frozen_backbone_unchanged"] = model_hash(base) == original_hash
        report["backbone_has_gradients"] = any(p.grad is not None for p in base.parameters())
        if not report["frozen_backbone_unchanged"] or report["backbone_has_gradients"]:
            raise RuntimeError("Frozen backbone changed during the speed benchmark")
        report["complete"] = True
    finally:
        candidates.close()
        dataset.close()
        save()


if __name__ == "__main__":
    main()
