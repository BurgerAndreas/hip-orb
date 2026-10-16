"""Compare bounded streaming training runs with the production runner update."""
import argparse
import copy
import gc
import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np
import torch

from hip_orb import ORB_COMMIT, ORB_VERSION
from hip_orb.benchmark_training import model_hash
from hip_orb.block_layout import layout
from hip_orb.data import AtomBatchSampler, OrbMolDataset, make_loader
from hip_orb.model import HIPOrb
from hip_orb.preparation import PreparedCollator
from hip_orb.profile_batch_steps import audit_graph, read_pool
from hip_orb.profile_head import layer_ranges, summarize_trace
from hip_orb.runner import train_update, validate
from hip_orb.compilation import compiler_candidate
from hip_orb.schedules import learning_rate


VARIANTS = {"runner2": (2, "sync"), "runner4": (4, "sync"),
            "prepare2": (2, "cpu_prepare"), "prepare4": (4, "cpu_prepare"),
            "graph2": (2, "cpu_graph"), "deferred2": (2, "sync"),
            "shape2": (2, "sync"), "compile2": (2, "sync"), "fusion2": (2, "sync"), "compile_fusion2": (2, "sync"),
            "triton2": (2, "sync"), "compile_triton2": (2, "sync")}


def restrict_atoms(dataset, minimum, maximum):
    selected = np.flatnonzero((dataset.natoms >= minimum) & (dataset.natoms <= maximum))
    if not len(selected):
        raise ValueError("No source shards in the configured atom range")
    counts = np.diff(dataset.ptr)[selected]
    dataset.paths = [dataset.paths[i] for i in selected]
    dataset.sources = [dataset.sources[i] for i in selected]
    dataset.natoms = dataset.natoms[selected]
    dataset.ptr = np.r_[0, np.cumsum(counts)]


def planned_batches(dataset, *, batch_size, budget, seed, count):
    """Check real sampler rows without reading or preloading source payloads."""
    sampler = AtomBatchSampler(dataset, batch_size=batch_size,
                               max_hessian_elements=budget, seed=seed)
    batches = list(islice(sampler, count))
    if len(batches) != count:
        raise ValueError("Not enough fresh source batches for this bounded run")
    rows = [row for batch in batches for row in batch]
    if len(set(rows)) != len(rows):
        raise ValueError("The sampler repeated a record within one epoch")
    return batches


def batch_identity(batch, digest):
    # Include source identity and batch boundaries, not only molecule IDs.
    shape = batch["coords"].shape if "coords" in batch else batch["shape"]
    digest.update(np.asarray([shape[0], shape[1]], dtype=np.int64).tobytes())
    for source, identity in zip(batch["source_dataset"], batch["omol_index"].tolist(), strict=True):
        digest.update(json.dumps([source, identity], separators=(",", ":")).encode() + b"\n")


def shutdown(loader, dataset):
    if loader is not None and getattr(loader, "_iterator", None) is not None:
        loader._iterator._shutdown_workers()
    if dataset is not None:
        dataset.close()


def profile_continuation(model, optimizer, iterator, seed, steps, directory, *, scalar_loss=True,
                         loss_type="mse", set_lr=None, start_step=0):
    """Trace new batches before validation pauses the training pipeline."""
    if not steps:
        return None
    measured_head = copy.deepcopy(model.head.state_dict())
    try:
        with layer_ranges(model.head), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as trace:
            for step in range(steps):
                with torch.profiler.record_function("hip_loader_wait"):
                    batch = next(iterator)
                if set_lr is not None:
                    set_lr(start_step + step + 1)
                train_update(model, optimizer, batch, "cuda", seed, scalar_loss=scalar_loss,
                             loss_type=loss_type)
                trace.step()
            torch.cuda.synchronize()
        result = {"steps": steps, **summarize_trace(trace)}
        result["stage_ranges"] = [
            {"name": event.key, "device": str(event.device_type), "calls": event.count,
             "cpu_total_ms": event.cpu_time_total / 1000, "gpu_total_ms": event.device_time_total / 1000}
            for event in trace.key_averages() if event.key.startswith("hip_")]
        trace.export_chrome_trace(str(directory / "private_trace.json"))
        return result
    finally:
        # Validation and the saved checkpoint use the uninstrumented head state.
        # This optimizer is discarded after validation; no updates follow it.
        model.head.load_state_dict(measured_head)
        model.head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()


def run_case(model, initial, variant, repeat, args):
    workers, preparation = VARIANTS[variant]
    model.head.zero_grad(set_to_none=True)
    model.head.load_state_dict(initial)
    model.train()
    layout.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0)
    seed = args.seed + repeat
    scalar_loss = variant != "deferred2"
    loader = validation_loader = None
    dataset = validation_data = None
    case_start = time.perf_counter()
    try:
        dataset = OrbMolDataset(args.train_index, training_only=True)
        restrict_atoms(dataset, args.min_atoms, args.max_atoms)
        total = args.warmup + args.steps + args.profile_steps
        plan_start = time.perf_counter()
        plan = planned_batches(dataset, batch_size=args.batch_size, budget=args.max_hessian_elements,
                               seed=seed, count=total)
        plan_checks_s = time.perf_counter() - plan_start
        # Exclude the benchmark's duplicate row-plan audit from production timing.
        case_start += plan_checks_s
        kwargs = {} if preparation == "sync" else {"collate_fn": PreparedCollator(
            seed=seed, adapter=model.adapter if preparation == "cpu_graph" else None)}
        loader = make_loader(dataset, workers=workers, batch_size=args.batch_size,
                             max_hessian_elements=args.max_hessian_elements,
                             seed=seed, max_batches=total, pin_memory=True, **kwargs)
        iterator = iter(loader)
        captured_batches = []

        def capture(batch):
            # Retain only small CPU metadata. Hash it after timing finishes.
            captured_batches.append({"shape": batch["coords"].shape,
                "source_dataset": batch["source_dataset"], "omol_index": batch["omol_index"]})

        warmup_samples = 0
        def set_lr(step):
            rate = learning_rate(step, args.lr, getattr(args, "lr_schedule", "constant"),
                                 getattr(args, "schedule_steps", None),
                                 getattr(args, "warmup_fraction", 0.15))
            for group in optimizer.param_groups:
                group["lr"] = rate

        for step in range(args.warmup):
            batch = next(iterator)
            capture(batch)
            set_lr(step + 1)
            loss, size = train_update(model, optimizer, batch, "cuda", seed, scalar_loss=scalar_loss,
                                      loss_type=args.loss_type)
            warmup_samples += size
        torch.cuda.synchronize()
        startup_and_warmup_s = time.perf_counter() - case_start
        torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        loader_wait_s, samples = 0.0, 0
        atom_counts, batch_sizes = {}, []
        loss_sum = 0.0
        for step in range(args.steps):
            waited = time.perf_counter()
            batch = next(iterator)
            loader_wait_s += time.perf_counter() - waited
            capture(batch)
            set_lr(args.warmup + step + 1)
            loss, size = train_update(model, optimizer, batch, "cuda", seed, scalar_loss=scalar_loss,
                                      loss_type=args.loss_type)
            samples += size
            loss_sum += (loss.double() if torch.is_tensor(loss) else loss) * size
            n = str(batch["coords"].shape[1])
            atom_counts[n] = atom_counts.get(n, 0) + size
            batch_sizes.append(size)
        torch.cuda.synchronize()
        last_loss = float(loss)
        mean_loss = float(loss_sum) / samples
        training_s = time.perf_counter() - measured_start
        training_with_startup_s = time.perf_counter() - case_start
        peak = torch.cuda.max_memory_allocated()
        identity_start = time.perf_counter()
        identity = hashlib.sha256()
        for captured in captured_batches:
            batch_identity(captured, identity)
        identity_checks_s = time.perf_counter() - identity_start
        checkpoint_start = time.perf_counter()
        directory = args.report.parent / f"{variant}_{repeat}"
        directory.mkdir(exist_ok=True)
        checkpoint = {"head": model.head.state_dict(), "head_config": model.head.config,
                      "optimizer": optimizer.state_dict(), "global_step": args.warmup + args.steps,
                      "loss_type": args.loss_type,
                      "orb_commit": ORB_COMMIT, "orb_version": ORB_VERSION,
                      "precision": "float32-high", "teacher": "orbmol_v2"}
        torch.save(checkpoint, directory / "head.tmp.pt")
        (directory / "head.tmp.pt").replace(directory / "head.pt")
        checkpoint_s = time.perf_counter() - checkpoint_start
        profile = profile_continuation(model, optimizer, iterator, seed, args.profile_steps, directory,
                                       scalar_loss=scalar_loss, loss_type=args.loss_type, set_lr=set_lr,
                                       start_step=args.warmup + args.steps)
        validation_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        validation_data = OrbMolDataset(args.validation_index, training_only=True)
        restrict_atoms(validation_data, args.min_atoms, args.max_atoms)
        if set(dataset.sources) & set(validation_data.sources) or set(dataset.paths) & set(validation_data.paths):
            raise ValueError("Training and validation sources overlap")
        validation_loader = make_loader(validation_data, workers=2, seed=seed, pin_memory=True,
            batch_size=args.validation_batch_size,
            max_hessian_elements=args.validation_max_hessian_elements,
            max_samples=args.validation_samples)
        validated = validate(model, validation_loader, "cuda", report_samples=0)
        torch.cuda.synchronize()
        shutdown(validation_loader, validation_data)
        validation_s = time.perf_counter() - validation_start
        validation_loader = validation_data = None
        row = {"variant": variant, "repeat": repeat, "workers": workers, "preparation": preparation,
               "seed": seed, "steps": args.steps, "samples": samples, "warmup_samples": warmup_samples,
               "loss_reporting": "per_step" if scalar_loss else "epoch",
               "loss_type": args.loss_type,
               "streaming_ms_per_step": training_s * 1000 / args.steps,
               "streaming_samples_per_s": samples / training_s,
               "training_s": training_s, "startup_and_warmup_s": startup_and_warmup_s,
               "benchmark_row_plan_checks_s_excluded": plan_checks_s,
               "benchmark_identity_checks_s_excluded": identity_checks_s,
               "training_with_startup_s": training_with_startup_s,
               "training_samples_per_s_including_startup": (samples + warmup_samples) / training_with_startup_s,
               "main_loader_wait_ms_per_step": loader_wait_s * 1000 / args.steps,
               "samples_by_natoms": atom_counts, "actual_batch_sizes": batch_sizes,
               "last_loss": last_loss, "sample_weighted_train_loss": mean_loss,
               "sample_sequence_sha256": identity.hexdigest(),
               "sampler_rows_sha256": hashlib.sha256(np.asarray(
                   [row for batch in plan[:args.warmup + args.steps] for row in batch], dtype=np.int64).tobytes()).hexdigest(),
               "fresh_sampler_rows": True, "peak_allocated_bytes": peak,
               "checkpoint_s": checkpoint_s, "validation_s": validation_s,
               "validation_samples": validated["validation_samples"], "validation": validated["validation"],
               "validation_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        row[f"sample_weighted_train_{args.loss_type}"] = mean_loss
        if profile is not None:
            row["profile"] = profile
        try:
            next(iterator)
            raise RuntimeError("Bounded training loader returned extra batches")
        except StopIteration:
            pass
        shutdown_start = time.perf_counter()
        shutdown(loader, dataset)
        loader = dataset = None
        row["training_worker_shutdown_s"] = time.perf_counter() - shutdown_start
        row["bounded_run_s_excluding_profiler"] = (training_with_startup_s + checkpoint_s
            + validation_s + row["training_worker_shutdown_s"])
        row["note"] = "Streaming timings include source reads, worker waits, preparation, transfer, backbone, loss, backward, optimizer, and loss reporting. Default variants retain per-step scalar synchronization. Deferred reporting aggregates detached float64 losses on device and reads them at the measurement boundary. Complete bounded-run time adds checkpoint and limited final validation. No full-epoch extrapolation."
        return row
    finally:
        shutdown(loader, dataset)
        shutdown(validation_loader, validation_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["runner2", "runner4", "prepare2", "prepare4", "graph2"])
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--validation-max-hessian-elements", type=int, default=8_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-hessian-elements", type=int, default=4_000_000)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=350)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--warmup-fraction", type=float, default=0.15)
    parser.add_argument("--loss-type", choices=("mse", "mae"), default="mse")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--hessian-cutoff", type=float)
    known, _ = parser.parse_known_args()
    if known.config:
        config = json.loads(known.config.read_text())
        selected = {key: config[key] for key in ("batch_size", "max_hessian_elements", "min_atoms",
                    "max_atoms", "seed", "lr", "lr_schedule", "warmup_fraction", "loss_type", "hidden", "hessian_cutoff",
                    "validation_batch_size", "validation_max_hessian_elements") if key in config}
        selected["schedule_steps"] = config.get("max_steps")
        parser.set_defaults(**selected)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats, args.batch_size, args.max_hessian_elements,
           args.validation_samples) < 1 or args.profile_steps < 0:
        parser.error("Use positive training limits and a nonnegative profile limit")
    torch.set_num_threads(2)
    if args.lr_schedule == "cosine" and (not args.schedule_steps or
            args.schedule_steps < args.warmup + args.steps + args.profile_steps):
        parser.error("Cosine schedule needs enough total optimizer steps")
    torch.manual_seed(args.seed)
    common_start = time.perf_counter()
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint,
                              hidden=args.hidden, cutoff=args.hessian_cutoff)
    initial = copy.deepcopy(model.head.state_dict())
    original_hash = model_hash(model.base)
    report = {"benchmark": "streaming_production_runner", "complete": False, "runs": [],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "precision": torch.get_float32_matmul_precision(), "source_reads_included": True,
              "loss_type": args.loss_type, "head_config": model.head.config,
              "lr_schedule": args.lr_schedule, "schedule_steps": args.schedule_steps,
              "warmup_fraction": args.warmup_fraction,
              "replayed_batches": False, "cached_frozen_features": False,
              "sampler": "production shard sampler", "batch_size_limit": args.batch_size,
              "max_hessian_elements": args.max_hessian_elements,
              "validation_sample_limit": args.validation_samples,
              "production_update": "hip_orb.runner.train_update",
              "benchmark_row_plan_and_identity_checks_excluded": True,
              "baseline_per_step_scalar_loss_reporting_included": True,
              "loss_reporting_note": "Default variants synchronize scalar loss every step. The deferred2 candidate changes only detached loss reporting; all optimizer updates stay unchanged.",
              "common_model_initialization_s": time.perf_counter() - common_start,
              "case_total_scope": "Case pipeline totals exclude shared model initialization and benchmark diagnostics. Common model initialization is reported separately. First-use costs and OS cache state differ between cases, so startup differences do not prove an optimization benefit.",
              "os_cache_note": "No OS cache eviction. First-touch and reused filesystem-cache states are uncontrolled. Fresh worker-local caches and bounded array caches match the runner.",
              "coverage_note": "Bounded prefixes of the actual shuffled shard sampler. Report actual atom counts. This is not a complete epoch or a dataset-wide throughput projection."}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if "graph2" in args.variants:
        audit_data = OrbMolDataset(args.train_index, training_only=True)
        try:
            report["cpu_graph_audits"] = []
            for requested in (2, 20, 50, 70, 110, 160, 250, 350):
                samples, n = read_pool(audit_data, requested, 4)
                report["cpu_graph_audits"].append(dict(natoms=n, **audit_graph(model, samples)))
        finally:
            audit_data.close()
        save()
    try:
        for repeat in range(args.repeats):
            order = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            sequences = set()
            for variant in order:
                with compiler_candidate(model, backbone=variant in ("compile2", "compile_fusion2", "compile_triton2"),
                                        pair_inputs=variant in ("fusion2", "compile_fusion2"),
                                        shaped_conditioning=variant == "shape2",
                                        triton_pairs=variant in ("triton2", "compile_triton2")):
                    row = run_case(model, initial, variant, repeat, args)
                row["compiled_backbone"] = variant in ("compile2", "compile_fusion2", "compile_triton2")
                row["compiled_pair_inputs"] = variant in ("fusion2", "compile_fusion2")
                row["triton_pair_inputs"] = variant in ("triton2", "compile_triton2")
                row["compiler_force_same_precision"] = row["compiled_backbone"]
                report["runs"].append(row)
                sequences.add((row["sample_sequence_sha256"], row["sampler_rows_sha256"]))
                save()
                print(json.dumps({key: row[key] for key in ("variant", "repeat", "steps", "samples",
                    "streaming_ms_per_step", "streaming_samples_per_s", "main_loader_wait_ms_per_step",
                    "startup_and_warmup_s", "checkpoint_s", "validation_s")}), flush=True)
            if len(sequences) != 1:
                raise RuntimeError("Strategies used different samples or batch boundaries")
        report["summary"] = []
        for variant in args.variants:
            runs = [row for row in report["runs"] if row["variant"] == variant]
            report["summary"].append({"variant": variant,
                "aggregate_streaming_samples_per_s": sum(row["samples"] for row in runs) / sum(row["training_s"] for row in runs),
                "aggregate_training_samples_per_s_including_startup": sum(row["samples"] + row["warmup_samples"] for row in runs)
                    / sum(row["training_with_startup_s"] for row in runs),
                "mean_bounded_run_s_excluding_profiler": float(np.mean([row["bounded_run_s_excluding_profiler"] for row in runs]))})
        report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
        report["backbone_has_gradients"] = any(p.grad is not None for p in model.base.parameters())
        if not report["frozen_backbone_unchanged"] or report["backbone_has_gradients"]:
            raise RuntimeError("Frozen backbone changed")
        report["paired_sample_sequences_match"] = True
        report["complete"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
