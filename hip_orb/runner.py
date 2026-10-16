"""Streaming training runner. Short and long runs differ only in configuration."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from hip_orb import ORB_COMMIT, ORB_VERSION
from hip_orb.benchmark_training import training_step
from hip_orb.data import OrbMolDataset, make_loader, unpack_hessian
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.metrics import hessian_metrics
from hip_orb.model import HIPOrb
from hip_orb.losses import LOSS_TYPES
from hip_orb.schedules import learning_rate
from hip_orb.preparation import prepare_batch
from hip_orb.execution import EXECUTION_MODES, configure_execution
from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.packed_training import collate_packed_compact, packed_update


def timed_batches(loader, timing):
    iterator = iter(loader)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        timing["loader_s"] += time.perf_counter() - started
        yield batch


def train_update(model, optimizer, batch, device, seed, *, scalar_loss=True, loss_type="mse",
                 translation_scale=0.0, consistency_weight=0.0, consistency_seed=1,
                 rotation_sampling="uniform", reference_views=None):
    """Use the runner's preparation, optimizer step, and scalar loss reporting."""
    with torch.profiler.record_function("hip_runner_prepare"):
        if "rotation" not in batch:
            batch = prepare_batch(batch, seed=seed, translation_scale=translation_scale,
                                  rotation_sampling=rotation_sampling,
                                  include_molecules=not getattr(model, "_tensor_graph", False))
    loss = training_step(model, optimizer, batch, device=device, loss_type=loss_type,
                         consistency_weight=consistency_weight, consistency_seed=consistency_seed,
                         translation_scale=translation_scale, rotation_sampling=rotation_sampling,
                         reference_views=reference_views)
    with torch.profiler.record_function("hip_loss_reporting"):
        # Default comparisons retain the existing per-step synchronization.
        return (float(loss) if scalar_loss else loss), len(batch["coords"])


def validate(model, loader, device, report_samples=32):
    """Stream the separate validation set and average each metric per molecule."""
    model.eval()
    totals, counts, batch_sizes, examples = {}, {}, [], []
    with torch.no_grad():
        for batch in loader:
            molecules = [Atoms(numbers=z.numpy(), positions=x.numpy())
                         for z, x in zip(batch["atomic_numbers"], batch["coords"], strict=True)]
            for i, atoms in enumerate(molecules):
                atoms.info.update(charge=int(batch["charge"][i]), spin=int(batch["spin"][i]))
            graph = refresh_message_vectors(canonical_message_graph(
                model.adapter.from_ase_atoms_list(molecules, device=device)))
            sizes = torch.full((len(molecules),), len(molecules[0]), dtype=torch.long)
            predicted = model.head(model.features(graph, sizes=sizes))
            target = unpack_hessian(batch["hessian_upper"].to(device), len(molecules[0]))
            rotated = prepare_batch(batch, seed=0)
            graph = refresh_message_vectors(canonical_message_graph(
                model.adapter.from_ase_atoms_list(rotated["molecules"], device=device)))
            rotated_predictions = model.head(model.features(graph, sizes=sizes))
            rotation = rotated["rotation"].to(device)
            transform = torch.kron(torch.eye(len(molecules[0]), device=device), rotation.contiguous())
            batch_sizes.append(len(molecules))
            for i, atoms in enumerate(molecules):
                positions = torch.as_tensor(atoms.positions, device=device, dtype=target.dtype)
                masses = torch.as_tensor(atoms.get_masses(), device=device, dtype=target.dtype)
                row = hessian_metrics(predicted[i], target[i], positions, masses)
                row["rotation_rmse"] = float((rotated_predictions[i] -
                    transform @ predicted[i] @ transform.T).square().mean().sqrt())
                for key, value in row.items():
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite validation metric: {key}")
                    totals[key] = totals.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
                if len(examples) < report_samples:
                    examples.append(dict(row, omol_index=int(batch["omol_index"][i])))
    if not batch_sizes:
        raise ValueError("Validation loader returned no samples")
    averaged = {key: value / counts[key] for key, value in totals.items()}
    if "first_mode_overlap" in averaged:
        averaged["first_mode_abs_cosine_rms"] = math.sqrt(averaged["first_mode_overlap"])
    return {"validation": averaged,
            "validation_samples": sum(batch_sizes), "validation_batch_sizes": batch_sizes,
            "per_sample": examples}


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    initial, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-hessian-elements", type=int, default=4_000_000)
    parser.add_argument("--batching", choices=("equal", "balanced"), default="equal")
    parser.add_argument("--batch-max-atoms", type=int, default=4096)
    parser.add_argument("--batch-window-shards", type=int, default=8)
    parser.add_argument("--batch-order", choices=("global", "locality"), default="locality")
    parser.add_argument("--loader-cache-mib", type=int, default=256)
    parser.add_argument("--loader-open-files", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--session-steps", type=int)
    parser.add_argument("--max-duration-seconds", type=float,
                        help="Stop updates after this elapsed time, including periodic validation")
    parser.add_argument("--session-duration-seconds", type=float,
                        help="Save and stop this job after this duration; continue in a linked job")
    parser.add_argument("--validation-every", type=int, default=0,
                        help="Optimizer steps between validations; zero means final validation only")
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--validation-every-seconds", type=float, default=0,
                        help="Elapsed seconds between validation starts; zero disables the timer")
    parser.add_argument("--validation-samples", type=int,
                        help="Maximum geometries per periodic and final validation")
    parser.add_argument("--validation-batch-size", type=int,
                        help="Validation batch size; defaults to training batch size")
    parser.add_argument("--validation-max-hessian-elements", type=int,
                        help="Validation memory budget; defaults to training budget")
    parser.add_argument("--report-samples", type=int, default=32,
                        help="Maximum per-molecule examples retained in the validation report")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--hessian-cutoff", type=float)
    parser.add_argument("--radial-features", choices=("log", "bessel"), default="log")
    parser.add_argument("--radial-cutoff", type=float, default=12.0)
    parser.add_argument("--radial-bases", type=int, default=16)
    parser.add_argument("--pair-residual-layers", type=int, default=0)
    parser.add_argument("--neighbor-pooling", action="store_true")
    parser.add_argument("--extra-orb-round", action="store_true")
    parser.add_argument("--all-to-all", action="store_true",
                        help="Use every Hessian pair, global context pooling, and independent untruncated radial features")
    parser.add_argument("--orb-initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--orb-message-graph", choices=("backbone", "hessian"), default="backbone")
    parser.add_argument("--translation-projection", action="store_true")
    parser.add_argument("--context-rounds", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int,
                        help="Internal MLP width; defaults to the readout feature width")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=0.15)
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="mse",
                        help="Hessian MSE, element MAE, atom-pair block RMSE, or mass-weighted Hessian MSE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="optimized",
                        help="Use float32 optimizations or the eager reference implementation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--translation-scale", type=float, default=0.0,
                        help="Random shifts per molecule, uniform within +/- this many angstroms")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="Weight of two-view rotation consistency in the selected loss basis")
    parser.add_argument("--rotation-sampling", choices=("uniform", "legacy"), default="uniform",
                        help="Use legacy restricted rotations only for the ablation baseline")
    parser.add_argument("--reference-views", type=int, choices=(1, 2), default=None,
                        help="Reference views per update; defaults to two when consistency is enabled")
    if initial.config:
        config = json.loads(initial.config.read_text())
        allowed = {action.dest for action in parser._actions if not action.required}
        if set(config) - allowed:
            parser.error("Configuration contains unknown settings or private path settings")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not math.isfinite(args.translation_scale) or args.translation_scale < 0:
        parser.error("Translation scale must be finite and nonnegative")
    if not math.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        parser.error("Consistency weight must be finite and nonnegative")
    if args.rotation_sampling not in ("uniform", "legacy"):
        parser.error("Rotation sampling must be uniform or legacy")
    if args.reference_views is None:
        args.reference_views = 2 if args.consistency_weight else 1
    if args.reference_views not in (1, 2) or (args.consistency_weight and args.reference_views != 2):
        parser.error("Consistency requires two reference views; views must be 1 or 2")
    if args.batching not in ("equal", "balanced"):
        parser.error("Batching must be equal or balanced")
    if args.batching == "balanced" and (
            args.loss_type not in ("block_rmse", "mse", "mae")
            or args.translation_scale or args.consistency_weight
            or args.reference_views != 1 or args.rotation_sampling != "uniform"):
        parser.error("Balanced packed training requires one uniform view, no shifts or consistency, and block_rmse/mse/mae")
    augmentation_config = {"rotation": ("haar_so3_qr_v2" if args.rotation_sampling == "uniform"
                                        else "legacy_qr_v1"),
                           "translation_scale_angstrom": args.translation_scale,
                           "consistency_weight": args.consistency_weight,
                           "consistency_loss_type": args.loss_type,
                           "reference_views": args.reference_views}
    if args.session_steps is not None and args.session_steps < 1:
        parser.error("Session steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be positive and finite")
    try:
        learning_rate(1, args.lr, args.lr_schedule, args.max_steps, args.warmup_fraction)
    except ValueError as error:
        parser.error(str(error))
    schedule_config = {"schedule": args.lr_schedule, "peak_lr": args.lr,
                       "total_steps": args.max_steps, "warmup_fraction": args.warmup_fraction}
    if args.loss_type not in LOSS_TYPES:
        parser.error(f"Hessian loss must be one of {LOSS_TYPES}")
    loss_key = f"train_{args.loss_type}"
    mass_weighting = "ase_standard_atomic_masses_amu" if args.loss_type == "mass_weighted_mse" else None
    if (min(args.batch_size, args.epochs, args.loader_open_files, args.prefetch_factor) < 1
            or args.workers < 0 or args.loader_cache_mib < 0
            or (args.max_hessian_elements is not None and args.max_hessian_elements < 1)):
        parser.error("Batch size, budget, and epochs must be positive; workers must be nonnegative")
    if args.batching == "equal" and args.max_hessian_elements is None:
        parser.error("Equal-size batching requires a Hessian budget")
    if args.report_samples < 0 or args.validation_every < 0 or any(value is not None and value < 1
                                      for value in (args.max_steps, args.validation_batches,
                                                    args.validation_samples, args.validation_batch_size,
                                                    args.validation_max_hessian_elements)):
        parser.error("Step and validation limits must be positive")
    if (not math.isfinite(args.validation_every_seconds) or args.validation_every_seconds < 0
            or (args.max_duration_seconds is not None and
                (not math.isfinite(args.max_duration_seconds) or args.max_duration_seconds <= 0))):
        parser.error("Duration must be positive and validation interval must be nonnegative and finite")
    if not 1 <= args.min_atoms <= args.max_atoms:
        parser.error("Invalid atom range")
    if args.session_duration_seconds is not None and (
            not math.isfinite(args.session_duration_seconds) or args.session_duration_seconds <= 0):
        parser.error("Session duration must be positive and finite")
    training = OrbMolDataset(args.train_index, training_only=True,
                             cache_bytes=args.loader_cache_mib << 20, max_open=args.loader_open_files)
    validation = OrbMolDataset(args.validation_index, training_only=True)
    if set(training.paths) & set(validation.paths) or set(training.sources) & set(validation.sources):
        parser.error("Training and validation must use separate sources and shard files")
    for dataset in (training, validation):
        selected = np.flatnonzero((dataset.natoms >= args.min_atoms) & (dataset.natoms <= args.max_atoms))
        if not len(selected):
            parser.error("The dataset has no shards in the configured atom range")
        counts = np.diff(dataset.ptr)[selected]
        dataset.paths = [dataset.paths[i] for i in selected]
        dataset.sources = [dataset.sources[i] for i in selected]
        dataset.natoms = dataset.natoms[selected]
        dataset.ptr = np.r_[0, np.cumsum(counts)]
    common = {"workers": args.workers, "batch_size": args.batch_size,
              "max_hessian_elements": args.max_hessian_elements, "seed": args.seed,
              "pin_memory": args.device.startswith("cuda")}
    common["prefetch_factor"] = args.prefetch_factor
    validation_common = dict(common,
                             batch_size=args.validation_batch_size or args.batch_size,
                             max_hessian_elements=args.validation_max_hessian_elements
                             or args.max_hessian_elements or 8_000_000)
    validation_loader = make_loader(validation, max_batches=args.validation_batches,
                                    max_samples=args.validation_samples, **validation_common)
    train_options = {}
    batching_config = {"mode": args.batching}
    if args.batching == "balanced":
        args.output.mkdir(parents=True, exist_ok=True)
        sampler = BalancedBatchSampler(
            training, max_atoms=args.batch_max_atoms, batch_size=args.batch_size,
            max_hessian_elements=args.max_hessian_elements,
            window_shards=args.batch_window_shards, seed=args.seed,
            order=args.batch_order,
            cache_path=args.output / "batch_plan.npz")
        train_options = {"batch_sampler": sampler, "collate_fn": collate_packed_compact}
        batching_config.update(settings=sampler.settings, fingerprint=sampler.fingerprint,
                               batches_per_epoch=len(sampler))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    weights = {} if args.weights_path is None else {"weights_path": args.weights_path}
    model = HIPOrb.pretrained(args.device, hidden=args.hidden, cutoff=args.hessian_cutoff,
                              radial_features=args.radial_features,
                              radial_cutoff=args.radial_cutoff,
                              radial_bases=args.radial_bases,
                              pair_residual_layers=args.pair_residual_layers,
                              neighbor_pooling=args.neighbor_pooling,
                              extra_orb_round=args.extra_orb_round,
                              translation_projection=args.translation_projection,
                              context_rounds=args.context_rounds, mlp_hidden=args.mlp_hidden,
                              all_to_all=args.all_to_all, orb_initialization=args.orb_initialization,
                              orb_message_graph=args.orb_message_graph, **weights)
    execution_config = configure_execution(model, args.execution_mode)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0)
    global_step = 0
    previous_elapsed, last_validation_elapsed = 0.0, 0.0
    resume_epoch, epoch_batches = 0, 0
    history, validation_history, training_batch_sizes = [], [], []
    execution_history = []
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if saved["head_config"] != model.head.config or saved["orb_commit"] != ORB_COMMIT:
            raise ValueError("Checkpoint architecture or Orb base differs")
        if saved.get("loss_type", "mse") != args.loss_type:
            raise ValueError("Checkpoint Hessian loss differs from the configured loss")
        if saved.get("lr_schedule_config", schedule_config) != schedule_config:
            raise ValueError("Checkpoint learning rate schedule differs")
        if saved.get("augmentation_config", augmentation_config) != augmentation_config:
            raise ValueError("Checkpoint augmentation or consistency settings differ")
        if saved.get("batching_config", {"mode": "equal"}) != batching_config:
            raise ValueError("Checkpoint batch plan or batching settings differ")
        model.head.load_state_dict(saved["head"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        global_step = saved.get("global_step", 0)
        previous_elapsed = saved.get("training_elapsed_s", 0.0)
        last_validation_elapsed = saved.get("last_validation_elapsed_s", 0.0)
        resume_epoch = saved.get("epoch", 0)
        epoch_batches = saved.get("epoch_batches", 0)
        history = saved.get("train_loss", saved.get(loss_key, []))
        validation_history = saved.get("validation_history", [])
        training_batch_sizes = saved.get("training_batch_sizes", [])
        execution_history = saved.get("execution_history", [])
        if not execution_history and saved.get("execution_config"):
            execution_history = [{"global_step": global_step, "config": saved["execution_config"]}]
    if not execution_history or execution_history[-1]["config"] != execution_config:
        execution_history.append({"global_step": global_step, "config": execution_config})
    train_loader = make_loader(training, max_batches=args.max_steps,
                               start_batch=epoch_batches, **common, **train_options)
    original_hash = state_hash(model.base)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    current_epoch = resume_epoch
    session_start_step = global_step

    def elapsed():
        return previous_elapsed + time.perf_counter() - started

    def total_limit_reached():
        return ((args.max_steps is not None and global_step >= args.max_steps)
                or (args.max_duration_seconds is not None
                    and elapsed() >= args.max_duration_seconds))

    def limit_reached():
        return (total_limit_reached() or (args.session_steps is not None
                and global_step - session_start_step >= args.session_steps)
                or (args.session_duration_seconds is not None
                and time.perf_counter() - started >= args.session_duration_seconds))

    def save_checkpoint():
        checkpoint = {"head": model.head.state_dict(), "head_config": model.head.config,
                      "batching_config": batching_config,
                      "augmentation_config": augmentation_config,
                      "optimizer": optimizer.state_dict(), "global_step": global_step,
                      "orb_commit": ORB_COMMIT, "orb_version": ORB_VERSION,
                      "precision": "float32-high",
                      "execution_config": execution_config, "execution_history": execution_history,
                      "teacher": "orbmol_v2",
                      "lr_schedule_config": schedule_config,
                      "training_elapsed_s": elapsed(),
                      "last_validation_elapsed_s": last_validation_elapsed,
                      "epoch": current_epoch, "epoch_batches": epoch_batches,
                      "loss_type": args.loss_type, "mass_weighting": mass_weighting,
                      "train_loss": history, loss_key: history,
                      "validation_history": validation_history,
                      "training_batch_sizes": training_batch_sizes}
        temporary = args.output / "head.tmp.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "head.pt")

    final = None
    try:
        print(json.dumps({"event": "training_started", "global_step": global_step,
                          "head_config": model.head.config,
                          "execution_config": execution_config,
                          "batching_config": batching_config,
                          "elapsed_s": previous_elapsed, "batch_size": args.batch_size,
                          "loss_type": args.loss_type,
                          "max_steps": args.max_steps, "lr_schedule_config": schedule_config,
                          "validation_batch_size": validation_common["batch_size"]}), flush=True)
        for epoch in range(resume_epoch, args.epochs):
            current_epoch = epoch
            if epoch != resume_epoch:
                epoch_batches = 0
            train_loader.batch_sampler.set_epoch(epoch)
            loss_sum, samples = 0.0, 0
            model.train()
            timing = {"loader_s": 0.0, "update_s": 0.0, "samples": 0, "updates": 0}
            for batch in timed_batches(train_loader, timing):
                if limit_reached():
                    break
                update_lr = learning_rate(global_step + 1, args.lr, args.lr_schedule,
                                          args.max_steps, args.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = update_lr
                update_started = time.perf_counter()
                if args.batching == "balanced":
                    loss, size = packed_update(model, optimizer, batch, args.device,
                                               args.seed + epoch, loss_type=args.loss_type)
                else:
                    loss, size = train_update(model, optimizer, batch, args.device, args.seed + epoch,
                                          loss_type=args.loss_type,
                                          translation_scale=args.translation_scale,
                                          consistency_weight=args.consistency_weight,
                                          consistency_seed=args.seed + 1_000_003 + global_step,
                                          rotation_sampling=args.rotation_sampling,
                                          reference_views=args.reference_views)
                timing["update_s"] += time.perf_counter() - update_started
                timing["updates"] += 1
                timing["samples"] += size
                loss_sum += loss * size
                samples += size
                global_step += 1
                epoch_batches += 1
                # Keep bounded diagnostics, not a list proportional to a long run.
                if len(training_batch_sizes) < 1024:
                    training_batch_sizes.append(size)
                now = elapsed()
                if global_step == 1 or global_step % 1000 == 0:
                    measured_s = max(timing["loader_s"] + timing["update_s"], 1e-12)
                    print(json.dumps({"global_step": global_step, "elapsed_s": now,
                                      loss_key: loss, "batch_size": size, "lr": update_lr,
                                      "loader_wait_percent": 100 * timing["loader_s"] / measured_s,
                                      "geometries_per_second": timing["samples"] / measured_s,
                                      "updates_per_second": timing["updates"] / measured_s,
                                      "window_mean_batch_size": timing["samples"] / timing["updates"]}), flush=True)
                    timing.update(loader_s=0.0, update_s=0.0, samples=0, updates=0)
                if not total_limit_reached() and (not limit_reached() or args.session_steps is not None) and (
                    (args.validation_every and global_step % args.validation_every == 0)
                    or (args.validation_every_seconds
                        and now - last_validation_elapsed >= args.validation_every_seconds)
                ):
                    last_validation_elapsed = now
                    save_checkpoint()
                    final = validate(model, validation_loader, args.device, args.report_samples)
                    validation_history.append({"global_step": global_step,
                                               "elapsed_s": now,
                                               "validation_samples": final["validation_samples"],
                                               **final["validation"]})
                    print(json.dumps({"global_step": global_step, "validation": final["validation"]}), flush=True)
                    model.train()
                    save_checkpoint()
                if limit_reached():
                    break
            if samples:
                history.append(loss_sum / samples)
                print(json.dumps({"epoch": epoch + 1, "global_step": global_step,
                                  loss_key: history[-1], "training_samples": samples}), flush=True)
            if limit_reached():
                break
            current_epoch, epoch_batches = epoch + 1, 0
        # Always exercise final validation, even if a periodic check just ran.
        training_elapsed_s = elapsed()
        session_paused = limit_reached() and not total_limit_reached()
        save_checkpoint()
        final = ({"validation": {}, "validation_samples": 0} if session_paused else
                 validate(model, validation_loader, args.device, args.report_samples))
        unchanged = original_hash == state_hash(model.base)
        if not unchanged:
            raise RuntimeError("Frozen Orb state changed")
        saved = torch.load(args.output / "head.pt", map_location="cpu", weights_only=True)
        report = {"runner": "hip_orb.train", "loader": "OrbMolDataset",
                  "batching_config": batching_config,
                  "execution_config": execution_config,
                  "execution_history": execution_history,
                  "augmentation_config": augmentation_config,
                  "atom_range": [args.min_atoms, args.max_atoms],
                  "global_step": global_step, "batch_size": args.batch_size,
                  "validation_batch_size": validation_common["batch_size"],
                  "validation_sample_limit": args.validation_samples,
                  "max_duration_seconds": args.max_duration_seconds,
                  "max_steps": args.max_steps, "validation_every": args.validation_every,
                  "lr_schedule_config": schedule_config,
                  "validation_every_seconds": args.validation_every_seconds,
                  "training_elapsed_s": training_elapsed_s,
                  "training_batch_sizes": training_batch_sizes,
                  "loss_type": args.loss_type, "mass_weighting": mass_weighting,
                  "train_loss": history, loss_key: history,
                  "validation_history": validation_history, **final,
                  "frozen_state_unchanged": unchanged, "separate_validation_set": True,
                  "final_validation_completed": not session_paused,
                  "training_complete": not session_paused,
                  "checkpoint_readable": bool(saved["head"]),
                  "orb_commit": ORB_COMMIT, "elapsed_s": elapsed(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                  if args.device.startswith("cuda") else None}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("global_step", "validation", "validation_samples",
                         "final_validation_completed", "checkpoint_readable")}), flush=True)
    finally:
        for loader, dataset in ((train_loader, training), (validation_loader, validation)):
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator._shutdown_workers()
            dataset.close()


if __name__ == "__main__":
    main()
