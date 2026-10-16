"""Compare bounded, complete GPU training steps; never start a full training run."""
import argparse
import copy
import gc
import hashlib
import math
import json
from pathlib import Path
import time
import tempfile
import shutil

import numpy as np
import torch
from ase.data import atomic_masses

from hip_orb.data import OrbMolDataset, make_loader, unpack_hessian
from hip_orb.block_layout import require, layout
from hip_orb.model import HIPOrb
from hip_orb.losses import LOSS_TYPES, hessian_error_loss, rotate_hessian, rotation_consistency_loss
from hip_orb.preparation import PreparedCollator, prepare_batch
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors, deterministic_graph_audit

ATOMIC_MASSES = torch.from_numpy(atomic_masses.copy())


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_step(model, optimizer, batch, device="cuda", *, host_sizes=True, loss_type="mse",
                  consistency_weight=0.0, consistency_seed=1, translation_scale=0.0,
                  return_components=False, rotation_sampling="uniform", reference_views=None):
    """Include rotations, ASE graphs, transfer, frozen features, and head update."""
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Hessian loss must be one of {LOSS_TYPES}")
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("Consistency weight must be finite and nonnegative")
    reference_views = (2 if consistency_weight else 1) if reference_views is None else reference_views
    if reference_views not in (1, 2) or (consistency_weight and reference_views != 2):
        raise ValueError("Rotation consistency requires two reference views; views must be 1 or 2")
    with torch.profiler.record_function("hip_cpu_prepare"):
        if "rotation" not in batch:
            batch = prepare_batch(batch, translation_scale=translation_scale,
                                  rotation_sampling=rotation_sampling,
                                  include_molecules=not getattr(model, "_tensor_graph", False))
    rotation = batch["rotation"]
    with torch.profiler.record_function("hip_graph_build_or_transfer"):
        if "prepared_graph" in batch:
            graph = batch["prepared_graph"].to(device)
        elif getattr(model, "_tensor_graph", False) and "rotated_coords" in batch:
            from hip_orb.tensor_graph import molecular_training_graph
            graph = molecular_training_graph(batch, model.adapter, device)
        else:
            graph = model.adapter.from_ase_atoms_list(batch["molecules"], device=device)
        if isinstance(model, HIPOrb):
            if "prepared_graph" not in batch:
                graph = canonical_message_graph(graph)
            graph = refresh_message_vectors(graph)
    # Equal-N batch sizes already exist on CPU; avoid reading them back from CUDA.
    sizes = torch.full((len(batch["coords"]),), batch["coords"].shape[1], dtype=torch.long)
    with torch.profiler.record_function("hip_frozen_backbone"):
        pairs = [{key: value.to(device, non_blocking=True) for key, value in pair.items()}
                 for pair in batch["prepared_pairs"]] if "prepared_pairs" in batch else None
        feature = model.features(graph, sizes=sizes if host_sizes else None, pairs=pairs) if isinstance(model, HIPOrb) else model.features(graph)
    # No feature cache: rerun the frozen backbone.
    with torch.profiler.record_function("hip_target_transfer_and_rotation"):
        upper = batch["hessian_upper"].to(device, non_blocking=True)
        n = batch["coords"].shape[1]
        r = rotation.to(device, non_blocking=True)
        target = unpack_hessian(upper, n)
        target = torch.einsum("ab,kibjd,cd->kiajc", r, target.reshape(-1, n, 3, n, 3), r)
        target = target.reshape(-1, 3 * n, 3 * n)
    # The head includes analytical fixed-charge Coulomb curvature exactly once.
    with torch.profiler.record_function("hip_hessian_head_coulomb_loss"):
        predicted = (model.head(feature, as_tensor=True) if getattr(model.head, "_tensor_training", False)
                     else torch.stack(model.head(feature)))
        error = predicted - target
        masses = (ATOMIC_MASSES[batch["atomic_numbers"].long()]
                  if loss_type == "mass_weighted_mse" else None)
        loss = hessian_error_loss(error, loss_type, masses=masses)
    reference_loss = loss
    consistency = loss.new_zeros(())
    if reference_views == 2:
        # The batch retains original coordinates and targets after preparation.
        second = prepare_batch(batch, seed=consistency_seed, translation_scale=translation_scale,
                               rotation_sampling=rotation_sampling)
        with torch.profiler.record_function("hip_consistency_second_graph_and_backbone"):
            second_graph = model.adapter.from_ase_atoms_list(second["molecules"], device=device)
            if isinstance(model, HIPOrb):
                second_graph = refresh_message_vectors(canonical_message_graph(second_graph))
                second_feature = model.features(second_graph, sizes=sizes if host_sizes else None)
            else:
                second_feature = model.features(second_graph)
        with torch.profiler.record_function("hip_consistency_head_and_loss"):
            second_prediction = torch.stack(model.head(second_feature))
            second_target = rotate_hessian(unpack_hessian(upper, n), second["rotation"])
            reference_loss = (reference_loss + hessian_error_loss(
                second_prediction - second_target, loss_type, masses)) * 0.5
            consistency = rotation_consistency_loss(predicted, second_prediction,
                rotation, second["rotation"], loss_type, masses)
            loss = reference_loss + consistency_weight * consistency
    require(torch.isfinite(loss), "Training loss is not finite")
    with torch.profiler.record_function("hip_backward_and_optimizer"):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
        optimizer.step()
    if return_components:
        return {"total": loss.detach(), "reference": reference_loss.detach(),
                "consistency": consistency.detach()}
    return loss.detach()


def validate_cpu_graphs(model, index):
    """Untimed CPU/GPU agreement check on one real molecule per source shard."""
    dataset = OrbMolDataset(index, training_only=True)
    from hip_orb.data import collate_samples
    checked = []
    try:
        for start in dataset.ptr[:-1]:
            batch = prepare_batch(collate_samples([dataset[int(start)]]), adapter=model.adapter)
            cpu = refresh_message_vectors(batch["prepared_graph"].to("cuda"))
            gpu = refresh_message_vectors(canonical_message_graph(
                model.adapter.from_ase_atoms_list(batch["molecules"], device="cuda")))
            n = batch["coords"].shape[1]
            a = (cpu.senders * n + cpu.receivers).sort().values
            b = (gpu.senders * n + gpu.receivers).sort().values
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(cpu.senders, gpu.senders, atol=0, rtol=0)
            torch.testing.assert_close(cpu.receivers, gpu.receivers, atol=0, rtol=0)
            torch.testing.assert_close(cpu.edge_features["vectors"], gpu.edge_features["vectors"], atol=0, rtol=0)
            with deterministic_graph_audit(), torch.no_grad():
                expected, actual = model(gpu)[0], model(cpu)[0]
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            checked.append(int(n))
    finally:
        dataset.close()
    return {"passed": True, "natoms": checked, "hessian_atol": 1e-4, "hessian_rtol": 1e-4,
            "audit_precision": "float32-highest", "audit_deterministic": True,
            "training_precision_unchanged": True}


def profile_pipeline(model, initial, index, config, batch_size):
    """Separate diagnostic profile; never include profiler overhead in timings."""
    model.head.load_state_dict(initial)
    model.head.fuse_pairs = config["fuse_pairs"]
    model.head.cache_pairs = config["fuse_pairs"]
    model.head.batch_pairs = config.get("batch_pairs", True)
    model.train()
    dataset = OrbMolDataset(index, training_only=True)
    prep = config["preparation"]
    kwargs = {} if prep == "sync" else {"collate_fn": PreparedCollator(
        adapter=model.adapter if prep == "cpu_graph" else None)}
    count = len(set(dataset.natoms.tolist()))
    loader = make_loader(dataset, workers=config["workers"], batch_size=batch_size,
                         stratified=True, max_batches=count, **kwargs)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            iterator = iter(loader)
            for _ in range(count):
                with torch.profiler.record_function("hip_loader_wait"):
                    batch = next(iterator)
                training_step(model, optimizer, batch, host_sizes=config["fuse_pairs"])
                trace.step()
            torch.cuda.synchronize()
            try:
                next(iterator)
                raise RuntimeError("Profiler loader returned extra batches")
            except StopIteration:
                pass
        stages = [{"name": event.key, "calls": event.count,
                   "cpu_total_ms": event.cpu_time_total / 1000,
                   "gpu_total_ms": event.device_time_total / 1000}
                  for event in trace.key_averages() if event.key.startswith("hip_")]
        return {"config": config, "steps": count, "stages": stages,
                "note": "Separate profiler run. CPU ranges include host waits. Worker CPU execution is outside this trace."}
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
        dataset.close()


def validate_batched_head(model, index):
    from hip_orb.data import collate_samples
    dataset = OrbMolDataset(index, training_only=True)
    checked = []
    try:
        with deterministic_graph_audit():
            for requested in (50, 70, 110, 250, 350):
                shard = int(np.abs(dataset.natoms - requested).argmin())
                start, end = map(int, dataset.ptr[shard:shard + 2])
                n = int(dataset.natoms[shard])
                count = min(4, max(1, 2_000_000 // (3*n)**2), end-start)
                batch = prepare_batch(collate_samples([dataset[i] for i in range(start, start+count)]))
                graph = refresh_message_vectors(canonical_message_graph(
                    model.adapter.from_ase_atoms_list(batch["molecules"], device="cuda")))
                features = model.features(graph, sizes=torch.full((count,), n, dtype=torch.long))
                model.head.batch_pairs = False
                model.head.zero_grad(set_to_none=True)
                reference = model.head(features)
                sum(h.square().mean() for h in reference).backward()
                gradients = [p.grad.clone() for p in model.head.parameters()]
                model.head.batch_pairs = True
                model.head.zero_grad(set_to_none=True)
                actual = model.head(features)
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
                sum(h.square().mean() for h in actual).backward()
                for p, expected in zip(model.head.parameters(), gradients):
                    torch.testing.assert_close(p.grad, expected, atol=1e-5, rtol=1e-4)
                checked.append({"natoms": n, "batch_size": count, "passed": True})
    finally:
        model.head.zero_grad(set_to_none=True)
        dataset.close()
    return checked


def run_case(model, initial, index, config, *, steps, warmup, repeats, batch_size):
    records = []
    for repeat in range(repeats):
        layout.cache_clear()
        model.head.load_state_dict(initial)
        model.train()
        torch.manual_seed(100 + repeat)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
        dataset = OrbMolDataset(index, backend=config["backend"], cache_mode=config["cache"], training_only=True)
        preparation = config.get("preparation", "sync")
        model.head.fuse_pairs = config.get("fuse_pairs", True)
        model.head.cache_pairs = config.get("fuse_pairs", True)
        model.head.batch_pairs = config.get("batch_pairs", True)
        kwargs = {} if preparation == "sync" else {"collate_fn": PreparedCollator(
            adapter=model.adapter if preparation == "cpu_graph" else None)}
        loader = make_loader(dataset, workers=config["workers"], order="shard", seed=31 + repeat,
                             batch_size=batch_size, max_hessian_elements=2_000_000, pin_memory=True,
                             max_batches=steps + warmup, stratified=config.get("stratified", False), **kwargs)
        iterator = None
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            startup = time.perf_counter()
            iterator = iter(loader)
            for _ in range(warmup):
                training_step(model, optimizer, next(iterator), host_sizes=config.get("fuse_pairs", True))
            torch.cuda.synchronize()
            startup_s = time.perf_counter() - startup
            started = time.perf_counter()
            cpu_started = time.process_time()
            samples, identities, atom_counts = 0, [], {}
            for _ in range(steps):
                batch = next(iterator)  # Includes waiting for the loader.
                loss = training_step(model, optimizer, batch, host_sizes=config.get("fuse_pairs", True))
                samples += len(batch["omol_index"])
                identities.extend(batch["omol_index"].tolist())
                n = str(batch["coords"].shape[1])
                atom_counts[n] = atom_counts.get(n, 0) + len(batch["omol_index"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - cpu_started
            # Exhaust the bounded iterator before stopping persistent workers.
            # No large, unused prefetch batches remain in worker queues.
            try:
                next(iterator)
                raise RuntimeError("The bounded loader returned extra batches")
            except StopIteration:
                pass
            if any(parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("Frozen backbone has gradients")
            records.append({"repeat": repeat, "startup_and_warmup_s": startup_s,
                            "steps": steps, "samples": samples, "elapsed_s": elapsed,
                            "ms_per_step": elapsed * 1000 / steps, "samples_per_s": samples / elapsed,
                            "main_process_cpu_s": cpu_elapsed,
                            "total_s_including_startup": startup_s + elapsed,
                            "last_loss": float(loss), "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                            "samples_by_natoms": atom_counts,
                            "sample_sequence_sha256": hashlib.sha256(np.asarray(identities, dtype=np.int64).tobytes()).hexdigest()})
        finally:
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator._shutdown_workers()
            dataset.close()
            del iterator, loader, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    return {**config, "runs": records,
            "median_ms_per_step": float(np.median([r["ms_per_step"] for r in records])),
            "median_samples_per_s": float(np.median([r["samples_per_s"] for r in records])),
            "aggregate_samples_per_s": sum(r["samples"] for r in records) / sum(r["elapsed_s"] for r in records),
            "median_total_s": float(np.median([r["total_s_including_startup"] for r in records]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--profile", nargs="+", default=["50_70", "70_110", "2_350"])
    parser.add_argument("--local-cache-parent", type=Path)
    parser.add_argument("--pipeline", action="store_true", help="Compare CPU preparation and pair-head fusion, using cached gzip")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--batch-pairs-comparison", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.batch_size) < 1 or args.warmup < 1:
        raise ValueError("Use positive step, warmup, repeat, and batch counts")
    torch.set_num_threads(2)
    stage_start = time.perf_counter()
    local_cache = tempfile.TemporaryDirectory(prefix="hip-step-", dir=args.local_cache_parent) if args.local_cache_parent else None
    mmap_indices = {}
    if local_cache:
        local_root = Path(local_cache.name)
        shutil.copytree(args.cache / "mmap", local_root / "mmap")
        for profile in args.profile:
            with np.load(args.cache / f"{profile}_mmap.npz", allow_pickle=False) as original:
                content = {key: original[key].copy() for key in original.files}
            content["paths"] = np.asarray([str(local_root / "mmap" / Path(p).name) for p in content["paths"]])
            mmap_indices[profile] = local_root / f"{profile}.npz"
            np.savez(mmap_indices[profile], **content)
    staging_s = time.perf_counter() - stage_start
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    initial = copy.deepcopy(model.head.state_dict())
    original_hash = model_hash(model.base)
    configs = [dict(storage="gzip", backend="hdf5", cache="none", workers=0),
               dict(storage="gzip", backend="hdf5", cache="chunks", workers=0),
               dict(storage="gzip", backend="hdf5", cache="arrays", workers=0),
               dict(storage="gzip", backend="hdf5", cache="arrays", workers=2),
               dict(storage="gzip", backend="hdf5", cache="arrays", workers=4),
               dict(storage="zstd", backend="hdf5", cache="arrays", workers=0),
               dict(storage="zstd", backend="hdf5", cache="arrays", workers=2),
               dict(storage="mmap", backend="mmap", cache="arrays", workers=0),
               dict(storage="mmap", backend="mmap", cache="arrays", workers=2)]
    if args.pipeline:
        configs = [dict(storage="gzip", backend="hdf5", cache="arrays", workers=w,
                        preparation=prep, fuse_pairs=fused, batch_pairs=False, stratified=True)
                   for prep, w, fused in [("sync", 0, False), ("sync", 0, True),
                                           ("cpu_prepare", 0, True), ("cpu_prepare", 2, True),
                                           ("cpu_prepare", 4, True), ("cpu_graph", 2, True)]]
    if args.batch_pairs_comparison:
        configs = [dict(storage="gzip", backend="hdf5", cache="arrays", workers=2,
                        preparation="cpu_prepare", fuse_pairs=True, stratified=True, batch_pairs=batched)
                   for batched in (False, True)]
    # Record effective defaults so reports describe the path that actually ran.
    for config in configs:
        config.setdefault("fuse_pairs", True)
        config.setdefault("batch_pairs", True)
    report = {"benchmark": "end_to_end_training", "full_training_started": False,
              "steps_per_run": args.steps, "warmup_steps": args.warmup, "batch_size_limit": args.batch_size,
              "feature_cache": False, "rotation_augmentation": True, "profiles": {}, "complete": False,
              "mmap_location": "node_local_temporary" if local_cache else "shared_cache_filesystem",
              "mmap_staging_s": staging_s,
              "training_fields_only": True, "bounded_worker_prefetch": True,
              "note": "Includes loader, graph, transfer, backbone, Coulomb, head, loss, backward, optimizer. No OS cache eviction."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.batch_pairs_comparison:
        report["batched_head_agreement"] = validate_batched_head(model, args.cache / "2_350_gzip.npz")
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.pipeline:
        try:
            report["cpu_graph_agreement"] = validate_cpu_graphs(model, args.cache / "2_350_gzip.npz")
        except (AssertionError, RuntimeError, NotImplementedError) as error:
            report["cpu_graph_agreement"] = {"passed": False, "reason": str(error)}
            configs = [case for case in configs if case["preparation"] != "cpu_graph"]
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    rng = np.random.default_rng(19)
    for profile in args.profile:
        # Common device/kernel startup must not burden only the first codec.
        priming_start = time.perf_counter()
        priming_data = OrbMolDataset(args.cache / f"{profile}_gzip.npz", training_only=True)
        for repeat in range(args.repeats):
            torch.manual_seed(100 + repeat)
            priming_loader = make_loader(priming_data, workers=0, seed=31 + repeat,
                                         batch_size=args.batch_size, max_batches=1)
            batch = next(iter(priming_loader))
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
            for _ in range(3):
                training_step(model, optimizer, batch)
        torch.cuda.synchronize()
        priming_data.close()
        del optimizer, priming_loader, priming_data, batch
        report.setdefault("shared_priming_s", {})[profile] = time.perf_counter() - priming_start
        results, sequences = [], None
        for i in rng.permutation(len(configs)):
            config = configs[int(i)]
            index = mmap_indices[profile] if config["storage"] == "mmap" and local_cache else args.cache / f"{profile}_{config['storage']}.npz"
            result = run_case(model, initial, index, config,
                              steps=args.steps, warmup=args.warmup, repeats=args.repeats, batch_size=args.batch_size)
            current = [r["sample_sequence_sha256"] for r in result["runs"]]
            if sequences is not None and current != sequences:
                raise ValueError("Strategies trained on different sample sequences")
            sequences = current
            results.append(result)
            report["profiles"][profile] = results
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"profile": profile, **{k:v for k,v in result.items() if k != "runs"}}), flush=True)
    if args.profile_stages and args.pipeline:
        results = report["profiles"]["2_350"]
        winner = max(results, key=lambda result: result["aggregate_samples_per_s"])
        config = {key: value for key, value in winner.items()
                  if key in ("storage", "backend", "cache", "workers", "preparation", "fuse_pairs", "stratified", "batch_pairs")}
        report["stage_profile"] = profile_pipeline(model, initial, args.cache / "2_350_gzip.npz", config, args.batch_size)
    report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
    if not report["frozen_backbone_unchanged"]:
        raise RuntimeError("Frozen backbone changed")
    report["complete"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if local_cache:
        local_cache.cleanup()


if __name__ == "__main__":
    main()
