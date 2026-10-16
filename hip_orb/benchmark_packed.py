"""Bounded current-versus-packed comparisons on the same real geometry cohort."""

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import DataLoader

from hip_orb.benchmark_capacity import cohort_batches, cohort_groups
from hip_orb.benchmark_speed import head_options
from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, collate_samples, unpack_hessian
from hip_orb.execution import configure_execution
from hip_orb.graphs import (
    canonical_message_graph,
    deterministic_graph_audit,
    refresh_message_vectors,
)
from hip_orb.losses import hessian_error_loss
from hip_orb.model import HIPOrb
from hip_orb.packed_training import (
    clear_device_indices,
    collate_packed,
    collate_packed_compact,
    device_indices,
    packed_blocks,
    packed_graph,
    packed_loss,
    packed_update,
    prepare_packed,
)
from hip_orb.profile_batch_steps import read_pool
from hip_orb.profile_head import summarize_trace
from hip_orb.runner import train_update
from hip_orb.schedules import learning_rate


def mixed_batches(groups, batch_size=128, limit=4_000_000, window=8):
    """Precompute an interleaved plan within bounded shard-locality windows."""
    batches = []
    batch = []
    elements = 0
    for start in range(0, len(groups), window):
        parts = groups[start : start + window]
        for k in range(max(len(ids) for _, ids in parts)):
            for n, ids in parts:
                if k >= len(ids):
                    continue
                cost = 9 * n * n
                if batch and (len(batch) >= batch_size or elements + cost > limit):
                    batches.append(batch)
                    batch = []
                    elements = 0
                batch.append(ids[k])
                elements += cost
    if batch:
        batches.append(batch)
    return batches


def coverage(batches):
    flat = sorted(i for b in batches for i in b)
    return hashlib.sha256(np.asarray(flat, dtype=np.int64).tobytes()).hexdigest()


def audit(model, dataset, requested=(3, 20, 70)):
    samples = [read_pool(dataset, n, 1)[0][0] for n in requested]
    raw = prepare_packed(collate_packed(samples))
    sizes = raw["sizes"]
    molecules = []
    offset = 0
    for k, n in enumerate(sizes.tolist()):
        atoms = Atoms(
            numbers=raw["atomic_numbers"][offset : offset + n].numpy(),
            positions=raw["rotated_coords"][offset : offset + n].numpy(),
        )
        atoms.info.update(charge=int(raw["charge"][k]), spin=int(raw["spin"][k]))
        molecules.append(atoms)
        offset += n
    with deterministic_graph_audit():
        expected = refresh_message_vectors(
            canonical_message_graph(model.adapter.from_ase_atoms_list(molecules, device="cuda"))
        )
        actual = packed_graph(raw, model.adapter, "cuda")
        for name in ("senders", "receivers", "n_node"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), atol=0, rtol=0
            )
        torch.testing.assert_close(
            actual.edge_features["vectors"], expected.edge_features["vectors"], atol=0, rtol=0
        )
        feature = model.features(actual, sizes=sizes)
        reference = model.head(feature)
        indices = device_indices(sizes, "cuda")
        for key, expected_index in raw["indices"].items():
            if isinstance(expected_index, torch.Tensor):
                torch.testing.assert_close(
                    indices[key], expected_index.cuda(), atol=1e-8, rtol=1e-7
                )
            else:
                assert indices[key] == expected_index
        result = packed_blocks(model, feature, indices, sizes)
        expected_blocks = torch.cat(
            [
                h.reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1, 3, 3)
                for n, h in zip(sizes.tolist(), reference, strict=True)
            ]
        )
        torch.testing.assert_close(result, expected_blocks, atol=1e-4, rtol=1e-4)
        r = raw["rotation"].cuda()
        upper = raw["hessian_upper"].cuda()
        target = r @ upper[indices["upper_ids"]].reshape(-1, 3, 3) @ r.T
        target_list = []
        offset = 0
        for n in sizes.tolist():
            count = 3 * n * (3 * n + 1) // 2
            h = unpack_hessian(upper[offset : offset + count], n)
            h = torch.einsum("ab,ibjd,cd->iajc", r, h.reshape(n, 3, n, 3), r).reshape(3 * n, 3 * n)
            target_list.append(h)
            offset += count
        loss = packed_loss(result, target, indices)
        reference_loss = torch.stack(
            [
                hessian_error_loss((p - t)[None], "block_rmse")
                for p, t in zip(reference, target_list, strict=True)
            ]
        ).mean()
        torch.testing.assert_close(loss, reference_loss, atol=1e-4, rtol=1e-4)
        model.head.zero_grad(set_to_none=True)
        reference_loss.backward(retain_graph=True)
        grads = [None if p.grad is None else p.grad.clone() for p in model.head.parameters()]
        model.head.zero_grad(set_to_none=True)
        loss.backward()
        maximum = 0.0
        for p, g in zip(model.head.parameters(), grads, strict=True):
            if g is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, g, atol=1e-4, rtol=1e-4)
                maximum = max(maximum, float((p.grad - g).abs().max()))
    model.head.zero_grad(set_to_none=True)
    return {
        "passed": True,
        "sizes": sizes.tolist(),
        "graph_exact": True,
        "prediction_loss_gradient_tolerance": 1e-4,
        "max_gradient_difference": maximum,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-indices", action="store_true")
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--open-files", type=int, default=8)
    args = parser.parse_args()
    packed_collator = collate_packed if args.cpu_indices else collate_packed_compact
    if args.smoke:
        args.shards, args.repeats = 4, 1
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = json.loads(args.config.read_text())
    saved = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    dataset = OrbMolDataset(
        args.index, training_only=True, cache_bytes=args.cache_mib << 20, max_open=args.open_files
    )
    model = HIPOrb.pretrained(
        "cuda", weights_path=args.weights, precision="float32-high", **head_options(config)
    )
    configure_execution(model)
    model.train()
    assert saved["head_config"] == model.head.config
    model.head.load_state_dict(saved["head"])
    initial_hash = model_hash(model.base)
    report = {
        "complete": False,
        "config": config,
        "checkpoint_step": saved["global_step"],
        "gpu": torch.cuda.get_device_name(),
        "feature_cache": False,
        "index_mode": "cpu_transfer" if args.cpu_indices else "device_vectorized_cached",
        "loader_cache_mib_per_worker": args.cache_mib,
        "loader_max_open_files": args.open_files,
        "precision": "float32-high",
        "cases": [],
        "implementation_sha256": hashlib.sha256(
            b"".join(
                p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
            )
        ).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.report.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.report)

    try:
        report["audit"] = audit(model, dataset)
        save()
        groups = cohort_groups(dataset, args.shards)
        equal = cohort_batches(groups, 128, 4_000_000)
        started = time.perf_counter()
        mixed = mixed_batches(groups)
        plan_time = time.perf_counter() - started
        assert coverage(equal) == coverage(mixed)
        samples = sum(len(ids) for _, ids in groups)
        n_by_id = {i: n for n, ids in groups for i in ids}
        report.update(
            samples=samples,
            coverage_sha256=coverage(equal),
            mixed_plan_seconds=plan_time,
            mixed_batches=len(mixed),
            equal_batches=len(equal),
            mixed_size_batch_fraction=sum(len({n_by_id[i] for i in b}) > 1 for b in mixed)
            / len(mixed),
            max_mixed_hessian_elements=max(sum(9 * n_by_id[i] ** 2 for i in b) for b in mixed),
        )
        save()
        warm = [dataset[i] for i in equal[0]]
        for label, plan, packed in [
            ("current_first", equal, False),
            ("packed_equal", equal, True),
            ("packed_mixed", mixed, True),
            ("current_last", equal, False),
        ]:
            repeats = []
            for repeat in range(args.repeats):
                model.head.load_state_dict(saved["head"])
                model.head.zero_grad(set_to_none=True)
                opt = torch.optim.AdamW(model.head.parameters(), lr=config["lr"], weight_decay=0)
                opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                update = packed_update if packed else train_update
                warm_batch = packed_collator(warm) if packed else collate_samples(warm)
                for _ in range(3):
                    update(
                        model,
                        opt,
                        warm_batch,
                        device="cuda",
                        seed=config.get("seed", 7),
                        loss_type=config["loss_type"],
                    )
                model.head.load_state_dict(saved["head"])
                opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                loader = DataLoader(
                    dataset,
                    batch_sampler=plan,
                    collate_fn=packed_collator if packed else collate_samples,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True,
                    multiprocessing_context="spawn",
                    prefetch_factor=2,
                )
                iterator = iter(loader)
                current = next(iterator)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                wait = 0.0
                seen = 0
                try:
                    for step in range(len(plan)):
                        if step:
                            t = time.perf_counter()
                            current = next(iterator)
                            wait += time.perf_counter() - t
                        lr = learning_rate(
                            saved["global_step"] + step + 1,
                            config["lr"],
                            config["lr_schedule"],
                            config["max_steps"],
                            config["warmup_fraction"],
                        )
                        for g in opt.param_groups:
                            g["lr"] = lr
                        _loss, b = update(
                            model,
                            opt,
                            current,
                            device="cuda",
                            seed=config.get("seed", 7),
                            loss_type=config["loss_type"],
                        )
                        seen += b
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    assert seen == samples
                    repeats.append(
                        {
                            "elapsed_s": elapsed,
                            "geometries_per_s": seen / elapsed,
                            "ms_per_update": 1000 * elapsed / len(plan),
                            "loader_wait_percent": 100 * wait / elapsed,
                            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                        }
                    )
                finally:
                    iterator._shutdown_workers()
            row = {
                "label": label,
                "repeats": repeats,
                "geometries_per_s": float(np.mean([r["geometries_per_s"] for r in repeats])),
            }
            report["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
            clear_device_indices()
            del opt, loader, iterator, current, warm_batch
            model.head.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        # Profile each path separately after the throughput comparison.
        report["profiles"] = []
        for label, plan, packed in [
            ("current", equal, False),
            ("packed_equal", equal, True),
            ("packed_mixed", mixed, True),
        ]:
            opt = torch.optim.AdamW(model.head.parameters(), lr=config["lr"], weight_decay=0)
            update = packed_update if packed else train_update
            selected = max(plan, key=lambda b: len({n_by_id[i] for i in b})) if packed else plan[0]
            batch = (packed_collator if packed else collate_samples)(dataset.__getitems__(selected))
            for _ in range(3):
                update(
                    model,
                    opt,
                    batch,
                    device="cuda",
                    seed=config.get("seed", 7),
                    loss_type=config["loss_type"],
                )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as trace:
                for _ in range(3):
                    update(
                        model,
                        opt,
                        batch,
                        device="cuda",
                        seed=config.get("seed", 7),
                        loss_type=config["loss_type"],
                    )
                    trace.step()
                torch.cuda.synchronize()
            report["profiles"].append(
                {
                    "label": label,
                    "sizes": [n_by_id[i] for i in selected],
                    "trace": summarize_trace(trace),
                }
            )
            model.head.zero_grad(set_to_none=True)
            del opt, batch
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report["frozen_backbone_unchanged"] = initial_hash == model_hash(model.base)
        report["backbone_has_gradients"] = any(p.grad is not None for p in model.base.parameters())
        assert report["frozen_backbone_unchanged"] and not report["backbone_has_gradients"]
        report["complete"] = True
    finally:
        dataset.close()
        save()


if __name__ == "__main__":
    main()
