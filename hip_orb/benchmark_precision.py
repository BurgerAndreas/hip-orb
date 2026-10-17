"""Audit low-precision linear operations, then measure full packed updates."""

import argparse
import copy
import gc
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_masses

from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.benchmark_speed import head_options
from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, make_loader
from hip_orb.execution import configure_execution
from hip_orb.graphs import refresh_message_vectors
from hip_orb.metrics import hessian_metrics
from hip_orb.mixed_precision import LinearPolicy, linear_precision
from hip_orb.model import HIPOrb
from hip_orb.packed_training import (
    clear_device_indices,
    collate_packed_compact,
    device_indices,
    packed_blocks,
    packed_graph,
    packed_loss,
    prepare_packed,
)
from hip_orb.precision_audit import (
    comparison,
    fused_operator_audit,
    operator_audit,
    scaled_operator_audit,
)


def cases():
    p = LinearPolicy
    return [
        ("fp32", p(), p(), False, "high"),
        ("fp32_highest", p(), p(), False, "highest"),
        ("bf16_head", p("bf16"), p(), False, "high"),
        ("bf16_head_activations", p("bf16", output_dtype="bf16"), p(), False, "high"),
        ("bf16_head_outputs", p("bf16"), p(), True, "high"),
        ("bf16_backbone", p(), p("bf16"), False, "high"),
        ("bf16_both", p("bf16"), p("bf16"), False, "high"),
        ("fp16_head", p("fp16"), p(), False, "high"),
        ("fp8_head_bwd_bf16", p("fp8", "bf16"), p(), False, "high"),
        ("fp8_head_bwd_e4m3", p("fp8", "fp8_e4m3"), p(), False, "high"),
        ("fp8_head_bwd_e5m2", p("fp8", "fp8_e5m2"), p(), False, "high"),
        ("fp8_head_outputs", p("fp8", "fp8_e4m3"), p(), True, "high"),
        ("fp8_backbone", p(), p("fp8", "fp8_e4m3"), False, "high"),
        ("fp8_both_bwd_bf16", p("fp8", "bf16"), p("fp8", "bf16"), False, "high"),
        ("fp8_both_bwd_e4m3", p("fp8", "fp8_e4m3"), p("fp8", "fp8_e4m3"), False, "high"),
        ("bf16_head_middle", p("bf16", selection="square"), p(), False, "high"),
        ("bf16_f32_head", p("bf16_f32"), p(), False, "high"),
        ("bf16_f32_head_middle", p("bf16_f32", selection="square"), p(), False, "high"),
        ("bf16_f32_backbone", p(), p("bf16_f32"), False, "high"),
        ("bf16_f32_both", p("bf16_f32"), p("bf16_f32"), False, "high"),
        ("fp16_f32_head", p("fp16_f32"), p(), False, "high"),
        ("fp16_fused_head", p("fp16_f32", fuse_silu=True), p(), False, "high"),
        (
            "fp16_fused_activations_head",
            p("fp16_f32", fuse_silu=True, activation_dtype="fp16"),
            p(),
            False,
            "high",
        ),
        ("fp16_fused_both", p("fp16_f32", fuse_silu=True), p("fp16_f32"), False, "high"),
        (
            "fp16_fused_activations_both",
            p("fp16_f32", fuse_silu=True, activation_dtype="fp16"),
            p("fp16_f32"),
            False,
            "high",
        ),
        ("fp16_f32_backbone", p(), p("fp16_f32"), False, "high"),
        ("fp16_f32_both", p("fp16_f32"), p("fp16_f32"), False, "high"),
        ("fp16_native_backbone", p("fp16_f32", fuse_silu=True), p("fp16"), False, "high"),
        (
            "fp16_native_no_gates",
            p("fp16_f32", fuse_silu=True),
            p("fp16", exclude=("*attn*",)),
            False,
            "high",
        ),
        (
            "fp16_native_mlp_activations",
            p("fp16_f32", fuse_silu=True),
            p("fp16", include=("base.model.*.mlp.NN-*",), activation_dtype="fp16"),
            False,
            "high",
        ),
        (
            "fp16_native_message_activations",
            p("fp16_f32", fuse_silu=True),
            p("fp16", include=("base.model.gnn_stacks.*._*mlp.mlp.NN-*",), activation_dtype="fp16"),
            False,
            "high",
        ),
        (
            "fp16_native_message_activations_half_head",
            p("fp16_f32", fuse_silu=True, activation_dtype="fp16"),
            p("fp16", include=("base.model.gnn_stacks.*._*mlp.mlp.NN-*",), activation_dtype="fp16"),
            False,
            "high",
        ),
        (
            "fp16_native_edges_half_head",
            p("fp16_f32", fuse_silu=True, activation_dtype="fp16"),
            p(
                "fp16",
                include=("base.model.gnn_stacks.*._edge_mlp.mlp.NN-*",),
                activation_dtype="fp16",
            ),
            False,
            "high",
        ),
        (
            "fp16_native_nodes_half_head",
            p("fp16_f32", fuse_silu=True, activation_dtype="fp16"),
            p(
                "fp16",
                include=("base.model.gnn_stacks.*._node_mlp.mlp.NN-*",),
                activation_dtype="fp16",
            ),
            False,
            "high",
        ),
        ("fp16_scaled_head", p("fp16_scaled"), p(), False, "high"),
        ("fp16_scaled_backbone", p(), p("fp16_scaled"), False, "high"),
        ("fp16_scaled_both", p("fp16_scaled"), p("fp16_scaled"), False, "high"),
        ("fp8_row_head", p("fp8_row"), p(), False, "high"),
        ("fp8_row_backbone", p(), p("fp8_row"), False, "high"),
        ("bf16_compensated_backbone", p(), p("bf16_compensated"), False, "high"),
        ("bf16_compensated_head", p("bf16_compensated"), p(), False, "high"),
        ("fp8_head_middle_bwd_bf16", p("fp8", "bf16", selection="square"), p(), False, "high"),
        ("fp8_head_middle_bwd_e4m3", p("fp8", "fp8_e4m3", selection="square"), p(), False, "high"),
        ("fp8_backbone_middle", p(), p("fp8", "fp8_e4m3", selection="square"), False, "high"),
        (
            "fp8_both_middle",
            p("fp8", "fp8_e4m3", selection="square"),
            p("fp8", "fp8_e4m3", selection="square"),
            False,
            "high",
        ),
        *[
            (
                f"fp16_backbone_round{round_id}",
                p("fp16_f32"),
                p("fp16_f32", include=(f"base.model.gnn_stacks.{round_id}.*",)),
                False,
                "high",
            )
            for round_id in range(5)
        ],
        ("fp16_backbone_middle", p("fp16_f32"), p("fp16_f32", selection="square"), False, "high"),
        (
            "fp16_backbone_encoder",
            p("fp16_f32"),
            p("fp16_f32", include=("base.model._encoder.*",)),
            False,
            "high",
        ),
        (
            "fp16_backbone_decoder",
            p("fp16_f32"),
            p("fp16_f32", include=("base.model._decoder.*",)),
            False,
            "high",
        ),
        ("fp32_repeat", p(), p(), False, "high"),
    ]


@contextmanager
def policy_context(model, case):
    _, head, backbone, outputs, matmul = case
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(matmul)
    try:
        with linear_precision(
            model, head=head, backbone=backbone, include_outputs=outputs
        ) as stats:
            yield stats
    finally:
        torch.set_float32_matmul_precision(previous)


def gradients(module):
    return {
        name: None if p.grad is None else p.grad.detach().float().cpu().clone()
        for name, p in module.named_parameters()
    }


def compare_gradients(actual, reference):
    presence = all((actual[k] is None) == (v is None) for k, v in reference.items())
    matched = [k for k, v in reference.items() if v is not None and actual[k] is not None]
    result = {"presence_matches": presence, "parameters_with_gradients": len(matched)}
    result["global"] = comparison(
        torch.cat([actual[k].reshape(-1) for k in matched]),
        torch.cat([reference[k].reshape(-1) for k in matched]),
    )
    result["per_parameter"] = {k: comparison(actual[k], reference[k]) for k in matched}
    return result


def forward(model, batch):
    prepared = prepare_packed(batch, seed=7)
    graph = packed_graph(prepared, model.adapter, "cuda")
    indices = device_indices(batch["sizes"], "cuda")
    graph._hip_dense_topology = indices["dense_a"], indices["dense_b"]
    features = model.features(graph, sizes=batch["sizes"])
    blocks = packed_blocks(model, features, indices, batch["sizes"]).float()
    r = prepared["rotation"].cuda()
    target = r @ batch["hessian_upper"].cuda()[indices["upper_ids"]].reshape(-1, 3, 3) @ r.T
    return packed_loss(blocks, target.float(), indices), blocks, target, prepared, features


def snapshot(model, batch, scale=1):
    model.head.zero_grad(set_to_none=True)
    loss, blocks, _, _, feature = forward(model, batch)
    (loss * scale).backward()
    if scale != 1:
        for p in model.head.parameters():
            if p.grad is not None:
                p.grad.div_(scale)
    result = {
        "loss": float(loss.detach()),
        "blocks": blocks.detach().cpu(),
        "nodes": feature.nodes.detach().float().cpu(),
        "edges": feature.edges.detach().float().cpu(),
        "charges": feature.charges.detach().float().cpu(),
        "gradients": gradients(model.head),
    }
    model.head.zero_grad(set_to_none=True)
    return result


def backbone_snapshot(model, batch):
    base = model.base.model
    flags = [p.requires_grad for p in base.parameters()]
    base.requires_grad_(True)
    base.zero_grad(set_to_none=True)
    try:
        prepared = prepare_packed(batch, seed=7)
        graph = packed_graph(prepared, model.adapter, "cuda")
        graph.node_features = dict(graph.node_features)
        graph.edge_features = dict(graph.edge_features)
        positions = graph.node_features["positions"].detach().requires_grad_(True)
        graph.node_features["positions"] = positions
        refresh_message_vectors(graph)
        out = base(graph)
        gen = torch.Generator(device="cuda").manual_seed(913)
        nodes, edges = out["node_features"], out["edge_features"]
        probe = (
            nodes
            * torch.randn(nodes.shape, device="cuda", generator=gen)
            / math.sqrt(nodes.numel())
        ).sum()
        probe = (
            probe
            + (
                edges
                * torch.randn(edges.shape, device="cuda", generator=gen)
                / math.sqrt(edges.numel())
            ).sum()
        )
        probe.backward()
        return {
            "gradients": gradients(base),
            "positions": positions.grad.detach().float().cpu(),
            "nodes": nodes.detach().float().cpu(),
            "edges": edges.detach().float().cpu(),
        }
    finally:
        base.zero_grad(set_to_none=True)
        for p, flag in zip(base.parameters(), flags, strict=True):
            p.requires_grad_(flag)


def pick_rows(dataset, requested, count):
    result = []
    for target in requested:
        shard = int(np.abs(dataset.natoms - target).argmin())
        available = int(dataset.ptr[shard + 1] - dataset.ptr[shard])
        result.extend(
            range(int(dataset.ptr[shard]), int(dataset.ptr[shard]) + min(count, available))
        )
    return result


@contextmanager
def evaluation_mode(model):
    previous = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(previous)


def evaluate(model, batches):
    totals = {}
    samples = 0
    with evaluation_mode(model), torch.no_grad():
        for batch in batches:
            _, blocks, target, prepared, _ = forward(model, batch)
            if not torch.isfinite(blocks).all():
                raise FloatingPointError("Nonfinite validation Hessian prediction")
            offset = atom_offset = 0
            for n in batch["sizes"].tolist():

                def matrix(value, offset=offset, n=n):
                    return (
                        value[offset : offset + n * n]
                        .reshape(n, n, 3, 3)
                        .permute(0, 2, 1, 3)
                        .reshape(3 * n, 3 * n)
                        .double()
                    )

                pos = prepared["rotated_coords"][atom_offset : atom_offset + n].cuda().double()
                z = batch["atomic_numbers"][atom_offset : atom_offset + n].long().numpy()
                masses = torch.tensor(atomic_masses[z], device="cuda", dtype=torch.float64)
                row = hessian_metrics(matrix(blocks), matrix(target), pos, masses)
                for k, v in row.items():
                    totals[k] = totals.get(k, 0) + v
                samples += 1
                offset += n * n
                atom_offset += n
    return {"samples": samples, "metrics": {k: v / samples for k, v in totals.items()}}


def training_update(model, opt, scaler, batch):
    opt.zero_grad(set_to_none=True)
    loss, *_ = forward(model, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach()), int(scaler.get_scale() < before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "validation-index", "weights", "config", "checkpoint", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case-labels", nargs="+", choices=[case[0] for case in cases()])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--validation-samples", type=int, default=48)
    parser.add_argument("--max-batch-atoms", type=int, default=2560)
    parser.add_argument("--stability-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.validation_samples, args.max_batch_atoms) < 1:
        parser.error("Step, repeat, and validation counts must be positive")
    if args.stability_steps < 0:
        parser.error("Stability steps must be nonnegative")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = json.loads(args.config.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = HIPOrb.pretrained("cuda", weights_path=args.weights, **head_options(config))
    configure_execution(model)
    model.train()
    initial_hash = model_hash(model.base)
    dataset = OrbMolDataset(args.index, training_only=True, cache_bytes=1024 << 20, max_open=16)
    validation = OrbMolDataset(
        args.validation_index,
        training_only=True,
        cache_bytes=512 << 20,
        max_open=8,
        cache_mode="arrays" if args.validation_samples == 48 else "chunks",
    )
    assert not set(dataset.sources) & set(validation.sources)
    # Audit three fixed batches, including the largest molecule and a single geometry.
    audit_batches = [
        collate_packed_compact(dataset.__getitems__(pick_rows(dataset, sizes, 1)))
        for sizes in ((3, 20, 70), (20, 150, 350), (350,))
    ]
    # Build a bounded representative cohort; cache host labels, never backbone features.
    selected = np.random.default_rng(7).permutation(len(dataset.paths))[:48]
    cohort = copy.copy(dataset)
    cohort._reset()
    counts = np.diff(dataset.ptr)[selected]
    cohort.paths = [dataset.paths[s] for s in selected]
    cohort.sources = [dataset.sources[s] for s in selected]
    cohort.natoms = dataset.natoms[selected]
    cohort.ptr = np.r_[0, np.cumsum(counts)]
    sampler = BalancedBatchSampler(cohort, max_atoms=args.max_batch_atoms, batch_size=64)
    plan = list(sampler)[: args.steps]
    resident = [collate_packed_compact(cohort.__getitems__(ids)) for ids in plan]
    loader = (
        make_loader(
            cohort,
            workers=4,
            prefetch_factor=4,
            batch_sampler=plan,
            collate_fn=collate_packed_compact,
            pin_memory=True,
        )
        if args.streaming
        else None
    )
    if args.validation_samples == 48:
        val_ids = pick_rows(validation, (3, 20, 70, 150, 250, 350), 8)
        val_groups = [val_ids[i : i + 8] for i in range(0, len(val_ids), 8)]
        validation_sampling = "stratified held-out diagnostic"
    else:
        val_ids = np.sort(
            np.random.default_rng(914).choice(
                len(validation),
                min(args.validation_samples, len(validation)),
                replace=False,
            )
        )
        val_groups = []
        group, atoms, elements = [], 0, 0
        for index in val_ids:
            shard = int(np.searchsorted(validation.ptr, index, side="right") - 1)
            n = int(validation.natoms[shard])
            if group and (
                len(group) >= 256 or atoms + n > 2560 or elements + 9 * n * n > 8_000_000
            ):
                val_groups.append(group)
                group, atoms, elements = [], 0, 0
            group.append(int(index))
            atoms += n
            elements += 9 * n * n
        if group:
            val_groups.append(group)
        validation_sampling = "uniform held-out geometries without replacement; seed 914"
    val_batches = [collate_packed_compact(validation.__getitems__(ids)) for ids in val_groups]
    report = {
        "complete": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint_step": saved["global_step"],
        "head_config": model.head.config,
        "batch_limits": {"geometries": 64, "atoms": args.max_batch_atoms, "hessian_elements": None},
        "cases": [],
        "resident_geometry_count": sum(len(b["sizes"]) for b in resident),
        "timed_source_reads": args.streaming,
        "startup_excluded": "worker startup and first batch read; model warmup uses one complete resident pass",
        "diagnostic_validation": validation_sampling
        + "; FP64 mass weighting, Eckart projection and eigensolver",
        "validation_samples": len(val_ids),
        "validation_cache_mode": validation.cache_mode,
        "gate": {
            "gradient_cosine_min": 0.999,
            "gradient_relative_l2_max": 0.05,
            "loss_relative_difference_max": 0.01,
            "prediction_relative_l2_max": 0.01,
        },
        "fp8_backward": "Straight-through surrogate; first-order gradients only",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.report.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.report)

    report["operator_audit"] = operator_audit()
    report["scaled_operator_audit"] = scaled_operator_audit()
    report["fused_operator_audit"] = fused_operator_audit()
    save()
    model.head.load_state_dict(saved["head"])
    references = [snapshot(model, b) for b in audit_batches]
    backbone_reference = backbone_snapshot(model, audit_batches[0])
    scaler = None
    try:
        for case in cases():
            label, head, backbone, outputs, matmul = case
            if args.case_labels and label not in args.case_labels:
                continue
            row = {
                "label": label,
                "head": asdict(head),
                "backbone": asdict(backbone),
                "include_output_layers": outputs,
                "matmul": matmul,
            }
            try:
                model.head.load_state_dict(saved["head"])
                with policy_context(model, case) as stats:
                    audit = []
                    for batch, ref in zip(audit_batches, references, strict=True):
                        got = snapshot(
                            model, batch, scale=1024 if head.dtype.startswith("fp16") else 1
                        )
                        gradient = compare_gradients(got["gradients"], ref["gradients"])
                        r = {
                            "sizes": batch["sizes"].tolist(),
                            "loss": got["loss"],
                            "loss_relative_difference": abs(got["loss"] - ref["loss"])
                            / max(abs(ref["loss"]), 1e-20),
                            "predictions": comparison(got["blocks"], ref["blocks"]),
                            "gradients": gradient,
                            "nodes": comparison(got["nodes"], ref["nodes"]),
                            "edges": comparison(got["edges"], ref["edges"]),
                            "charges": comparison(got["charges"], ref["charges"]),
                        }
                        g = gradient["global"]
                        r["quality_gate_passed"] = (
                            gradient["presence_matches"]
                            and g["finite"]
                            and g["cosine"] >= 0.999
                            and g["relative_l2"] <= 0.05
                            and r["loss_relative_difference"] <= 0.01
                            and r["predictions"]["finite"]
                            and r["predictions"]["relative_l2"] <= 0.01
                        )
                        audit.append(r)
                    row["audit"] = audit
                    row["quality_gate_passed"] = all(r["quality_gate_passed"] for r in audit)
                    if backbone.dtype != "fp32" or label in ("fp32", "fp32_highest", "fp32_repeat"):
                        got = backbone_snapshot(model, audit_batches[0])
                        row["backbone_backward"] = {
                            "gradients": compare_gradients(
                                got["gradients"], backbone_reference["gradients"]
                            ),
                            "position_gradient": comparison(
                                got["positions"], backbone_reference["positions"]
                            ),
                        }
                        bg = row["backbone_backward"]["gradients"]
                        pg = row["backbone_backward"]["position_gradient"]
                        row["backbone_backward"]["quality_gate_passed"] = (
                            bg["presence_matches"]
                            and bg["global"]["finite"]
                            and bg["global"]["cosine"] >= 0.999
                            and bg["global"]["relative_l2"] <= 0.05
                            and pg["finite"]
                            and pg["cosine"] >= 0.999
                            and pg["relative_l2"] <= 0.05
                        )
                    row["validation"] = evaluate(model, val_batches)
                    row["timings"] = []
                    for repeat in range(args.repeats):
                        model.head.load_state_dict(saved["head"])
                        opt = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0)
                        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                        scaler = torch.amp.GradScaler(
                            "cuda", enabled=head.dtype.startswith("fp16"), init_scale=1024
                        )

                        warmup_started = time.perf_counter()
                        for b in resident:
                            training_update(model, opt, scaler, b)
                        torch.cuda.synchronize()
                        warmup_seconds = time.perf_counter() - warmup_started
                        model.head.load_state_dict(saved["head"])
                        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                        timed_batches = resident
                        if loader is not None:
                            iterator = iter(loader)
                            timed_batches = chain([next(iterator)], iterator)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        skipped = 0
                        for b in timed_batches:
                            _, was_skipped = training_update(model, opt, scaler, b)
                            skipped += was_skipped
                        torch.cuda.synchronize()
                        duration = time.perf_counter() - started
                        row["timings"].append(
                            {
                                "seconds": duration,
                                "geometries_per_s": report["resident_geometry_count"] / duration,
                                "ms_per_update": 1000 * duration / len(resident),
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "optimizer_steps_skipped": skipped,
                                "warmup_seconds": warmup_seconds,
                            }
                        )
                        del opt
                    if args.stability_steps:
                        model.head.load_state_dict(saved["head"])
                        opt = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0)
                        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                        scaler = torch.amp.GradScaler(
                            "cuda",
                            enabled=head.dtype.startswith("fp16"),
                            init_scale=1024,
                        )
                        losses, skipped, geometries = [], 0, 0
                        for step in range(args.stability_steps):
                            batch = resident[step % len(resident)]
                            loss_value, skip = training_update(model, opt, scaler, batch)
                            losses.append(loss_value)
                            skipped += skip
                            geometries += len(batch["sizes"])
                        window = min(len(resident), len(losses))
                        row["stability"] = {
                            "steps": len(losses),
                            "geometries": geometries,
                            "optimizer_steps_skipped": skipped,
                            "first_window_mean_loss": sum(losses[:window]) / window,
                            "last_window_mean_loss": sum(losses[-window:]) / window,
                            "post_training_validation": evaluate(model, val_batches),
                            "repeated_cohort": True,
                        }
                        del opt
                    row["coverage"] = {k: v for k, v in stats.items()}
                    row["frozen_gradients_absent"] = all(
                        p.grad is None for p in model.base.parameters()
                    )
                    row["complete"] = True
            except Exception as error:  # noqa: BLE001
                # Triton compiler exceptions do not inherit RuntimeError.
                row.update(complete=False, error=str(error), error_type=type(error).__name__)
            finally:
                report["cases"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            "label": label,
                            "complete": row.get("complete"),
                            "quality_gate_passed": row.get("quality_gate_passed"),
                            "error": row.get("error"),
                            "timings": row.get("timings"),
                        }
                    ),
                    flush=True,
                )
                model.head.zero_grad(set_to_none=True)
                model.base.zero_grad(set_to_none=True)
                clear_device_indices()
                gc.collect()
                torch.cuda.empty_cache()
        report["frozen_state_unchanged"] = initial_hash == model_hash(model.base)
        assert report["frozen_state_unchanged"]
        report["complete"] = True
        save()
    finally:
        dataset.close()
        cohort.close()
        validation.close()


if __name__ == "__main__":
    main()
