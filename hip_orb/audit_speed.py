"""Compare speed candidates with eager predictions and all trainable gradients."""

import argparse
import copy
import json
from pathlib import Path

import torch

from hip_orb.acceleration import speed_candidate
from hip_orb.benchmark_speed import head_options
from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, collate_samples, unpack_hessian
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.losses import hessian_error_loss, rotate_hessian
from hip_orb.model import HIPOrb
from hip_orb.preparation import prepare_batch
from hip_orb.profile_batch_steps import read_pool


def discrepancy(actual, expected):
    return {"max_abs": float((actual - expected).abs().max()) if actual.numel() else 0.0,
            "relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--atoms", type=int, nargs="+", default=[20, 70, 350])
    parser.add_argument("--head-checkpoints", type=Path, nargs="+")
    parser.add_argument("--full-backbone", action="store_true")
    parser.add_argument("--precision", choices=("float32-high", "float32-highest"), default="float32-highest")
    parser.add_argument("--production-batches", action="store_true")
    args = parser.parse_args()
    if args.head_checkpoints is not None and len(args.head_checkpoints) != len(args.configs):
        parser.error("Supply one head checkpoint per configuration")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    report = {"complete": False, "checks": [], "precision": args.precision,
              "deterministic": True, "atol": 1e-5, "rtol": 1e-3,
              "head_checks_use_identical_frozen_features": not args.full_backbone,
              "frozen_feature_cache_used_only_in_numerical_audit": not args.full_backbone}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    dataset = OrbMolDataset(args.index, training_only=True)
    base = adapter = None
    try:
        for config_index, config_path in enumerate(args.configs):
            config = json.loads(config_path.read_text())
            torch.manual_seed(7)
            if base is None:
                model = HIPOrb.pretrained("cuda", weights_path=args.weights, precision=args.precision,
                                          **head_options(config))
                base, adapter = model.base, model.adapter
                frozen_hash = model_hash(base)
            else:
                model = HIPOrb(base, adapter, **head_options(config))
            model.train()
            if args.head_checkpoints is not None:
                saved = torch.load(args.head_checkpoints[config_index], map_location="cuda", weights_only=True)
                if saved["head_config"] != model.head.config:
                    raise ValueError("Audit checkpoint architecture differs")
                model.head.load_state_dict(saved["head"])
            initial = copy.deepcopy(model.head.state_dict())
            for requested in args.atoms:
                limit = (min(config["batch_size"], max(1, config["max_hessian_elements"] // (3 * requested)**2))
                         if args.production_batches else 2)
                samples, n = read_pool(dataset, requested, limit)
                batch = prepare_batch(collate_samples(samples), seed=7)
                graph = refresh_message_vectors(canonical_message_graph(
                    adapter.from_ase_atoms_list(batch["molecules"], device="cuda")))
                sizes = torch.full((len(samples),), n, dtype=torch.long)
                target = rotate_hessian(unpack_hessian(batch["hessian_upper"].to("cuda"), n),
                                        batch["rotation"])

                original_forward = base.model.forward
                with torch.no_grad():
                    # Warm first-use libraries before recording the reference.
                    base.model(graph)
                    frozen_features = {key: value.detach() for key, value in base.model(graph).items()}

                def evaluate(cached=True, model=model, initial=initial, base=base,
                             frozen_features=frozen_features, graph=graph, sizes=sizes,
                             original_forward=original_forward, target=target, config=config):
                    model.head.load_state_dict(initial)
                    model.head.zero_grad(set_to_none=True)
                    try:
                        if cached:
                            base.model.forward = lambda graph: frozen_features
                        features = model.features(graph.clone(), sizes=sizes)
                    finally:
                        base.model.forward = original_forward
                    if getattr(model.head, "_tensor_training", False):
                        predicted = model.head.forward_tensor(features)
                    else:
                        predicted = torch.stack(model.head(features))
                    loss = hessian_error_loss(predicted - target, config["loss_type"])
                    gradients = torch.autograd.grad(loss, tuple(model.head.parameters()))
                    return (features.nodes.detach(), features.edges.detach(), predicted.detach(),
                            loss.detach(), torch.cat([g.flatten() for g in gradients]).detach())

                if args.full_backbone:
                    evaluate(cached=False)
                expected = evaluate(cached=not args.full_backbone)
                for candidate in ["eager", *args.candidates]:
                    changes_backbone = args.full_backbone
                    reference = evaluate(cached=False) if changes_backbone else expected
                    with speed_candidate(model, candidate):
                        actual = evaluate(cached=not changes_backbone)
                    row = {"config": config_path.name, "natoms": n, "candidate": candidate}
                    for name, a, b in zip(("nodes", "edges", "hessian", "loss", "gradients"),
                                           actual, reference, strict=True):
                        row[name] = discrepancy(a, b)
                        try:
                            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-3)
                        except AssertionError as error:
                            row.setdefault("failures", {})[name] = str(error)
                    row["passed"] = not row.get("failures")
                    report["checks"].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if k != "failures"}), flush=True)
            del model, initial
        report["frozen_backbone_unchanged"] = model_hash(base) == frozen_hash
        report["frozen_backbone_has_gradients"] = any(p.grad is not None for p in base.parameters())
        report["all_passed"] = all(row["passed"] for row in report["checks"])
        report["complete"] = True
    finally:
        dataset.close()
        save()


if __name__ == "__main__":
    main()
