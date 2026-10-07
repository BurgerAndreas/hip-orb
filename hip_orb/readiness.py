"""Bounded real-checkpoint HIP checks, without using external validation labels."""
import argparse
import copy
import json
from pathlib import Path

from ase import Atoms
import numpy as np
import torch

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.data import OrbMolDataset, collate_samples, unpack_hessian
from hip_orb.model import HIPOrb


def molecules(batch):
    result = []
    for i in range(len(batch["coords"])):
        atoms = Atoms(numbers=batch["atomic_numbers"][i].numpy(), positions=batch["coords"][i].numpy())
        atoms.info.update(charge=int(batch["charge"][i]), spin=int(batch["spin"][i]))
        result.append(atoms)
    return result


def primary_batch(dataset, desired, count=1):
    shard = int(np.abs(dataset.natoms - desired).argmin())
    start, end = (int(value) for value in dataset.ptr[shard:shard + 2])
    return collate_samples([dataset[i] for i in range(start, min(end, start + count))])


def verify_checkpoint_roundtrip(model, directory, batch):
    path = directory / "test_head.pt"
    original = copy.deepcopy(model.head.state_dict())
    model.eval()
    graph = lambda: model.adapter.from_ase_atoms_list(molecules(batch), device="cuda")
    with torch.no_grad():
        before = model(graph())
    torch.save({"head": original, "config": model.head.config}, path)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    if checkpoint["config"] != model.head.config:
        raise RuntimeError("Checkpoint configuration differs")
    model.head.load_state_dict(checkpoint["head"])
    with torch.no_grad():
        after = model(graph())
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overfit-steps", type=int, default=500)
    args = parser.parse_args()
    if args.overfit_steps < 1:
        raise ValueError("Overfit step count must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    dataset = OrbMolDataset(args.index, training_only=True)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    initial = copy.deepcopy(model.head.state_dict())
    original_hash = model_hash(model.base)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "full_training_started": False, "implementation_checks": [],
              "external_validation_pending": True, "molecule_disjoint_split_verified": False,
              "full_training_allowed": False}
    path = args.output / "readiness.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for requested in (2, 50, 70, 110, 160, 250, 350):
        model.head.load_state_dict(initial)
        batch = primary_batch(dataset, requested)
        atoms = molecules(batch)
        graph = lambda: model.adapter.from_ase_atoms_list(atoms, device="cuda")
        model.base.eval()
        torch.cuda.reset_peak_memory_stats()
        before = {key: value.detach().clone() for key, value in model.base.predict(graph(), split=False).items()
                  if key in ("energy", "forces", "grad_forces")}
        if "energy" not in before or not ({"forces", "grad_forces"} & set(before)):
            raise RuntimeError("Original model does not expose energy and forces")
        model.train()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
        loss = training_step(model, optimizer, batch)
        model.eval()
        with torch.no_grad():
            predicted = model(graph())[0]
        if not torch.isfinite(predicted).all():
            raise RuntimeError("Predicted Hessian is not finite")
        torch.testing.assert_close(predicted, predicted.T, atol=0, rtol=0)
        after = model.base.predict(graph(), split=False)
        for key, value in before.items():
            torch.testing.assert_close(after[key], value, atol=1e-6, rtol=1e-6)
        if any(parameter.grad is not None for parameter in model.base.parameters()):
            raise RuntimeError("Frozen backbone has gradients")
        report["implementation_checks"].append({"requested_natoms": requested, "actual_natoms": len(atoms[0]),
                                                "finite_training_loss": float(loss), "symmetric": True,
                                                "energy_force_unchanged": True,
                                                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                                                "backbone_has_no_gradients": True})
        save()
        print(json.dumps(report["implementation_checks"][-1]), flush=True)
    # This expressivity check may cache features. Timed training never does.
    model.head.load_state_dict(initial)
    batch = primary_batch(dataset, 50, count=4)
    graph = model.adapter.from_ase_atoms_list(molecules(batch), device="cuda")
    feature = model.features(graph)
    target = unpack_hessian(batch["hessian_upper"].to("cuda"), batch["coords"].shape[1])
    model.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-3, weight_decay=0)
    with torch.no_grad():
        initial_mse = float((torch.stack(model.head(feature)) - target).square().mean())
    for _ in range(args.overfit_steps):
        loss = (torch.stack(model.head(feature)) - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Overfit loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10)
        optimizer.step()
    with torch.no_grad():
        final_mse = float((torch.stack(model.head(feature)) - target).square().mean())
        relative_rmse = (final_mse / float(target.square().mean().clamp_min(1e-12)))**0.5
    report["overfit"] = {"samples": len(batch["coords"]), "steps": args.overfit_steps,
                         "initial_mse": initial_mse, "final_mse": final_mse,
                         "relative_rmse": relative_rmse, "passed": relative_rmse < 0.05}
    report["checkpoint_roundtrip_passed"] = verify_checkpoint_roundtrip(model, args.output, batch)
    report["frozen_backbone_unchanged"] = model_hash(model.base) == original_hash
    if not report["frozen_backbone_unchanged"]:
        raise RuntimeError("Frozen backbone state changed")
    report["complete"] = True
    report["implementation_passed"] = bool(report["overfit"]["passed"] and report["checkpoint_roundtrip_passed"])
    save()
    dataset.close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
