"""Bounded overfit and pilot training. Full collection training stays disabled."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms

from hip_orb import ORB_COMMIT, ORB_VERSION
from hip_orb.coulomb import fixed_charge_hessian
from hip_orb.metrics import hessian_metrics
from hip_orb.model import HIPOrb


def read_samples(directory, count, minimum, maximum, assignments=None):
    """Read a bounded pilot; skip incomplete shards and retain scientific IDs."""
    samples = []
    paths = sorted(directory.glob("shard_*.h5"))
    cap = count
    if assignments is not None:
        with np.load(assignments) as packed:
            sizes = packed["shard_natoms"]
        selected = np.flatnonzero((sizes >= minimum) & (sizes <= maximum))
        groups = [selected[sizes[selected] == n] for n in np.unique(sizes[selected])]
        if not groups:
            raise ValueError("No shards in the requested atom range")
        cap = int(np.ceil(count / len(groups)))
        ids = [
            int(group[row])
            for row in range(max(map(len, groups)))
            for group in groups
            if row < len(group)
        ]
        paths = [directory / f"shard_{index:05d}.h5" for index in ids]
    for path in paths:
        if ".partial." in path.name:
            continue
        with h5py.File(path) as f:
            if not bool(f.attrs.get("complete", False)):
                continue
            if str(f.attrs.get("model", "")) != "orbmol_v2":
                raise ValueError("Pilot requires OrbMol-v2 teacher labels")
            accepted = 0
            for row, n in enumerate(f["natoms"][:]):
                if not minimum <= n <= maximum:
                    continue
                lo, hi = f["atom_ptr"][row : row + 2]
                hlo, hhi = f["hessian_ptr"][row : row + 2]
                atoms = Atoms(numbers=f["atomic_numbers"][lo:hi], positions=f["coords"][lo:hi])
                atoms.info.update(charge=int(f["charge"][row]), spin=int(f["spin"][row]))
                samples.append(
                    (
                        atoms,
                        f["hessian_flat"][hlo:hhi].reshape(3 * n, 3 * n),
                        int(f["omol_index"][row]),
                    )
                )
                if len(samples) == count:
                    return samples
                accepted += 1
                if accepted >= cap:
                    break
    raise ValueError(f"Only {len(samples)} eligible pilot records; need {count}")


def random_rotation(device, dtype):
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device, dtype=dtype))
    # Randomize QR signs; then enforce det=+1.
    signs = torch.where(torch.rand(3, device=device) < 0.5, -1.0, 1.0).to(dtype)
    q = q * signs
    q[:, 0] *= torch.linalg.det(q)
    return q


def transformed_sample(sample, rotation):
    atoms, h, index = sample
    atoms = atoms.copy()
    r = rotation.detach().cpu().numpy()
    atoms.positions = atoms.positions @ r.T
    blocks = h.reshape(len(atoms), 3, len(atoms), 3)
    h = np.einsum("ab,ibjd,cd->iajc", r, blocks, r).reshape(h.shape)
    return atoms, h, index


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("overfit", "pilot"), default="pilot")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-atoms", type=int, default=50)
    parser.add_argument("--max-atoms", type=int, default=70)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--hessian-cutoff", type=float)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.train_count, args.validation_count, args.epochs) < 1:
        raise ValueError("Counts and epochs must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    model = HIPOrb.pretrained(args.device, hidden=args.hidden, cutoff=args.hessian_cutoff)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if checkpoint["head_config"] != model.head.config or checkpoint["orb_commit"] != ORB_COMMIT:
            raise ValueError("Checkpoint architecture or Orb base differs")
        model.head.load_state_dict(checkpoint["head"])
    samples = read_samples(
        args.shards,
        args.train_count + args.validation_count,
        args.min_atoms,
        args.max_atoms,
        args.assignments,
    )
    order = np.random.default_rng(args.seed).permutation(len(samples))
    training = [samples[int(i)] for i in order[: args.train_count]]
    validation = (
        training
        if args.stage == "overfit"
        else [samples[int(i)] for i in order[args.train_count :]]
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0)
    original_hash = state_hash(model.base)
    args.output.mkdir(parents=True, exist_ok=True)
    cached = {}

    def features(sample):
        atoms, target, _ = sample
        graph = model.adapter.from_ase_atoms_list([atoms], device=args.device)
        return model.features(graph), torch.as_tensor(
            target, device=args.device, dtype=next(model.head.parameters()).dtype
        )

    if args.stage == "overfit":
        cached = {sample[2]: features(sample) for sample in training}
    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for i in np.random.permutation(len(training)):
            sample = training[int(i)]
            if args.stage == "overfit":
                feature, target = cached[sample[2]]
            else:
                rotated = transformed_sample(sample, random_rotation(args.device, torch.float32))
                feature, target = features(rotated)
            baseline = fixed_charge_hessian(
                feature.positions, feature.charges, feature.constant, feature.sigma
            )
            residual = model.head(feature, residual_only=True)[0]
            # Subtract only fixed-charge curvature, not the full Coulomb Hessian.
            loss = (residual - (target - baseline)).square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite pilot loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        history.append(loss_sum / len(training))
        print(json.dumps({"epoch": epoch + 1, "train_mse": history[-1]}), flush=True)
    model.eval()
    metrics = []
    with torch.no_grad():
        for sample in validation:
            feature, target = features(sample)
            predicted = model.head(feature)[0]
            masses = torch.as_tensor(sample[0].get_masses(), device=args.device, dtype=target.dtype)
            row = hessian_metrics(predicted, target, feature.positions, masses)
            rotation = random_rotation(args.device, target.dtype)
            rotated_feature, _ = features(transformed_sample(sample, rotation))
            rotated_prediction = model.head(rotated_feature)[0]
            transform = torch.kron(
                torch.eye(len(sample[0]), device=args.device), rotation.contiguous()
            )
            row["rotation_rmse"] = float(
                (rotated_prediction - transform @ predicted @ transform.T).square().mean().sqrt()
            )
            row["omol_index"] = sample[2]
            metrics.append(row)
    unchanged = original_hash == state_hash(model.base)
    if not unchanged:
        raise RuntimeError("Frozen Orb state changed")
    checkpoint = {
        "head": model.head.state_dict(),
        "head_config": model.head.config,
        "orb_commit": ORB_COMMIT,
        "orb_version": ORB_VERSION,
        "precision": "float32-high",
        "teacher": "orbmol_v2",
    }
    torch.save(checkpoint, args.output / "head.pt")
    average = {
        key: float(np.mean([row[key] for row in metrics]))
        for key in metrics[0]
        if key != "omol_index"
    }
    overfit_passed = args.stage == "overfit" and average["relative_rmse"] < 0.05
    report = {
        "stage": args.stage,
        "orb_commit": ORB_COMMIT,
        "frozen_state_unchanged": unchanged,
        "train_indices": [x[2] for x in training],
        "validation_indices": [x[2] for x in validation],
        "train_mse": history,
        "validation": average,
        "per_sample": metrics,
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else None,
        "molecule_disjoint_split_verified": False,
        "prfo_passed": False,
        "overfit_passed": overfit_passed,
        "full_training_allowed": False,
        "gate_notes": "Provisional geometry split. Require molecular identity and real TS tests before full training.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation": average, "full_training_allowed": False}), flush=True)
    if args.stage == "overfit" and not overfit_passed:
        raise SystemExit("Overfit gate failed; do not start the pilot")


if __name__ == "__main__":
    main()
