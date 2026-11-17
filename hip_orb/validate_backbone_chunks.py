"""Compare real-checkpoint features and Hessian blocks before row chunking."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms

from hip_orb.backbone_chunks import bounded_backbone_mlps
from hip_orb.benchmark_neighbor_search import digest
from hip_orb.cutoff_inference import cutoff_hessian
from hip_orb.model import HIPOrb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision", default="float32-highest", choices=["float32-high", "float32-highest"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261117)
    p = json.loads(args.provenance.read_text())
    record = next(x for x in p["input_structures"] if x["natoms"] == 10000)
    cutoff = p["arguments"]["cutoff"]
    model = HIPOrb.pretrained(
        "cuda",
        precision=args.precision,
        weights_path=args.weights,
        hidden=128,
        cutoff=cutoff,
        neighbor_pooling=True,
    )
    with np.load(record["path"]) as data:
        atoms = Atoms(numbers=data["numbers"], positions=data["positions"])
    atoms.set_cell(np.ptp(atoms.positions, axis=0) + 2 * cutoff)
    atoms.info.update(charge=0, spin=1)
    graph = model.adapter.from_ase_atoms_list([atoms], device="cuda")
    baseline = model.features(graph)
    with bounded_backbone_mlps(model.base.model, 65536):
        bounded = model.features(graph)
    row = {
        "structure_id": record["structure_id"],
        "natoms": len(atoms),
        "geometry_sha256": record["geometry_sha256"],
        "method_version": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("backbone_chunks.py").read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": digest(args.weights),
        "mode": "backbone_chunk_validation",
        "definition": "max absolute difference across every value, bounded minus unbounded",
        "row_chunk": 65536,
        "precision": args.precision,
        "cutoff_angstrom": cutoff,
    }
    for name in ["nodes", "edges", "charges"]:
        row[f"{name}_max_abs"] = float(
            (getattr(baseline, name) - getattr(bounded, name)).abs().max()
        )
    a = cutoff_hessian(
        SimpleNamespace(head=model.head, features=lambda _: baseline),
        None,
        cutoff,
        pair_chunk=65536,
        search="gpu",
    )
    b = cutoff_hessian(
        SimpleNamespace(head=model.head, features=lambda _: bounded),
        None,
        cutoff,
        pair_chunk=65536,
        search="gpu",
    )
    row["pairs"] = len(a["i"])
    row["blocks_max_abs_eV_per_angstrom2"] = float((a["blocks"] - b["blocks"]).abs().max())
    row["diagonal_max_abs_eV_per_angstrom2"] = float((a["diagonal"] - b["diagonal"]).abs().max())
    row["charges_unit"] = "elementary charge"
    row["latent_unit"] = "dimensionless"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
