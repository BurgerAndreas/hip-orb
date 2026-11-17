"""Measure full cutoff inference and sparse-head training on fixed structures.

The timed boundary starts with CPU ASE atoms already in memory and includes
message graph construction, transfers, the frozen Orb backbone, radius search,
and the two-pass Hessian head. Training also includes zero_grad, block MSE,
backward, and AdamW. File reads, checkpoint loading, and warmup are excluded.
Training targets are zero blocks: this measures execution, not model accuracy.
"""

import argparse
import copy
import gc
import hashlib
import json
import platform
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from hip_orb.backbone_chunks import bounded_backbone_mlps
from hip_orb.cutoff_inference import cutoff_forward
from hip_orb.model import HIPOrb
from hip_orb.neighbor_search import AsyncCPUPairs


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        while data := handle.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def prepare_inputs(path, output, counts):
    """Save the exact float32 coordinates used by every backend."""
    atoms = read(path)
    # ASE resolves PDB coordinates. Keep the selected atom identities verbatim.
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for n in counts:
        copies = (n + len(atoms) - 1) // len(atoms)
        span = np.ptp(atoms.positions, axis=0) + 80
        positions = np.concatenate([atoms.positions + [k * span[0], 0, 0] for k in range(copies)])[
            :n
        ].astype(np.float32)
        numbers = np.tile(atoms.numbers, copies)[:n]
        identity = hashlib.sha256(numbers.tobytes() + positions.tobytes()).hexdigest()
        item = output / f"{path.stem}-{n}-{identity[:12]}.npz"
        np.savez_compressed(item, positions=positions, numbers=numbers)
        records.append(
            {
                "structure_id": f"{path.stem}:{n}:{identity[:16]}",
                "path": str(item.resolve()),
                "natoms": n,
                "geometry_sha256": identity,
                "source": str(path.resolve()),
                "source_sha256": digest(path),
                "source_url": "https://files.rcsb.org/download/1AON.pdb"
                if path.stem.upper() == "1AON"
                else None,
                "geometry_kind": "prefix" if copies == 1 else "translated-copy scaling proxy",
                "position_unit": "angstrom",
            }
        )
    return records


def run_step(model, atoms, backend, cutoff, pair_chunk, optimizer=None, backbone_chunk=0):
    work = None
    try:
        if backend == "cpu_async":
            work = AsyncCPUPairs(
                torch.tensor(atoms.positions, dtype=torch.float32), [len(atoms)], cutoff
            )
        graph = model.adapter.from_ase_atoms_list([atoms], device="cuda")
        graph._hip_known_molecular = True
        model.base.eval()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        context = (
            bounded_backbone_mlps(model.base.model, backbone_chunk)
            if backbone_chunk
            else nullcontext()
        )
        with context, torch.set_grad_enabled(optimizer is not None):
            output = cutoff_forward(
                model,
                graph,
                cutoff,
                pair_chunk=pair_chunk,
                search="cpu" if backend == "cpu_async" else backend,
                pair_future=work,
                checkpoint_chunks=optimizer is not None,
            )
            loss = output["diagonal"].square().mean()
            if len(output["blocks"]):
                loss = loss + output["blocks"].square().mean()
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        return output, loss.detach()
    finally:
        if work is not None:
            work.close()


def timed_step(model, atoms, backend, args, optimizer=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output, loss = run_step(
        model, atoms, backend, args.cutoff, args.pair_chunk, optimizer, args.backbone_chunk
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    row = {
        "seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "pairs": len(output["i"]),
        "loss": float(loss),
        "blocks_sum": float(output["blocks"].detach().double().sum()),
        "diagonal_sum": float(output["diagonal"].detach().double().sum()),
    }
    return row


def append(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[100, 350, 1204, 10000, 30000, 100000]
    )
    parser.add_argument("--backends", nargs="+", default=["brute", "cpu_async", "gpu"])
    parser.add_argument("--cutoff", type=float, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--pair-chunk", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-max-atoms", type=int, default=1204)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backbone-chunk", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "per_structure.jsonl"
    if metrics.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {metrics}")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(20261117)
    model = HIPOrb.pretrained(
        "cuda",
        weights_path=args.weights,
        hidden=args.hidden,
        cutoff=args.cutoff,
        neighbor_pooling=True,
    )
    initial = copy.deepcopy(model.head.state_dict())
    records = prepare_inputs(args.input, args.output / "inputs", args.counts)
    source_dir = Path(__file__).resolve().parent
    sources = {
        p.name: digest(p)
        for p in [
            source_dir / "neighbor_search.py",
            source_dir / "neighbor_kernels.py",
            source_dir / "cutoff_inference.py",
            source_dir / "backbone_chunks.py",
            Path(__file__),
        ]
    }
    version = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    provenance = {
        "method_version": version,
        "source_sha256": sources,
        "backbone_checkpoint_sha256": digest(args.weights),
        "head_config": model.head.config,
        "head_seed": 20261117,
        "head_weights": "deterministic random initialization; no trained head checkpoint",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "timing_boundary": __doc__,
        "units": {
            "seconds": "s",
            "memory": "bytes",
            "blocks": "eV/angstrom^2",
            "loss": "(eV/angstrom^2)^2",
        },
        "training_loss": "mean(blocks**2) + mean(diagonal**2); zero target; one forward/backward/AdamW",
        "training_scope": "cutoff sparse-head path; existing dense runner is unchanged",
        "input_structures": records,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    for record in records:
        with np.load(record["path"]) as arrays:
            atoms = Atoms(numbers=arrays["numbers"], positions=arrays["positions"])
        # The backbone's large-system cell search rejects zero-volume cells,
        # including nonperiodic molecules. This box does not enable PBC or
        # move atoms; every Hessian backend receives the same coordinates.
        atoms.set_cell(np.ptp(atoms.positions, axis=0) + 2 * args.cutoff)
        atoms.info.update(charge=0, spin=1)
        for mode in ["inference"] + (["training"] if len(atoms) <= args.train_max_atoms else []):
            # Rotate backend order across structures to reduce order bias.
            rotation = args.counts.index(len(atoms)) % len(args.backends)
            backends = args.backends[rotation:] + args.backends[:rotation]
            for backend in backends:
                model.head.load_state_dict(initial)
                model.head.train(mode == "training")
                optimizer = (
                    torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0)
                    if mode == "training"
                    else None
                )
                base = {
                    "structure_id": record["structure_id"],
                    "geometry_sha256": record["geometry_sha256"],
                    "geometry_kind": record["geometry_kind"],
                    "natoms": len(atoms),
                    "mode": mode,
                    "backend": backend,
                    "method_version": version,
                    "cutoff_angstrom": args.cutoff,
                }
                try:
                    for _ in range(args.warmup):
                        timed_step(model, atoms, backend, args, optimizer)
                    for repeat in range(args.repeats):
                        row = dict(
                            base,
                            repeat=repeat,
                            status="ok",
                            **timed_step(model, atoms, backend, args, optimizer),
                        )
                        append(metrics, row)
                        print(
                            json.dumps(
                                {
                                    k: row[k]
                                    for k in [
                                        "natoms",
                                        "mode",
                                        "backend",
                                        "repeat",
                                        "seconds",
                                        "pairs",
                                    ]
                                }
                            ),
                            flush=True,
                        )
                except (torch.OutOfMemoryError, RuntimeError) as error:
                    append(metrics, dict(base, status="error", error=str(error)))
                    print(
                        json.dumps(dict(base, status="error", error=str(error)[:300])), flush=True
                    )
                finally:
                    del optimizer
                    model.head.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        stdout=(args.output / "gpu.txt").open("w"),
        check=True,
        timeout=15,
    )


if __name__ == "__main__":
    main()
