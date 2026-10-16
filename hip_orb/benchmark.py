"""Compare complete HIP inference with batched-VJP autograd Hessians."""

import argparse
import ast
import inspect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms

from hip_orb import ORB_COMMIT
from hip_orb.data import unpack_hessian
from hip_orb.metrics import hessian_metrics
from hip_orb.model import HIPOrb


def read_samples(directory, count, minimum, maximum, assignments=None):
    """Read a bounded benchmark; skip incomplete shards and retain scientific IDs."""
    samples = []
    paths = sorted(directory.glob("shard_*.h5")) or sorted(directory.glob("*/shard_*.h5"))
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
        paths = [directory / f"shard_{index:05d}.h5"
                 if (directory / f"shard_{index:05d}.h5").exists()
                 else directory / f"{index // 1000:03d}" / f"shard_{index:05d}.h5"
                 for index in ids]
    for path in paths:
        if ".partial." in path.name:
            continue
        with h5py.File(path) as f:
            if not bool(f.attrs.get("complete", False)):
                continue
            if str(f.attrs.get("model", "")) != "orbmol_v2":
                raise ValueError("Benchmark requires OrbMol-v2 teacher labels")
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
                        unpack_hessian(torch.from_numpy(f["hessian_upper"][hlo:hhi]), int(n)).numpy()
                        if "hessian_upper" in f else f["hessian_flat"][hlo:hhi].reshape(3 * n, 3 * n),
                        int(f["omol_index"][row]),
                    )
                )
                if len(samples) == count:
                    return samples
                accepted += 1
                if accepted >= cap:
                    break
    raise ValueError(f"Only {len(samples)} eligible benchmark records; need {count}")


def enable_batched_harmonics():
    """Use the original Python angular body, as in optimized collection."""
    from orb_models.common.models import angular

    source = Path(inspect.getfile(angular))
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_spherical_harmonics"
    )
    function.decorator_list = []
    exec(  # noqa: S102 - compile the pinned local upstream function, not external input.
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(source),
            "exec",
        ),
        vars(angular),
    )


def autograd_hessian(base, adapter, atoms, block=8):
    base.train(True)
    graph = adapter.from_ase_atoms_list([atoms], device="cuda")
    positions = graph.node_features["positions"]
    positions.requires_grad_(True)
    try:
        force = base.predict(graph, split=False)["forces"].reshape(-1)
        rows = []
        for start in range(0, len(force), block):
            stop = min(start + block, len(force))
            seed = force.new_zeros((stop - start, len(force)))
            seed[:, start:stop] = torch.eye(stop - start, device=force.device)
            derivative = torch.autograd.grad(
                force,
                positions,
                grad_outputs=seed,
                is_grads_batched=True,
                retain_graph=stop < len(force),
            )[0]
            rows.append(derivative.reshape(stop - start, -1))
        h = -torch.cat(rows).T
        return ((h + h.T) * 0.5).detach()
    finally:
        base.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 70, 110, 200, 350])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    if checkpoint["orb_commit"] != ORB_COMMIT:
        raise ValueError("Checkpoint uses a different teacher")
    config = checkpoint["head_config"]
    model = HIPOrb.pretrained(hidden=config["hidden"], cutoff=config["cutoff"])
    model.head.load_state_dict(checkpoint["head"])
    model.eval()
    enable_batched_harmonics()
    rows = []

    def measure(function):
        function()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.repeats):
            result = function()
        torch.cuda.synchronize()
        return (
            result,
            (time.perf_counter() - started) / args.repeats,
            torch.cuda.max_memory_allocated(),
        )

    for size in args.sizes:
        atoms, target, index = read_samples(args.shards, 1, size, size, args.assignments)[0]

        def hip(atoms=atoms):
            with torch.no_grad():
                return model(model.adapter.from_ase_atoms_list([atoms], device="cuda"))[0]

        predicted, seconds, memory = measure(hip)
        teacher, reference_seconds, reference_memory = measure(
            lambda atoms=atoms: autograd_hessian(model.base, model.adapter, atoms)
        )
        target = torch.as_tensor(target, device="cuda")
        positions = torch.as_tensor(atoms.positions, device="cuda", dtype=predicted.dtype)
        masses = torch.as_tensor(atoms.get_masses(), device="cuda", dtype=predicted.dtype)
        rows.append(
            dict(
                natoms=size,
                batch_size=1,
                omol_index=index,
                hip_seconds=seconds,
                autograd_seconds=reference_seconds,
                speedup=reference_seconds / seconds,
                hip_peak_bytes=memory,
                autograd_peak_bytes=reference_memory,
                recomputed_teacher_rmse=float((teacher - target).square().mean().sqrt()),
                **hessian_metrics(predicted, target, positions, masses),
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"measurements": rows, "full_training_allowed": False}, indent=2) + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
