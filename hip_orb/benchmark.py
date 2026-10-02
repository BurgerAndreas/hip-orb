"""Compare complete HIP inference with batched-VJP autograd Hessians."""

import argparse
import ast
import inspect
import json
import time
from pathlib import Path

import torch

from hip_orb import ORB_COMMIT
from hip_orb.metrics import hessian_metrics
from hip_orb.model import HIPOrb
from hip_orb.train import read_samples


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
