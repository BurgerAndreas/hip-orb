"""Separate repeated graph construction, GPU inference, and head-update effects."""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.model import HIPOrb


def difference(a, b):
    delta = (a.double() - b.double()).abs()
    return {"max_abs": float(delta.max()) if delta.numel() else 0.0,
            "rms": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
            "failed_1e6_components": int((delta > 1e-6 + 1e-6 * b.double().abs()).sum())}


def envelope(values):
    rows = [difference(a, b) for a, b in itertools.combinations(values, 2)]
    return {key: max(row[key] for row in rows) for key in rows[0]}


def summary(values):
    return {key: envelope([row[key] for row in values]) for key in values[0]}


def predict(model, graph):
    graph = graph.clone()
    original = model.base.predict(graph, split=False)
    result = {key: value.detach().cpu() for key, value in original.items()
              if key in ("energy", "forces", "grad_forces")}
    with torch.no_grad():
        result["hessian"] = torch.cat([h.reshape(-1) for h in model(graph.clone())]).cpu()
    return result


def graph_audit(graphs):
    canonical = [canonical_message_graph(g.clone()) for g in graphs]
    first = graphs[0]
    return {"edges": len(first.senders),
            "native_order_changes": sum(not (torch.equal(g.senders, first.senders) and
                                               torch.equal(g.receivers, first.receivers)) for g in graphs[1:]),
            "topology_changes": sum(not (torch.equal(g.senders, canonical[0].senders) and
                                           torch.equal(g.receivers, canonical[0].receivers)) for g in canonical[1:]),
            "canonical_vector_repeat": envelope([g.edge_features["vectors"].cpu() for g in canonical])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("Require at least two repeats")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint).eval()
    initial_head = copy.deepcopy(model.head.state_dict())
    initial_hash = model_hash(model.base)
    dataset = OrbMolDataset(args.index, training_only=True)
    report = {"complete": False, "repeats": args.repeats, "cases": [],
              "training_started": False,
              "note": "Bounded diagnostic optimizer updates only. No learned checkpoint is saved."}

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for requested in (2, 50, 70, 110, 250, 350):
            shard = int(np.abs(dataset.natoms - requested).argmin())
            start, end = map(int, dataset.ptr[shard:shard + 2])
            for desired_count in (1, 4):
                model.head.load_state_dict(initial_head)
                batch = collate_samples([dataset[i] for i in range(start, min(end, start + desired_count))])
                atoms = []
                for z, x, charge, spin in zip(batch["atomic_numbers"], batch["coords"], batch["charge"], batch["spin"], strict=True):
                    item = Atoms(numbers=z.numpy(), positions=x.numpy())
                    item.info.update(charge=int(charge), spin=int(spin))
                    atoms.append(item)
                record = {"requested_natoms": requested, "natoms": len(atoms[0]),
                          "batch_size": len(atoms), "sample_ids": batch["omol_index"].tolist(), "modes": []}
                report["cases"].append(record)
                # Native CPU and GPU builders, without resetting RNG between calls.
                cpu = [model.adapter.from_ase_atoms_list(atoms, device="cpu", edge_method="knn_scipy").to("cuda")
                       for _ in range(args.repeats)]
                gpu = [model.adapter.from_ase_atoms_list(atoms, device="cuda") for _ in range(args.repeats)]
                record["cpu_graph"] = graph_audit(cpu)
                record["gpu_graph"] = graph_audit(gpu)
                record["cpu_gpu_native_order_equal"] = torch.equal(cpu[0].senders, gpu[0].senders) and torch.equal(cpu[0].receivers, gpu[0].receivers)
                a, b = [canonical_message_graph(g.clone()) for g in (cpu[0], gpu[0])]
                record["cpu_gpu_topology_equal"] = torch.equal(a.senders, b.senders) and torch.equal(a.receivers, b.receivers)
                if not record["cpu_gpu_topology_equal"]:
                    save()
                    raise RuntimeError("CPU/GPU topology differs; inspect diagnostic record")
                record["cpu_gpu_native_vector_difference"] = difference(a.edge_features["vectors"].cpu(), b.edge_features["vectors"].cpu())
                fixed = {name: [refresh_message_vectors(canonical_message_graph(g.clone())) for g in graphs]
                         for name, graphs in (("cpu", cpu), ("gpu", gpu))}
                record["cpu_gpu_refreshed_vectors_equal"] = torch.equal(fixed["cpu"][0].edge_features["vectors"], fixed["gpu"][0].edge_features["vectors"])
                for precision, deterministic in (("high", False), ("highest", False), ("highest", True), ("high", True)):
                    torch.set_float32_matmul_precision(precision)
                    torch.use_deterministic_algorithms(deterministic)
                    mode = {"precision": precision, "deterministic": deterministic, "repeat_envelopes": {}, "cross_backend": {}}
                    record["modes"].append(mode)
                    for label, graphs in (("native", {"cpu": cpu, "gpu": gpu}), ("canonical_refreshed", fixed)):
                        values = {name: [predict(model, graph) for graph in items] for name, items in graphs.items()}
                        for name, repeated in values.items():
                            mode["repeat_envelopes"][f"{name}_{label}_fresh_graph"] = summary(repeated)
                        cross = {}
                        for key in values["cpu"][0]:
                            pairs = [difference(c[key], g[key]) for c in values["cpu"] for g in values["gpu"]]
                            mean_cpu = torch.stack([v[key].double() for v in values["cpu"]]).mean(0)
                            mean_gpu = torch.stack([v[key].double() for v in values["gpu"]]).mean(0)
                            cross[key] = {"ensemble_mean_difference": difference(mean_cpu, mean_gpu),
                                          "max_pair_difference": max(p["max_abs"] for p in pairs),
                                          "cpu_within": summary(values["cpu"])[key],
                                          "gpu_within": summary(values["gpu"])[key]}
                        mode["cross_backend"][label] = cross
                    for name in ("cpu", "gpu"):
                        mode["repeat_envelopes"][f"{name}_native_same_graph"] = summary(
                            [predict(model, (cpu if name == "cpu" else gpu)[0]) for _ in range(args.repeats)])
                    # Reproduce the readiness comparison, plus a fixed-graph control.
                    model.head.load_state_dict(initial_head)
                    base_before = model_hash(model.base)
                    before = predict(model, gpu[0])
                    fixed_before = predict(model, fixed["gpu"][0])
                    model.train()
                    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0)
                    loss = training_step(model, optimizer, batch)
                    model.eval()
                    after = predict(model, model.adapter.from_ase_atoms_list(atoms, device="cuda"))
                    fixed_after = predict(model, fixed["gpu"][0])
                    mode["head_update"] = {"loss": float(loss), "base_state_unchanged": model_hash(model.base) == base_before,
                                            "base_has_gradients": any(p.grad is not None for p in model.base.parameters()),
                                            "native_fresh_graph": {k: difference(before[k], after[k]) for k in before if k != "hessian"},
                                            "canonical_same_graph": {k: difference(fixed_before[k], fixed_after[k]) for k in before if k != "hessian"}}
                    model.head.load_state_dict(initial_head)
                    save()
                print(json.dumps({"natoms": record["natoms"], "batch_size": len(atoms),
                                  "topology_equal": record["cpu_gpu_topology_equal"], "complete": True}), flush=True)
                save()
        report["base_state_unchanged"] = model_hash(model.base) == initial_hash
        report["complete"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        torch.use_deterministic_algorithms(False)
        torch.set_float32_matmul_precision("high")
        dataset.close()
        save()


if __name__ == "__main__":
    main()
