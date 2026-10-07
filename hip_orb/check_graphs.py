"""Bounded CPU/GPU graph diagnostics on real training molecules."""
import argparse
import json

import torch

from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.model import HIPOrb
from hip_orb.preparation import prepare_batch


def maximum(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = HIPOrb.pretrained("cuda", weights_path=args.checkpoint)
    model.eval()
    initial_hash = model_hash(model.base)
    dataset = OrbMolDataset(args.index, training_only=True)
    report = {"complete": False, "checks": [], "full_training_started": False}

    def save():
        from pathlib import Path
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for shard, start in enumerate(dataset.ptr[:-1]):
            for count in (1, 4):
                indices = range(int(start), min(int(dataset.ptr[shard + 1]), int(start) + count))
                batch = prepare_batch(collate_samples([dataset[i] for i in indices]), adapter=model.adapter)
                cpu = batch["prepared_graph"].to("cuda")
                raw_gpu = model.adapter.from_ase_atoms_list(batch["molecules"], device="cuda")
                record = {"natoms": int(dataset.natoms[shard]), "batch_size": len(batch["coords"])}
                # CPU preparation already sorts edges. Retain native GPU ordering for diagnosis.
                total = int(raw_gpu.batch_total_n_node)
                order = torch.argsort(raw_gpu.senders * total + raw_gpu.receivers, stable=True)
                record["native_order_matches_cpu"] = torch.equal(cpu.senders, raw_gpu.senders) and torch.equal(cpu.receivers, raw_gpu.receivers)
                torch.testing.assert_close(cpu.senders, raw_gpu.senders[order], atol=0, rtol=0)
                torch.testing.assert_close(cpu.receivers, raw_gpu.receivers[order], atol=0, rtol=0)
                record["native_vector_max_error"] = maximum(cpu.edge_features["vectors"], raw_gpu.edge_features["vectors"][order])
                with torch.no_grad():
                    native = torch.stack(model(raw_gpu.clone()))
                    raw_cpu = torch.stack(model(cpu.clone()))
                    native_repeat = torch.stack(model(raw_gpu.clone()))
                record["native_hessian_max_error"] = maximum(native, raw_cpu)
                record["native_repeat_max_error"] = maximum(native, native_repeat)
                fixed_cpu = refresh_message_vectors(cpu.clone())
                fixed_gpu = refresh_message_vectors(canonical_message_graph(raw_gpu.clone()))
                for key in fixed_cpu.node_features:
                    torch.testing.assert_close(fixed_cpu.node_features[key], fixed_gpu.node_features[key], atol=0, rtol=0)
                for key in fixed_cpu.edge_features:
                    torch.testing.assert_close(fixed_cpu.edge_features[key], fixed_gpu.edge_features[key], atol=0, rtol=0)
                pairs = [{key: value.to("cuda") for key, value in pair.items()} for pair in batch["prepared_pairs"]]
                torch.use_deterministic_algorithms(True)
                record["audit_deterministic"] = True
                for precision in ("high", "highest"):
                    torch.set_float32_matmul_precision(precision)
                    with torch.no_grad():
                        reference_features = model.features(fixed_gpu.clone())
                        cpu_features = model.features(fixed_cpu.clone(), sizes=batch["prepared_graph"].n_node, pairs=pairs)
                        expected = torch.stack(model.head(reference_features))
                        actual = torch.stack(model.head(cpu_features))
                        repeat = torch.stack(model(fixed_gpu.clone()))
                    record[f"{precision}_fixed_hessian_max_error"] = maximum(expected, actual)
                    record[f"{precision}_repeat_max_error"] = maximum(expected, repeat)
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(repeat, expected, atol=1e-4, rtol=1e-4)
                # Compare head gradients using exactly the same frozen features.
                reference_features.pairs = None
                model.head.zero_grad(set_to_none=True)
                torch.stack(model.head(reference_features)).square().mean().backward()
                gradients = {key: value.grad.detach().clone() for key, value in model.head.named_parameters()}
                reference_features.pairs = pairs
                model.head.zero_grad(set_to_none=True)
                torch.stack(model.head(reference_features)).square().mean().backward()
                for key, value in model.head.named_parameters():
                    torch.testing.assert_close(value.grad, gradients[key], atol=1e-5, rtol=1e-5)
                record["head_gradients_match"] = True
                record["passed"] = True
                report["checks"].append(record)
                save()
                print(json.dumps(record), flush=True)
                torch.set_float32_matmul_precision("high")
                torch.use_deterministic_algorithms(False)
        report["frozen_backbone_unchanged"] = model_hash(model.base) == initial_hash
        if not report["frozen_backbone_unchanged"]:
            raise RuntimeError("Frozen backbone changed")
        report["complete"] = True
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["failure"] = str(error)
        report["failed_case"] = record if "record" in locals() else None
        raise
    finally:
        save()
        dataset.close()


if __name__ == "__main__":
    main()
