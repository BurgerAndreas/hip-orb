"""Bounded real-GPU check of paired-view training; run through Slurm."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hip_orb.benchmark_training import model_hash, training_step
from hip_orb.data import OrbMolDataset, collate_samples
from hip_orb.model import HIPOrb
from hip_orb.preparation import prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation-config-dir", type=Path)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = HIPOrb.pretrained("cuda", weights_path=args.weights_path)
    model.train()
    frozen_hash = model_hash(model.base)
    initial = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    dataset = OrbMolDataset(args.train_index, training_only=True)
    rows = []
    cases = [(loss, {"loss_type": loss, "consistency_weight": 0.1,
                     "translation_scale": 1., "reference_views": 2,
                     "rotation_sampling": "uniform"})
             for loss in ("mae", "block_rmse", "mass_weighted_mse")]
    if args.ablation_config_dir:
        cases = [(p.stem, json.loads(p.read_text())) for p in
                 sorted(args.ablation_config_dir.glob("train_2m_mse_ablation_*.json"))]
        assert len(cases) == 9
    try:
        available = dataset.natoms[(dataset.natoms >= 2) & (dataset.natoms <= 350)]
        for n in sorted({int(available[np.abs(available - 50).argmin()]), int(available.max())}):
            count = min(128, 4_000_000 // (3 * n)**2)
            samples = []
            for shard in np.flatnonzero(dataset.natoms == n):
                start, stop = map(int, dataset.ptr[shard:shard + 2])
                samples.extend(dataset[i] for i in range(start, min(stop, start + count - len(samples))))
                if len(samples) == count:
                    break
            batch = collate_samples(samples)
            for name, config in cases:
                loss_type = config["loss_type"]
                options = {k: config[k] for k in ("consistency_weight", "translation_scale",
                                                 "reference_views", "rotation_sampling")}
                model.head.load_state_dict(initial)
                optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for step in range(2):
                    prepared = prepare_batch(batch, seed=7 + step,
                        translation_scale=options["translation_scale"],
                        rotation_sampling=options["rotation_sampling"])
                    result = training_step(model, optimizer, prepared, loss_type=loss_type,
                        consistency_seed=1_000_010 + step, return_components=True, **options)
                    assert all(bool(torch.isfinite(v)) for v in result.values())
                    assert all(p.grad is None or bool(torch.isfinite(p.grad).all())
                               for p in model.head.parameters())
                assert any(not torch.equal(v, initial[k]) for k, v in model.head.state_dict().items())
                row = {"case": name, "loss_type": loss_type, **options,
                       "natoms": n, "batch_size": len(samples),
                       "steps": 2, "elapsed_s": time.perf_counter() - started,
                       "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                       **{k: float(v) for k, v in result.items()}}
                rows.append(row)
                print(json.dumps(row), flush=True)
        assert model_hash(model.base) == frozen_hash
        args.output.write_text(json.dumps({"passed": True, "frozen_backbone_unchanged": True,
                                          "checks": rows}, indent=2) + "\n")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
