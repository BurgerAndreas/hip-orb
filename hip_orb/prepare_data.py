"""Build private shard indexes and an atom-balanced batch plan on CPUs."""

import argparse
import json
from pathlib import Path

import numpy as np

from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.data import OrbMolDataset, build_index


def prepare_data(train_shards, validation_shards, train_index, validation_index, config, output):
    paths = [sorted(Path(root).rglob("shard_*.h5")) for root in (train_shards, validation_shards)]
    if any(not split for split in paths):
        raise ValueError("Both splits require downloaded shard_*.h5 files")
    if {p.resolve() for p in paths[0]} & {p.resolve() for p in paths[1]}:
        raise ValueError("Training and validation shard files must be separate")
    if Path(train_index).resolve() == Path(validation_index).resolve():
        raise ValueError("Training and validation need separate index paths")
    for split, index in zip(paths, (train_index, validation_index), strict=True):
        build_index(split, index)
    training = OrbMolDataset(train_index, training_only=True)
    validation = OrbMolDataset(validation_index, training_only=True)
    try:
        if set(training.sources) & set(validation.sources):
            raise ValueError("Training and validation source identities must be separate")
        for dataset in (training, validation):
            selected = np.flatnonzero((dataset.natoms >= config.get("min_atoms", 2))
                                      & (dataset.natoms <= config.get("max_atoms", 350)))
            if not len(selected):
                raise ValueError("A split has no shards in the configured atom range")
            counts = np.diff(dataset.ptr)[selected]
            dataset.paths = [dataset.paths[i] for i in selected]
            dataset.sources = [dataset.sources[i] for i in selected]
            dataset.natoms = dataset.natoms[selected]
            dataset.ptr = np.r_[0, np.cumsum(counts)]
        report = {"training_samples": len(training), "validation_samples": len(validation)}
        if config.get("batching", "equal") == "balanced":
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            sampler = BalancedBatchSampler(
                training, max_atoms=config.get("batch_max_atoms", 4096),
                batch_size=config.get("batch_size", 128),
                max_hessian_elements=config.get("max_hessian_elements", 4_000_000),
                window_shards=config.get("batch_window_shards", 8), seed=config.get("seed", 7),
                order=config.get("batch_order", "locality"), cache_path=output / "batch_plan.npz")
            report.update(batches_per_epoch=len(sampler), batch_plan_fingerprint=sampler.fingerprint)
        return report
    finally:
        training.close()
        validation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_data(args.train_shards, args.validation_shards, args.train_index,
                          args.validation_index, json.loads(args.config.read_text()), args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
