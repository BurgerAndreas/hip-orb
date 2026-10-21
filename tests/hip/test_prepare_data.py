import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.data import OrbMolDataset
from hip_orb.prepare_data import prepare_data


def shards(root, sizes, source):
    root.mkdir(parents=True)
    for i, n in enumerate(sizes):
        count = 3
        width = 3 * n * (3 * n + 1) // 2
        with h5py.File(root / f"shard_{i:05d}.h5", "w") as f:
            f.attrs.update(complete=True, model="orbmol_v2", source_dataset=source)
            f.create_dataset("natoms", data=np.full(count, n))
            f.create_dataset("hessian_ptr", data=np.arange(count + 1) * width)
            f.create_dataset("hessian_upper", shape=(count * width,), dtype="f4")


def test_preparation_matches_runner_filtered_plan_and_reuses_cache(tmp_path, monkeypatch):
    train, val = tmp_path / "train", tmp_path / "validation"
    shards(train, (2, 3, 351), "train")
    shards(val, (2,), "validation")
    config = json.loads(Path("configs/train.json").read_text())
    monkeypatch.setattr(OrbMolDataset, "__getitem__", lambda *_: pytest.fail("Planning must use metadata only"))
    args = (train, val, tmp_path / "train.npz", tmp_path / "val.npz", config, tmp_path / "run")
    first = prepare_data(*args)
    assert first["training_samples"] == 6 and first["validation_samples"] == 3
    dataset = OrbMolDataset(args[2])
    dataset.paths, dataset.sources = dataset.paths[:2], dataset.sources[:2]
    dataset.natoms, dataset.ptr = dataset.natoms[:2], dataset.ptr[:3]
    sampler = BalancedBatchSampler(dataset, max_atoms=config["batch_max_atoms"],
                                  batch_size=config["batch_size"], max_hessian_elements=None,
                                  window_shards=config["batch_window_shards"], seed=config["seed"],
                                  order=config["batch_order"], cache_path=args[5] / "batch_plan.npz")
    assert first["batch_plan_fingerprint"] == sampler.fingerprint
    assert sorted(i for batch in sampler for i in batch) == list(range(6))
    before = (args[5] / "batch_plan.npz").read_bytes()
    assert prepare_data(*args) == first
    assert (args[5] / "batch_plan.npz").read_bytes() == before
    dataset.close()


@pytest.mark.parametrize("shared_files", [False, True])
def test_preparation_rejects_overlapping_splits(tmp_path, shared_files):
    train, val = tmp_path / "train", tmp_path / "val"
    shards(train, (2,), "same_source")
    if shared_files:
        val = train
    else:
        shards(val, (3,), "same_source")
    with pytest.raises(ValueError, match="separate"):
        prepare_data(train, val, tmp_path / "train.npz", tmp_path / "val.npz",
                     {"batching": "balanced"}, tmp_path / "run")
