from types import SimpleNamespace

import numpy as np
import pytest

from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.data import make_loader


class MetadataOnly:
    def __init__(self, sizes=(3, 10, 20, 35), counts=(37, 23, 31, 17)):
        self.natoms = np.asarray(sizes)
        self.ptr = np.r_[0, np.cumsum(counts)]
        self.paths = [f"shard_{i}" for i in range(len(sizes))]

    def __len__(self):
        return int(self.ptr[-1])

    def __getitem__(self, index):
        raise AssertionError("Planning must not load geometries")


def sizes_for(data, ids):
    return data.natoms[np.searchsorted(data.ptr, ids, side="right") - 1]


def test_coverage_budgets_balance_and_metadata_only():
    data = MetadataOnly()
    sampler = BalancedBatchSampler(data, max_atoms=120, batch_size=16, max_hessian_elements=30_000)
    batches = list(sampler)
    assert len(sampler) == len(batches)
    assert sorted(i for b in batches for i in b) == list(range(len(data)))
    totals = []
    for b in batches:
        ns = sizes_for(data, b)
        assert len(b) <= 16
        assert ns.sum() <= 120
        assert (9 * ns**2).sum() <= 30_000
        totals.append(ns.sum())
    # Most bins fill the atom budget despite the tenfold size range.
    assert np.median(totals) >= 110
    assert any(len(np.unique(sizes_for(data, b))) > 1 for b in batches)
    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches
    assert sorted(map(sorted, sampler)) == sorted(map(sorted, batches))


def test_persistent_plan_epoch_and_resume(tmp_path):
    data = MetadataOnly()
    path = tmp_path / "plan.npz"
    a = BalancedBatchSampler(data, max_atoms=120, cache_path=path)
    b = BalancedBatchSampler(data, max_atoms=120, cache_path=path)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)
    # Existing loader wrappers preserve deterministic batch-boundary resume.
    loader = make_loader(data, batch_sampler=b, collate_fn=lambda x: x, start_batch=2)
    loader.batch_sampler.set_epoch(3)
    assert list(loader.batch_sampler) == list(a)[2:]
    with pytest.raises(ValueError, match="differs"):
        BalancedBatchSampler(data, max_atoms=121, cache_path=path)
    altered = MetadataOnly()
    altered.paths[0] = "different_source"
    with pytest.raises(ValueError, match="differs"):
        BalancedBatchSampler(altered, max_atoms=120, cache_path=path)
    with np.load(path) as saved:
        contents = dict(saved)
    contents["indices"][0] = contents["indices"][1]
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="coverage"):
        BalancedBatchSampler(data, max_atoms=120, cache_path=path)


def test_packed_loader(tmp_path):
    from hip_orb.packed_training import collate_packed_compact

    class Toy(MetadataOnly):
        def __getitem__(self, index):
            n = int(sizes_for(self, [index])[0])
            return {
                "coords": np.zeros((n, 3), dtype=np.float32),
                "atomic_numbers": np.ones(n, dtype=np.int64),
                "hessian_upper": np.zeros(3 * n * (3 * n + 1) // 2, dtype=np.float32),
                "charge": np.float32(0),
                "spin": np.float32(0),
                "omol_index": np.int64(index),
            }

    data = Toy()
    sampler = BalancedBatchSampler(data, max_atoms=120)
    loader = make_loader(
        data, batch_sampler=sampler, collate_fn=collate_packed_compact, pin_memory=False
    )
    batch = next(iter(loader))
    assert batch["coords"].shape[0] == int(batch["sizes"].sum()) <= 120
    with pytest.raises(ValueError, match="collate_fn"):
        make_loader(data, batch_sampler=sampler)
    with pytest.raises(ValueError, match="one process"):
        make_loader(data, batch_sampler=sampler, world_size=2)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_atoms": 0}, {"batch_size": 0}, {"max_hessian_elements": 8}, {"window_shards": 0}],
)
def test_invalid_budgets(kwargs):
    settings = {"max_atoms": 120}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        BalancedBatchSampler(MetadataOnly(), **settings)


def test_invalid_metadata_and_oversized_sample():
    with pytest.raises(ValueError, match="metadata"):
        BalancedBatchSampler(SimpleNamespace(), max_atoms=120)
    with pytest.raises(ValueError, match="exceeds"):
        BalancedBatchSampler(MetadataOnly(), max_atoms=34)
    data = MetadataOnly()
    data.ptr[-1] = data.ptr[-2]
    with pytest.raises(ValueError, match="metadata"):
        BalancedBatchSampler(data, max_atoms=120)


def test_no_shuffle_and_cache_budget_validation(tmp_path):
    data = MetadataOnly()
    path = tmp_path / "plan.npz"
    a = BalancedBatchSampler(data, max_atoms=120, shuffle=False, cache_path=path)
    batches = list(a)
    a.set_epoch(9)
    assert list(a) == batches
    with np.load(path) as saved:
        contents = dict(saved)
    contents["batch_ptr"] = np.array([0, len(data)])
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="budget"):
        BalancedBatchSampler(data, max_atoms=120, cache_path=path)


@pytest.mark.parametrize("hessian_budget", [4_000_000, None])
def test_runner_uses_packed_updates_and_resumes(monkeypatch, tmp_path, hessian_budget):
    import json
    import sys

    import torch

    from hip_orb import runner

    for name in ("train", "val"):
        np.savez(
            tmp_path / f"{name}.npz",
            paths=np.array([str(tmp_path / f"{name}.h5")]),
            natoms=np.array([3]),
            ptr=np.array([0, 4]),
            sources=np.array([name]),
        )

    class Model:
        def __init__(self):
            self.base = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 1)
            self.head.config = {}

        def train(self):
            pass

    monkeypatch.setattr(runner.HIPOrb, "pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(runner, "configure_execution", lambda *a: {})
    settings_seen = []

    class Loader:
        def __init__(self, settings):
            self.batch_sampler = settings.get(
                "batch_sampler", SimpleNamespace(set_epoch=lambda e: None)
            )

        def __iter__(self):
            return iter([{"sizes": torch.tensor([3, 3])}] * 4)

    def loader(dataset, **settings):
        settings_seen.append(settings)
        return Loader(settings)

    monkeypatch.setattr(runner, "make_loader", loader)
    calls = []

    def update(model, optimizer, batch, device, seed, loss_type, *, scaler=None, loss_scale=1.0):
        assert loss_scale == 1.0
        calls.append(loss_type)
        return 1.0, len(batch["sizes"])

    monkeypatch.setattr(runner, "packed_update", update)
    monkeypatch.setattr(
        runner, "train_update", lambda *a, **k: pytest.fail("Requires packed update")
    )
    monkeypatch.setattr(runner, "validate", lambda *a: {"validation": {}, "validation_samples": 10})
    out = tmp_path / "out"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"max_hessian_elements": hessian_budget, "loss_scale": 1.0}))
    argv = [
        "train",
        "--config",
        str(config_path),
        "--train-index",
        str(tmp_path / "train.npz"),
        "--validation-index",
        str(tmp_path / "val.npz"),
        "--output",
        str(out),
        "--device",
        "cpu",
        "--batching",
        "balanced",
        "--loss-type",
        "block_rmse",
        "--max-steps",
        "2",
        "--session-steps",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    saved = torch.load(out / "head.pt", weights_only=True)
    assert saved["global_step"] == 1
    assert saved["loss_scale"] == 1.0
    assert saved["batching_config"]["mode"] == "balanced"
    assert isinstance(settings_seen[1]["batch_sampler"], BalancedBatchSampler)
    assert "batch_sampler" not in settings_seen[0]  # Validation stays equal-size.
    monkeypatch.setattr(sys, "argv", argv + ["--checkpoint", str(out / "head.pt")])
    runner.main()
    assert calls == ["block_rmse", "block_rmse"]
    assert json.loads((out / "report.json").read_text())["training_complete"]
    assert settings_seen[3]["start_batch"] == 1
    monkeypatch.setattr(
        sys,
        "argv",
        argv
        + [
            "--checkpoint",
            str(out / "head.pt"),
            "--batching",
            "equal",
            "--max-hessian-elements",
            "4000000",
        ],
    )
    with pytest.raises(ValueError, match="batch plan"):
        runner.main()


def test_uncapped_hessians_and_locality_groups(tmp_path):
    data = MetadataOnly(sizes=(350, 70, 150, 20), counts=(30, 30, 30, 30))
    sampler = BalancedBatchSampler(
        data, max_atoms=4096, batch_size=64, window_shards=2, cache_path=tmp_path / "uncapped.npz"
    )
    batches = list(sampler)
    assert any((9 * sizes_for(data, b) ** 2).sum() > 4_000_000 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(len(data)))
    # All batches from a locality window stay together within each epoch.
    membership = {}
    for g, (a, b) in enumerate(zip(sampler.group_ptr[:-1], sampler.group_ptr[1:], strict=True)):
        for i in range(a, b):
            membership[
                frozenset(sampler.indices[sampler.batch_ptr[i] : sampler.batch_ptr[i + 1]])
            ] = g
    for epoch in (0, 1, 9):
        sampler.set_epoch(epoch)
        groups = [membership[frozenset(b)] for b in sampler]
        runs = [g for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g]
        assert len(runs) == len(set(runs))
    restored = BalancedBatchSampler(
        data, max_atoms=4096, batch_size=64, window_shards=2, cache_path=tmp_path / "uncapped.npz"
    )
    restored.set_epoch(9)
    assert list(restored) == list(sampler)
