import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hip_orb import runner
from hip_orb import profile_training_e2e as benchmark
from hip_orb.profile_training_e2e import batch_identity, planned_batches


def test_runner_update_prepares_raw_batches_and_reads_scalar_loss(monkeypatch):
    prepared = {"coords": [0, 1], "rotation": object()}
    calls = []
    monkeypatch.setattr(runner, "prepare_batch", lambda batch, seed, **kwargs: calls.append(seed) or prepared)
    monkeypatch.setattr(runner, "training_step", lambda model, optimizer, batch, device, loss_type="mse", **kwargs:
                        torch.tensor(0.25) if batch is prepared and device == "cpu" else pytest.fail())
    assert runner.train_update(None, None, {"coords": [0, 1]}, "cpu", 11) == (0.25, 2)
    assert calls == [11]
    assert runner.train_update(None, None, prepared, "cpu", 11) == (0.25, 2)
    assert calls == [11]


def test_plan_keeps_production_budget_and_never_reads_payloads():
    dataset = SimpleNamespace(paths=["a", "b", "c"], ptr=np.array([0, 16, 24, 40]),
                              natoms=np.array([2, 10, 2]))
    batches = planned_batches(dataset, batch_size=4, budget=900, seed=7, count=10)
    assert len(batches) == 10
    rows = [i for batch in batches for i in batch]
    assert len(set(rows)) == len(rows)
    for batch in batches:
        shard = np.searchsorted(dataset.ptr, batch[0], side="right") - 1
        n = dataset.natoms[shard]
        assert len(batch) * (3 * n)**2 <= 900
        assert len(batch) <= 4
        assert all(dataset.ptr[shard] <= row < dataset.ptr[shard + 1] for row in batch)
    with pytest.raises(ValueError, match="Not enough fresh"):
        planned_batches(dataset, batch_size=4, budget=900, seed=7, count=100)


def test_sequence_hash_checks_sources_and_batch_boundaries():
    def digest(sources, split=False):
        h = hashlib.sha256()
        if split:
            for source in sources:
                batch_identity({"coords": torch.zeros(1, 2, 3), "source_dataset": [source],
                                "omol_index": torch.tensor([7])}, h)
        else:
            batch_identity({"coords": torch.zeros(2, 2, 3), "source_dataset": sources,
                            "omol_index": torch.tensor([7, 7])}, h)
        return h.hexdigest()

    assert digest(["a", "a"]) != digest(["a", "b"])
    assert digest(["a", "a"]) != digest(["a", "a"], split=True)


def test_deferred_reporting_preserves_optimizer_updates_and_detaches_loss():
    import copy

    class Adapter:
        def from_ase_atoms_list(self, molecules, device):
            return molecules

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(6))

        def forward(self, molecules):
            return [self.weight for _ in molecules]

    a = SimpleNamespace(adapter=Adapter(), head=Head(), features=lambda graph: graph)
    b = SimpleNamespace(adapter=Adapter(), head=copy.deepcopy(a.head), features=lambda graph: graph)
    optimizers = [torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0) for model in (a, b)]
    batch = {"atomic_numbers": torch.ones(2, 2, dtype=torch.int16),
             "coords": torch.tensor([[[0., 0., 0.], [0., 0., 1.]]] * 2),
             "charge": torch.zeros(2, dtype=torch.int64), "spin": torch.ones(2, dtype=torch.int64),
             "hessian_upper": torch.arange(42, dtype=torch.float32).reshape(2, 21)}
    for seed in (7, 8):
        scalar, size = runner.train_update(a, optimizers[0], batch, "cpu", seed)
        tensor, other_size = runner.train_update(b, optimizers[1], batch, "cpu", seed, scalar_loss=False)
        assert isinstance(scalar, float) and isinstance(tensor, torch.Tensor)
        assert not tensor.requires_grad
        assert scalar * size == float(tensor.double() * other_size)
        torch.testing.assert_close(a.head.weight, b.head.weight, atol=0, rtol=0)
        torch.testing.assert_close(a.head.weight.grad, b.head.weight.grad, atol=0, rtol=0)


@pytest.mark.parametrize("loss_type", ["mse", "mae"])
@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_streaming_timing_includes_reads_startup_checkpoint_and_validation(monkeypatch, tmp_path, loss_type, schedule):
    clock = [0.0]
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)

    class Source:
        def __init__(self, index, **kwargs):
            clock[0] += 2
            self.paths = [str(index)]
            self.sources = [str(index)]
            self.ptr = np.array([0, 8])
            self.natoms = np.array([2])

        def close(self):
            clock[0] += 1

    class Loader:
        def __init__(self, source, settings):
            self.source, self.settings = source, settings

        def __iter__(self):
            clock[0] += 3
            plan = planned_batches(self.source, batch_size=2, budget=100,
                                   seed=self.settings["seed"], count=self.settings["max_batches"])

            def records():
                for rows in plan:
                    clock[0] += 4  # Source read and collation happen during next().
                    yield {"coords": torch.zeros(len(rows), 2, 3),
                           "source_dataset": self.source.sources * len(rows),
                           "omol_index": torch.tensor(rows)}

            return records()

    monkeypatch.setattr(benchmark, "OrbMolDataset", Source)
    monkeypatch.setattr(benchmark, "make_loader", lambda source, **settings: Loader(source, settings))

    def audit_plan(*args, **kwargs):
        clock[0] += 13  # Benchmark-only metadata checks must stay outside timing.
        return planned_batches(*args, **kwargs)

    monkeypatch.setattr(benchmark, "planned_batches", audit_plan)

    rates = []
    def update(model, optimizer, batch, device, seed, **kwargs):
        assert kwargs["loss_type"] == loss_type
        rates.append(optimizer.param_groups[0]["lr"])
        clock[0] += 5
        return 0.25, len(batch["coords"])

    monkeypatch.setattr(benchmark, "train_update", update)
    real_save = torch.save

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        clock[0] += 6

    monkeypatch.setattr(torch, "save", save)

    def validate(*args, **kwargs):
        clock[0] += 7
        return {"validation_samples": 2, "validation": {"rmse": 1.0}}

    monkeypatch.setattr(benchmark, "validate", validate)
    head = torch.nn.Linear(1, 1)
    head.config = {}
    model = SimpleNamespace(head=head, train=lambda: None)
    args = SimpleNamespace(train_index="train", validation_index="validation", min_atoms=2,
        max_atoms=350, lr=1e-3, lr_schedule=schedule, schedule_steps=10, warmup_fraction=0.5,
        loss_type=loss_type, seed=7, warmup=1, steps=2, profile_steps=0,
        batch_size=2, max_hessian_elements=100, validation_batch_size=2,
        validation_max_hessian_elements=100, validation_samples=2, report=tmp_path / "report.json")
    row = benchmark.run_case(model, head.state_dict(), "runner2", 0, args)
    assert row["training_s"] == 18  # Two source reads and two complete updates.
    assert row["startup_and_warmup_s"] == 14  # Metadata, iterator startup, and first update.
    assert row["benchmark_row_plan_checks_s_excluded"] == 13
    assert row["training_with_startup_s"] == 32
    assert row["main_loader_wait_ms_per_step"] == 4000
    assert row["checkpoint_s"] == 6
    assert row["validation_s"] == 10  # Validation metadata, metrics, and shutdown.
    assert row["bounded_run_s_excluding_profiler"] == 49
    assert row["samples"] == 4 and row["warmup_samples"] == 2
    assert row[f"sample_weighted_train_{loss_type}"] == 0.25
    assert (tmp_path / "runner2_0/head.pt").exists()
    assert rates == pytest.approx([1e-3] * 3 if schedule == "constant" else [0.0002, 0.0004, 0.0006])
