from types import SimpleNamespace

import torch
from torch import nn

from hip_orb.benchmark_training import training_step
from hip_orb.preparation import PreparedCollator, prepare_batch
from hip_orb.data import collate_samples
import numpy as np
import hashlib
import pytest


def test_complete_step_updates_head_and_reruns_frozen_features():
    calls = []

    class Adapter:
        def from_ase_atoms_list(self, molecules, device):
            calls.append(len(molecules))
            assert all(type(a.info["charge"]) is int and type(a.info["spin"]) is int for a in molecules)
            return molecules

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(6))

        def forward(self, molecules):
            return [self.weight for _ in molecules]

    base = nn.Linear(3, 3).requires_grad_(False)
    model = SimpleNamespace(base=base, adapter=Adapter(), head=Head(), features=lambda graph: graph)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    batch = {"atomic_numbers": torch.ones(2, 2, dtype=torch.int16),
             "coords": torch.tensor([[[0., 0, 0], [0, 0, 1.]]] * 2),
             "charge": torch.zeros(2, dtype=torch.int64), "spin": torch.ones(2, dtype=torch.int64),
             "hessian_upper": torch.arange(42, dtype=torch.float32).reshape(2, 21)}
    original = model.head.weight.detach().clone()
    for _ in range(2):
        loss = training_step(model, optimizer, batch, device="cpu")
        assert torch.isfinite(loss)
    assert calls == [2, 2]
    assert not torch.equal(original, model.head.weight)
    assert all(parameter.grad is None for parameter in base.parameters())


def test_preparation_preserves_geometry_and_does_not_change_global_rng():
    samples = [{"atomic_numbers": np.array([1, 1]), "coords": np.array([[0., 0, 0], [0, 0, 1.]], dtype=np.float32),
                "charge": np.array(0), "spin": np.array(1), "omol_index": np.array(i),
                "hessian_upper": np.arange(21, dtype=np.float32)} for i in range(2)]
    batch = collate_samples(samples)
    state = torch.get_rng_state().clone()
    sync = prepare_batch(batch)
    worker = PreparedCollator()(samples)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(sync["rotation"], worker["rotation"], atol=0, rtol=0)
    torch.testing.assert_close(sync["rotation"] @ sync["rotation"].T, torch.eye(3), atol=1e-6, rtol=1e-6)
    assert abs(float(torch.linalg.det(sync["rotation"])) - 1) < 1e-6
    assert np.isclose(np.linalg.norm(sync["molecules"][0].positions[1] - sync["molecules"][0].positions[0]), 1)
    torch.testing.assert_close(batch["coords"], collate_samples(samples)["coords"], atol=0, rtol=0)


def test_rotation_sampling_covers_both_hemispheres():
    batch = {"atomic_numbers": torch.ones(1, 2, dtype=torch.long),
             "coords": torch.tensor([[[0., 0., 0.], [0., 0., 1.]]]),
             "charge": torch.zeros(1, dtype=torch.long), "spin": torch.ones(1, dtype=torch.long)}
    directions = torch.stack([prepare_batch(batch, seed=i)["rotation"][:, 0]
                              for i in range(256)])
    # Uncorrected torch QR confines the first column to one hemisphere.
    assert 80 < int((directions[:, 0] > 0).sum()) < 176
    assert float(directions.mean(0).abs().max()) < 0.15
    torch.testing.assert_close(directions.square().mean(0), torch.full((3,), 1 / 3),
                               atol=0.08, rtol=0)


def test_legacy_sampler_reproduces_previous_rotation():
    batch = {"atomic_numbers": torch.ones(1, 2, dtype=torch.long),
             "coords": torch.tensor([[[0., 0., 0.], [0., 0., 1.]]]),
             "charge": torch.zeros(1, dtype=torch.long), "spin": torch.ones(1, dtype=torch.long)}
    seed = 19
    identity = hashlib.sha256(batch["coords"].numpy().tobytes() + str(seed).encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(identity[:8], "little") % (2**63 - 1))
    expected, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    expected[:, 0] *= torch.linalg.det(expected)
    actual = prepare_batch(batch, seed=seed, rotation_sampling="legacy")
    torch.testing.assert_close(actual["rotation"], expected, atol=0, rtol=0)


def test_translation_preserves_pair_distances_and_targets():
    samples = [{"atomic_numbers": np.array([1, 1]),
                "coords": np.array([[0., 0., 0.], [0., 0., 1.]], dtype=np.float32),
                "charge": np.array(0), "spin": np.array(1), "omol_index": np.array(i),
                "hessian_upper": np.arange(21, dtype=np.float32)} for i in range(2)]
    batch = collate_samples(samples)
    state = torch.get_rng_state().clone()
    baseline = prepare_batch(batch)
    shifted = prepare_batch(batch, translation_scale=1.0)
    worker = PreparedCollator(translation_scale=1.0)(samples)
    assert torch.equal(state, torch.get_rng_state())
    assert bool((shifted["translation"].abs() <= 1).all())
    assert not torch.equal(shifted["translation"][0], shifted["translation"][1])
    torch.testing.assert_close(shifted["rotation"], baseline["rotation"], atol=0, rtol=0)
    torch.testing.assert_close(shifted["translation"], worker["translation"], atol=0, rtol=0)
    torch.testing.assert_close(shifted["hessian_upper"], batch["hessian_upper"], atol=0, rtol=0)
    for i, atoms in enumerate(shifted["molecules"]):
        np.testing.assert_allclose(atoms.positions,
                                   baseline["molecules"][i].positions + shifted["translation"][i].numpy(),
                                   atol=1e-6)
        assert np.isclose(np.linalg.norm(atoms.positions[1] - atoms.positions[0]), 1)


@pytest.mark.parametrize("scale", [-1., float("nan"), float("inf")])
def test_invalid_translation_scale(scale):
    with pytest.raises(ValueError, match="Translation scale"):
        prepare_batch({}, translation_scale=scale)


@pytest.mark.parametrize("loss_type,expected_loss,expected_gradient", [
    ("mse", 4.0, 4.0 / 36), ("mae", 2.0, 1.0 / 36), ("block_rmse", 2.0, 1.0 / 36),
    ("mass_weighted_mse", 4.0 / 1.008**2, 4.0 / (36 * 1.008**2))])
def test_hessian_loss_value_and_gradient(loss_type, expected_loss, expected_gradient):
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.full((6, 6), 2.0))

        def forward(self, features):
            return [self.matrix]

    model = SimpleNamespace(head=Head(), features=lambda graph: graph,
                            adapter=SimpleNamespace(from_ase_atoms_list=lambda molecules, device: molecules))
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0)
    batch = {"rotation": torch.eye(3), "molecules": [object()],
             "atomic_numbers": torch.ones(1, 2, dtype=torch.long),
             "coords": torch.zeros(1, 2, 3), "hessian_upper": torch.zeros(1, 21)}
    loss = training_step(model, optimizer, batch, device="cpu", loss_type=loss_type)
    assert float(loss) == pytest.approx(expected_loss)
    torch.testing.assert_close(model.head.matrix.grad, torch.full((6, 6), expected_gradient))
