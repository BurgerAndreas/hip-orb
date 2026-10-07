from types import SimpleNamespace

import torch
from torch import nn

from hip_orb.benchmark_training import training_step
from hip_orb.preparation import PreparedCollator, prepare_batch
from hip_orb.data import collate_samples
import numpy as np


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
