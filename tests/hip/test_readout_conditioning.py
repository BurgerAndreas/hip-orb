"""Direct molecular-state conditioning preserves batching and checkpoint behavior."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from hip_orb.data import UniformSampleBatches
from hip_orb.model import Features, HessianHead


def features():
    torch.manual_seed(9)
    return Features(
        torch.randn(7, 4),
        torch.randn(4, 4),
        torch.tensor([0, 1, 3, 4]),
        torch.tensor([1, 0, 4, 3]),
        torch.randn(7, 3),
        torch.randn(7),
        torch.tensor([3, 4]),
        total_charge=torch.tensor([0.0, 2.0]),
        spin_multiplicity=torch.tensor([1.0, 5.0]),
    )


def test_zero_initialization_preserves_predictions_and_random_state():
    f = features()
    torch.manual_seed(3)
    reference = HessianHead(4, 8)
    following = torch.rand(5)
    torch.manual_seed(3)
    head = HessianHead(4, 8, readout_conditioning=True)
    torch.testing.assert_close(torch.rand(5), following, rtol=0, atol=0)
    for expected, got in zip(reference(f), head(f), strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    sum(h.square().mean() for h in head(f)).backward()
    assert head.readout_conditioner[-1].weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in head.readout_conditioner.parameters()
    )


@pytest.mark.parametrize("field", ["total_charge", "spin_multiplicity"])
def test_state_changes_both_readouts_without_crossing_molecules(field):
    f = features()
    head = HessianHead(4, 8, readout_conditioning=True)
    torch.nn.init.normal_(head.readout_conditioner[-1].weight, std=0.05)
    old = head(f, residual_only=True)
    changed = getattr(f, field).clone()
    changed[0] += 2
    new = head(replace(f, **{field: changed}), residual_only=True)
    assert not torch.allclose(old[0][:3, :3], new[0][:3, :3])
    assert not torch.allclose(old[0][:3, 3:6], new[0][:3, 3:6])
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    torch.testing.assert_close(new[0], new[0].T, rtol=0, atol=0)


def test_loop_and_batched_paths_and_checkpoint_agree():
    f = features()
    head = HessianHead(4, 8, readout_conditioning=True)
    torch.nn.init.normal_(head.readout_conditioner[-1].weight, std=0.05)
    original = head(f)
    loop = copy.deepcopy(head)
    loop.batch_pairs = False
    for got, wanted in zip(loop(f), original, strict=True):
        torch.testing.assert_close(got, wanted, atol=1e-5, rtol=1e-5)
    restored = HessianHead(4, 8, readout_conditioning=True)
    restored.load_state_dict(head.state_dict())
    assert restored.config["readout_conditioning"]
    for got, wanted in zip(restored(f), original, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires total charge"):
        head(replace(f, total_charge=None))


def test_uniform_validation_covers_sizes_without_dropping_or_repeating_samples():
    class Dataset:
        natoms = np.array([2, 5, 11, 24, 50])
        ptr = np.arange(6) * 200

        def __len__(self):
            return 1000

    d = Dataset()
    options = {"batch_size": 32, "max_hessian_elements": 100000, "seed": 7}
    sampler = UniformSampleBatches(d, 400, **options)
    batches = list(sampler)
    ids = [i for batch in batches for i in batch]
    assert len(ids) == len(set(ids)) == 400
    assert all(40 < sum(200 * s <= i < 200 * (s + 1) for i in ids) < 120 for s in range(5))
    for batch in batches:
        n = d.natoms[np.searchsorted(d.ptr, batch, side="right") - 1]
        assert np.all(n == n[0]) and len(batch) <= 32 and len(batch) * (3 * n[0]) ** 2 <= 100000
    sampler.set_epoch(3)
    assert list(sampler) == batches
    assert list(UniformSampleBatches(d, 400, **options)) == batches
    all_ids = [i for b in UniformSampleBatches(d, 2000, **options) for i in b]
    assert sorted(all_ids) == list(range(1000))
