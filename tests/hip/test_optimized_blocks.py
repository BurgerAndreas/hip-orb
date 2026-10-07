import copy

import pytest
import torch

from hip_orb.block_layout import layout
from hip_orb.graphs import hessian_pair_topology
from hip_orb.model import Features, HessianHead


@pytest.mark.parametrize("sizes", [[1], [3, 3], [1, 2, 5]])
@pytest.mark.parametrize("sigma", [None, 0.7])
@pytest.mark.parametrize("cutoff", [None, 1.2, 1e-6])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("residual_only", [False, True])
def test_batched_coulomb_outputs_and_gradients(sizes, sigma, cutoff, precomputed, residual_only):
    torch.set_num_threads(2)
    torch.manual_seed(91)
    senders, receivers, offset = [], [], 0
    for n in sizes:
        if n > 1:
            senders.extend((offset, offset+1))
            receivers.extend((offset+1, offset))
        offset += n
    features = Features(torch.randn(offset, 4, dtype=torch.float64, requires_grad=True),
                        torch.randn(len(senders), 4, dtype=torch.float64, requires_grad=True),
                        torch.tensor(senders, dtype=torch.long), torch.tensor(receivers, dtype=torch.long),
                        torch.randn(offset, 3, dtype=torch.float64, requires_grad=True),
                        torch.randn(offset, dtype=torch.float64, requires_grad=True), torch.tensor(sizes), sigma=sigma)
    if precomputed:
        features.pairs = hessian_pair_topology(features.senders, features.receivers, sizes)
    other = copy.deepcopy(features)
    reference = HessianHead(4, 8, cutoff=cutoff).double()
    optimized = copy.deepcopy(reference)
    reference.batch_pairs = False
    expected = reference(features, residual_only=residual_only)
    actual = optimized(other, residual_only=residual_only)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    sum(h.square().mean() for h in expected).backward()
    sum(h.square().mean() for h in actual).backward()
    for a, b in zip(optimized.parameters(), reference.parameters()):
        if a.grad is None or b.grad is None:
            existing = a.grad if a.grad is not None else b.grad
            assert existing is None or not bool(existing.any())
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)
    for name in ("nodes", "edges", "positions", "charges"):
        a, b = getattr(other, name).grad, getattr(features, name).grad
        if a is None or b is None:
            existing = a if a is not None else b
            assert existing is None or not bool(existing.any())
        else:
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
    assert optimized.state_dict().keys() == reference.state_dict().keys()


def test_layout_cache_reuses_indices_and_is_bounded():
    layout.cache_clear()
    assert layout((2, 3), "cpu") is layout((2, 3), "cpu")
    for n in range(1, 12):
        layout((n,), "cpu")
    assert layout.cache_info().currsize == 8


def test_invalid_geometry_still_fails():
    features = Features(torch.zeros(2, 4), torch.zeros(0, 4),
                        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                        torch.zeros(2, 3), torch.zeros(2), torch.tensor([2]))
    head = HessianHead(4, 8)
    with pytest.raises(ValueError, match="nonzero distance"):
        head(features)
