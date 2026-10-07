import copy

import pytest
import torch

from hip_orb.graphs import hessian_pair_topology
from hip_orb.model import Features, HessianHead


@pytest.mark.parametrize("sizes", [[1], [1, 1], [3, 3, 3], [1, 2, 5]])
@pytest.mark.parametrize("cutoff", [None, 1.2, 1e-6])
@pytest.mark.parametrize("precomputed", [False, True])
def test_batched_readouts_preserve_outputs_and_all_gradients(sizes, cutoff, precomputed):
    torch.set_num_threads(2)
    torch.manual_seed(6)
    senders, receivers, offset = [], [], 0
    for n in sizes:
        if n > 1:
            senders += [offset, offset + 1]
            receivers += [offset + 1, offset]
        offset += n
    features = Features(torch.randn(offset, 4, dtype=torch.float64),
                        torch.randn(len(senders), 4, dtype=torch.float64),
                        torch.tensor(senders, dtype=torch.long), torch.tensor(receivers, dtype=torch.long),
                        torch.randn(offset, 3, dtype=torch.float64), torch.randn(offset, dtype=torch.float64),
                        torch.tensor(sizes))
    if precomputed:
        features.pairs = hessian_pair_topology(features.senders, features.receivers, sizes)
    for name in ("nodes", "edges", "positions", "charges"):
        getattr(features, name).requires_grad_(True)
    reference_features = copy.deepcopy(features)
    batched = HessianHead(4, 8, cutoff=cutoff).double()
    reference = copy.deepcopy(batched)
    reference.batch_pairs = False
    calls = []
    handle = batched.pair_adapter.register_forward_hook(lambda *args: calls.append(1))
    expected, actual = reference(reference_features), batched(features)
    handle.remove()
    assert len(calls) == 1
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(a, a.T, atol=0, rtol=0)
    sum(h.square().sum() for h in actual).backward()
    sum(h.square().sum() for h in expected).backward()
    for a, b in zip(batched.parameters(), reference.parameters()):
        if a.grad is None or b.grad is None:
            # Empty/cutoff reference paths may leave edge parameters unused.
            existing = a.grad if a.grad is not None else b.grad
            assert existing is None or not bool(existing.any())
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)
    assert batched.state_dict().keys() == reference.state_dict().keys()
    for name in ("nodes", "edges", "positions", "charges"):
        a, b = getattr(features, name).grad, getattr(reference_features, name).grad
        if a is None or b is None:
            existing = a if a is not None else b
            assert existing is None or not bool(existing.any())
        else:
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_measured_head_defaults():
    head = HessianHead(4, 8)
    assert head.batch_pairs
    assert head.fuse_pairs
    assert head.cache_pairs
    assert not hasattr(head, "factor_projection")
    assert not hasattr(head, "optimized_blocks")
    assert not hasattr(head, "packed_training")
