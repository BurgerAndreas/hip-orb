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

@pytest.mark.parametrize('sizes', [[1], [1, 2, 5], [3, 3, 3], [2, 7]])
def test_dense_ordered_slots_match_generic_lookup_and_gradients(sizes):
    from hip_orb.orb_extension import dense_message_topology
    from hip_orb.batched_pairs import batch_topology
    senders, receivers = dense_message_topology(tuple(sizes), 'cpu')
    total = sum(sizes)
    torch.manual_seed(31)
    feature = Features(torch.randn(total, 4, dtype=torch.float64),
                       torch.randn(len(senders), 4, dtype=torch.float64),
                       senders, receivers, torch.randn(total, 3, dtype=torch.float64),
                       torch.randn(total, dtype=torch.float64), torch.tensor(sizes))
    generic = batch_topology(feature)
    feature.dense_ordered = True
    direct = batch_topology(feature)
    for a, b in zip(generic[1:], direct[1:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    direct_head = HessianHead(4, 8).double()
    generic_head = copy.deepcopy(direct_head)
    reference = copy.deepcopy(feature)
    reference.dense_ordered = False
    feature.edges.requires_grad_(); reference.edges.requires_grad_()
    actual, expected = direct_head(feature), generic_head(reference)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum(h.square().sum() for h in actual).backward()
    sum(h.square().sum() for h in expected).backward()
    if feature.edges.grad is not None:
        torch.testing.assert_close(feature.edges.grad, reference.edges.grad, atol=0, rtol=0)
    for a, b in zip(direct_head.parameters(), generic_head.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

@pytest.mark.parametrize('radial', ['bessel','log_gaussian'])
@pytest.mark.parametrize('context', [False,True])
@pytest.mark.parametrize('sizes', [(1,), (3,2,1)])
def test_joined_directions_preserve_values_and_all_gradients(radial,context,sizes):
    from hip_orb.orb_extension import dense_message_topology
    torch.manual_seed(89)
    s,r=dense_message_topology(sizes,'cpu');total=sum(sizes)
    f=Features(torch.randn(total,4,dtype=torch.float64,requires_grad=True),
               torch.randn(len(s),4,dtype=torch.float64,requires_grad=True),s,r,
               torch.randn(total,3,dtype=torch.float64),torch.randn(total,dtype=torch.float64),torch.tensor(sizes))
    fr=copy.deepcopy(f)
    joined=HessianHead(4,8,mlp_hidden=16,all_to_all=True,radial_features=radial,
                       pair_residual_layers=2,neighbor_pooling=context).double()
    native=copy.deepcopy(joined);joined._join_radial_pairs=True
    a,b=joined(f),native(fr)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    sum(x.square().sum() for x in a).backward();sum(x.square().sum() for x in b).backward()
    for x,y in zip(joined.parameters(),native.parameters()):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-10)
    for key in ['nodes','edges']:
        x,y=getattr(f,key).grad,getattr(fr,key).grad
        if x is not None:torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
