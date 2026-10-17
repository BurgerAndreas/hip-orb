"""Compare speed candidates with the independent eager execution path."""

from types import SimpleNamespace

import pytest
import torch

from hip_orb.execution import known_charge_forward
from hip_orb.model import Features, HessianHead


def features(sizes):
    total = sum(sizes)
    sources, targets = [], []
    offset = 0
    for n in sizes:
        if n > 1:
            sources += [offset, offset + 1]
            targets += [offset + 1, offset]
        offset += n
    return Features(torch.randn(total, 4, dtype=torch.float64, requires_grad=True),
                    torch.randn(len(sources), 4, dtype=torch.float64, requires_grad=True),
                    torch.tensor(sources, dtype=torch.long), torch.tensor(targets, dtype=torch.long),
                    torch.randn(total, 3, dtype=torch.float64, requires_grad=True),
                    torch.randn(total, dtype=torch.float64, requires_grad=True), torch.tensor(sizes))


@pytest.mark.parametrize("projection", [False, True])
def test_tensor_api_preserves_projection_and_gradients(projection):
    f = features((3, 3))
    head = HessianHead(4, 8, radial_features="bessel", neighbor_pooling=True,
                       translation_projection=projection).double()
    expected = torch.stack(head(f))
    actual = head.forward_tensor(f)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    parameters = tuple(head.parameters())
    ga = torch.autograd.grad(actual.square().sum(), parameters)
    gb = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_tensor_api_rejects_mixed_sizes():
    with pytest.raises(ValueError, match="equal atom counts"):
        HessianHead(4, 8).double().forward_tensor(features((2, 3)))


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("total_charge", [False, True])
def test_known_charge_owners_preserve_orb_values_and_gradients(enforce, total_charge):
    from orb_models.forcefield.models.forcefield_heads import LatentChargeHead
    torch.manual_seed(29)
    head = LatentChargeHead(4, num_mlp_layers=2, mlp_hidden_dim=8,
                            enforce_total_charge=enforce).double()
    data = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    sizes = torch.tensor([1, 2, 4])
    graph = SimpleNamespace(n_node=sizes,
                            node_batch_index=torch.arange(3).repeat_interleave(sizes),
                            system_features={"total_charge": torch.tensor([1., -1., 2.])}
                            if total_charge else {})
    expected = head(data, graph)
    actual = known_charge_forward(head, data, graph)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    params = (data, *head.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_radial_backward_shortcut_requires_frozen_feature_inputs():
    f = features((3, 5))
    head = HessianHead(4, 8, radial_features="bessel", all_to_all=True).double()
    head._radial_only_backward = True
    head(f)
    assert head.pair_adapter[0]._hip_input_grad_columns is None
    for name in ("nodes", "edges", "positions", "charges"):
        setattr(f, name, getattr(f, name).detach())
    head(f)
    assert head.pair_adapter[0]._hip_input_grad_columns == (15, 32)
    f.positions.requires_grad_(True)
    head(f)
    assert head.pair_adapter[0]._hip_input_grad_columns is None
