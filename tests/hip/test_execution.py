import copy

import pytest
import torch
from torch import nn

from hip_orb.execution import configure_execution
from hip_orb.model import Features, HIPOrb


def model_and_features(radial):
    base = nn.Module()
    base.model = nn.Linear(4, 4)
    base.model.latent_dim = 4
    base.heads = nn.ModuleDict()
    model = HIPOrb(base, None, hidden=8, radial_features=radial,
                    neighbor_pooling=True, pair_residual_layers=1).double()
    f = Features(torch.randn(3, 4, dtype=torch.float64, requires_grad=True),
                 torch.randn(2, 4, dtype=torch.float64, requires_grad=True),
                 torch.tensor([0, 1]), torch.tensor([1, 0]),
                 torch.randn(3, 3, dtype=torch.float64), torch.randn(3, dtype=torch.float64),
                 torch.tensor([3]))
    return model, f


@pytest.mark.parametrize("radial", ["log", "bessel"])
def test_execution_preserves_parameters_checkpoint_outputs_and_gradients(radial):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    model, f = model_and_features(radial)
    reference = copy.deepcopy(model)
    parameter_ids = [id(p) for p in model.parameters()]
    keys = list(model.state_dict())
    config = copy.deepcopy(model.head.config)
    expected = torch.stack(reference.head(f))
    settings = configure_execution(model)
    assert settings["mode"] == "optimized"
    actual = model.head(f, as_tensor=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(actual.square().sum(), tuple(model.head.parameters()))
    gb = torch.autograd.grad(expected.square().sum(), tuple(reference.head.parameters()))
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert list(model.state_dict()) == keys
    assert [id(p) for p in model.parameters()] == parameter_ids
    assert model.head.config == config
    configure_execution(model, "eager")
    assert not hasattr(model.head, "_edge_gather")
    assert not model._tensor_graph
    torch.testing.assert_close(torch.stack(model.head(f)), expected, atol=0, rtol=0)
    configure_execution(model)
    torch.testing.assert_close(model.head(f, as_tensor=True), expected, atol=0, rtol=0)


def test_invalid_execution_mode_and_unsupported_precision_do_not_change_model():
    model, _ = model_and_features("bessel")
    with pytest.raises(ValueError, match="Execution mode"):
        configure_execution(model, "invalid")
    with pytest.raises(ValueError, match="Execution mode"):
        configure_execution(model, "optimized-bf16")
    assert not hasattr(model, "_execution_originals")


def test_bfloat16_context_uses_float32_pooling():
    from hip_orb.model import pool_directed_pairs
    first = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    i, j = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    weights = torch.rand(3, 1)
    count = weights.new_zeros((3, 1)).index_add(0, i, weights).index_add(0, j, weights)
    expected = pool_directed_pairs(first.float(), second.float(), i, j, weights, count)
    actual = pool_directed_pairs(first, second, i, j, weights, count)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
