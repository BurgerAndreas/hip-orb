"""Frozen row chunking keeps weights, reductions, and gradients unchanged."""

import pytest
import torch
from torch import nn

from hip_orb.backbone_chunks import bounded_backbone_mlps


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_values_and_restoration(device):
    torch.manual_seed(12)
    model = nn.Module().to(device)
    model._edge_mlp = (
        nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 8), nn.LayerNorm(8))
        .to(device)
        .double()
    )
    model.requires_grad_(False).eval()
    x = torch.randn(137, 8, device=device, dtype=torch.float64)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    expected = model._edge_mlp(x)
    calls = []
    handle = model._edge_mlp[0].register_forward_hook(lambda m, args, out: calls.append(len(out)))
    with torch.no_grad(), bounded_backbone_mlps(model, 23):
        actual = model._edge_mlp(x)
    handle.remove()
    assert max(calls) <= 23
    assert "forward" not in model._edge_mlp.__dict__
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], atol=0, rtol=0)
    with pytest.raises(RuntimeError), bounded_backbone_mlps(model, 23):
        raise RuntimeError("restore on error")
    assert "forward" not in model._edge_mlp.__dict__


def test_reject_trainable_or_training_backbone():
    model = nn.Module()
    model._edge_mlp = nn.Linear(3, 3)
    with pytest.raises(ValueError), bounded_backbone_mlps(model):
        pass
    model.requires_grad_(False)
    with pytest.raises(ValueError), bounded_backbone_mlps(model):
        model._edge_mlp(torch.ones(2, 3))
