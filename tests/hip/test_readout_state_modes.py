"""Orb state encodings reach both readouts without changing initial predictions."""

import copy
from dataclasses import replace

import pytest
import torch
from test_readout_conditioning import features

from hip_orb.model import HessianHead


def state_features():
    f = features()
    return replace(f, state_embedding=torch.randn(2, 4))


@pytest.mark.parametrize("mode", ["concat", "residual"])
@pytest.mark.parametrize("context", [False, True])
def test_state_mode_preserves_initial_predictions_and_processor_random_state(mode, context):
    f = state_features()
    kwargs = {"neighbor_pooling": context, "pair_residual_layers": 1}
    torch.manual_seed(23)
    reference = HessianHead(4, 8, **kwargs)
    following = torch.rand(7)
    torch.manual_seed(23)
    head = HessianHead(4, 8, readout_state_mode=mode, **kwargs)
    torch.testing.assert_close(torch.rand(7), following, rtol=0, atol=0)
    for actual, expected in zip(head(f), reference(f), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    sum(m.square().mean() for m in head(f)).backward()
    if mode == "concat":
        for module in (head.pair_adapter, head.node_head):
            assert module[0].weight.grad[:, -4:].abs().sum() > 0
    else:
        for module in (head.pair_state_correction, head.node_state_correction):
            assert module[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize("mode", ["concat", "residual"])
@pytest.mark.parametrize("radial", ["log", "bessel"])
def test_state_mode_locality_symmetry_loop_tensor_and_checkpoint(mode, radial):
    f = state_features()
    head = HessianHead(4, 8, readout_state_mode=mode, radial_features=radial)
    with torch.no_grad():
        if mode == "concat":
            for module in (head.pair_adapter, head.node_head):
                module[0].weight[:, -4:].normal_(std=0.1)
        else:
            for module in (head.pair_state_correction, head.node_state_correction):
                module[-1].weight.normal_(std=0.1)
    original = head(f, residual_only=True)
    state = f.state_embedding.clone()
    state[0] += 2
    changed = head(replace(f, state_embedding=state), residual_only=True)
    assert not torch.allclose(original[0][:3, :3], changed[0][:3, :3])
    assert not torch.allclose(original[0][:3, 3:6], changed[0][:3, 3:6])
    torch.testing.assert_close(original[1], changed[1], atol=0, rtol=0)
    for matrix in changed:
        torch.testing.assert_close(matrix, matrix.T, atol=0, rtol=0)
    loop = copy.deepcopy(head)
    loop.batch_pairs = False
    for actual, expected in zip(loop(f), head(f), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    equal = replace(f, sizes=torch.tensor([7]), state_embedding=f.state_embedding[:1])
    torch.testing.assert_close(head(equal, as_tensor=True)[0], head(equal)[0], atol=1e-6, rtol=1e-6)
    restored = HessianHead(4, 8, readout_state_mode=mode, radial_features=radial)
    restored.load_state_dict(head.state_dict())
    assert restored.config == head.config
    for actual, expected in zip(restored(f), head(f), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Orb state embedding"):
        head(replace(f, state_embedding=None))


def test_state_modes_reject_affine_combination_and_unknown_mode():
    with pytest.raises(ValueError, match="either affine"):
        HessianHead(readout_conditioning=True, readout_state_mode="concat")
    with pytest.raises(ValueError, match="Unknown readout"):
        HessianHead(readout_state_mode="unknown")
