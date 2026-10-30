"""Fused molecular conditioning retains readout values and gradients."""

import pytest
import torch
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows,width,outputs,silu,bias", [
    (0, 256, 9, False, True), (17, 256, 9, False, False),
    (513, 256, 9, False, True), (513, 257, 1024, True, True),
    (17, 512, 9, False, False), (513, 512, 9, False, True),
])
@pytest.mark.parametrize("gradient_scale", [1.0, 1e-12])
def test_affine_readout_matches_values_and_all_gradients(rows, width, outputs, silu, bias, gradient_scale):
    from hip_orb.affine_readout import affine_readout

    torch.manual_seed(129)
    x = torch.randn(rows, width + 3, device="cuda")[:, :width].requires_grad_()
    state = torch.randn(5, width * 2, device="cuda", requires_grad=True)
    scale, shift = state.split(width, -1)
    owners = torch.arange(rows, device="cuda") * 5 // max(rows, 1)
    w = (torch.randn(outputs, width, device="cuda") / width**0.5).requires_grad_()
    b = torch.randn(outputs, device="cuda", requires_grad=True) if bias else None
    params = (x, state, w, b) if bias else (x, state, w)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        expected = F.linear(x * (1 + scale[owners]) + shift[owners], w, b)
        if silu:
            expected = F.silu(expected)
        actual = affine_readout(x, scale, shift, owners, w, b, silu=silu)
        torch.testing.assert_close(actual, expected, atol=4e-5, rtol=2e-4)
        g = torch.randn_like(actual) * gradient_scale
        aa = torch.autograd.grad(actual, params, g)
        bb = torch.autograd.grad(expected, params, g)
        for a, b in zip(aa, bb, strict=True):
            torch.testing.assert_close(a, b, atol=2e-4 * gradient_scale, rtol=5e-4)
    finally:
        torch.set_float32_matmul_precision(old)


def test_affine_readout_higher_derivatives_and_frozen_inputs():
    from hip_orb.affine_readout import affine_readout

    x = torch.randn(19, 32, device="cuda", requires_grad=True)
    state = torch.randn(3, 64, device="cuda", requires_grad=True)
    s, t = state.chunk(2, -1)
    owners = torch.arange(19, device="cuda") * 3 // 19
    w = torch.randn(9, 32, device="cuda", requires_grad=True)
    y = affine_readout(x, s, t, owners, w, silu=True)
    z = F.silu(F.linear(x * (1 + s[owners]) + t[owners], w))
    a = torch.autograd.grad(y.sum(), (x, state, w), create_graph=True)
    b = torch.autograd.grad(z.sum(), (x, state, w), create_graph=True)
    for av, bv in zip(a, b, strict=True):
        torch.testing.assert_close(av, bv)
    second_a = torch.autograd.grad(sum(v.square().sum() for v in a), (x, state, w))
    second_b = torch.autograd.grad(sum(v.square().sum() for v in b), (x, state, w))
    for av, bv in zip(second_a, second_b, strict=True):
        torch.testing.assert_close(av, bv)
    y = affine_readout(x.detach(), s.detach(), t.detach(), owners, w)
    assert torch.isfinite(torch.autograd.grad(y.sum(), w)[0]).all()


def test_compact_affine_readout_bounds_large_saved_activations():
    from hip_orb.affine_readout import affine_readout

    x = torch.full((33, 32), 100000.0, device="cuda", requires_grad=True)
    s = torch.zeros(2, 32, device="cuda", requires_grad=True)
    t = torch.zeros_like(s, requires_grad=True)
    owners = torch.arange(33, device="cuda") * 2 // 33
    w = torch.ones(16, 32, device="cuda", requires_grad=True)
    y = affine_readout(x, s, t, owners, w, silu=True, compact=True)
    grads = torch.autograd.grad(y.sum(), (x, s, t, w))
    assert torch.isfinite(y).all() and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("width", [8, 512])
def test_fused_head_preserves_symmetry_and_conditioner_gradients(context, width):
    import copy
    from dataclasses import fields, replace

    from test_readout_conditioning import features

    from hip_orb.model import HessianHead

    f = features()
    f = replace(f, **{field.name: getattr(f, field.name).cuda()
                       for field in fields(f) if isinstance(getattr(f, field.name), torch.Tensor)
                       and field.name != "sizes"})
    reference = HessianHead(4, width, readout_conditioning=True, neighbor_pooling=context).cuda()
    torch.nn.init.normal_(reference.readout_conditioner[-1].weight, std=0.05)
    fused = copy.deepcopy(reference)
    fused._fused_affine_readout = True
    for actual, expected in zip(fused(f), reference(f), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    fused._fused_affine_min_pairs = 0
    a, b = fused(f), reference(f)
    for actual, expected in zip(a, b, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual, actual.T, atol=0, rtol=0)
    sum(v.square().sum() for v in a).backward()
    sum(v.square().sum() for v in b).backward()
    for (name, pa), (_, pb) in zip(fused.named_parameters(), reference.named_parameters(), strict=True):
        if pa.grad is not None:
            relative = (pa.grad - pb.grad).norm() / pb.grad.norm().clamp_min(1e-12)
            assert relative < 1e-3, (name, float(relative))
