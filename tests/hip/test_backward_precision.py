import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "rows,scale,weight_only",
    [
        (0, 1.0, False),
        (0, 1.0, True),
        (513, 0.0, False),
        (513, 1e-18, False),
        (513, 1e-18, True),
        (513, 1.0, False),
        (513, 1.0, True),
        (513, 1e10, False),
        (513, 1e10, True),
    ],
)
def test_backward_mlp_preserves_forward_and_gradient_range(rows, scale, weight_only):
    from hip_orb.backward_precision import _BackwardMLP

    torch.manual_seed(42)
    layers = torch.nn.Sequential(
        torch.nn.Linear(257, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 256),
    ).cuda()
    x = torch.randn(rows, 257, device="cuda", requires_grad=True)
    params = (x, *layers.parameters())
    operations = tuple(layer.forward for layer in layers)
    out = _BackwardMLP.apply(*params, operations, True, weight_only)
    ref = layers(x)
    assert torch.equal(out, ref)
    g = torch.randn_like(out) * scale
    actual = torch.autograd.grad(out, params, g)
    expected = torch.autograd.grad(ref, params, g)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        if scale and rows:
            assert float((a.double() - b.double()).norm() / b.double().norm()) < 0.003
        else:
            assert torch.count_nonzero(a) == 0


def test_backward_mlp_higher_derivatives():
    from hip_orb.backward_precision import _BackwardMLP

    layers = torch.nn.Sequential(
        torch.nn.Linear(17, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 16),
    ).cuda()
    x = torch.randn(7, 17, device="cuda", requires_grad=True)
    params = (x, *layers.parameters())
    out = _BackwardMLP.apply(*params, tuple(layer.forward for layer in layers))
    ref = layers(x)
    g = torch.randn_like(out) * 1e-12
    a = torch.autograd.grad(out, params, g, create_graph=True)
    b = torch.autograd.grad(ref, params, g, create_graph=True)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-16, rtol=1e-4)
    aa = torch.autograd.grad(sum(t.square().sum() for t in a), x)[0]
    bb = torch.autograd.grad(sum(t.square().sum() for t in b), x)[0]
    torch.testing.assert_close(aa, bb, atol=1e-28, rtol=1e-3)


@pytest.mark.parametrize(
    "scale,need_bias",
    [
        (0.0, False),
        (1e-18, True),
        (1.0, False),
        (1e10, True),
    ],
)
def test_gradient_quantization(scale, need_bias):
    from hip_orb.gradient_quantization import quantize_gradient

    torch.manual_seed(21)
    g = torch.randn(513, 257, device="cuda") * scale
    q, maximum, bias = quantize_gradient(g, need_bias=need_bias)
    restored = q.float() * torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(1e-30) / 1024.0)))
    assert torch.isfinite(restored).all()
    if scale:
        assert float((restored.double() - g.double()).norm() / g.double().norm()) < 0.0003
    else:
        assert torch.count_nonzero(restored) == 0
    if need_bias:
        torch.testing.assert_close(bias, g.sum(0), rtol=1e-4, atol=abs(scale) * 1e-4)
