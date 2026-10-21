import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "width,rows,affine",
    [
        (3, 0, False),
        (257, 0, True),
        (3, 1, True),
        (256, 19, False),
        (257, 513, True),
        (1024, 513, False),
    ],
)
def test_fused_norm_forward_and_backward(width, rows, affine):
    from hip_orb.fused_norm import rms_norm

    torch.manual_seed(14)
    x = torch.randn(rows, width, device="cuda", requires_grad=True)
    w = torch.randn(width, device="cuda", requires_grad=True) if affine else None
    ref = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_() if affine else None
    g = torch.randn_like(x)
    y = rms_norm(x, (width,), w)
    expected = torch.nn.functional.rms_norm(ref, (width,), wr)
    torch.testing.assert_close(y, expected, atol=2e-6, rtol=3e-6)
    y.backward(g)
    expected.backward(g)
    torch.testing.assert_close(x.grad, ref.grad, atol=4e-6, rtol=1e-4)
    if affine:
        torch.testing.assert_close(w.grad, wr.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e8])
def test_fused_norm_range_and_second_derivative(scale):
    from hip_orb.fused_norm import rms_norm

    x = (torch.randn(19, 256, device="cuda") * scale).requires_grad_()
    w = torch.ones(256, device="cuda", requires_grad=True)
    y = rms_norm(x, (256,), w, 1e-5)
    ref = torch.nn.functional.rms_norm(x, (256,), w, 1e-5)
    torch.testing.assert_close(y, ref, atol=2e-6, rtol=3e-6)
    g = torch.autograd.grad(y.square().sum(), (x, w), create_graph=True)
    gr = torch.autograd.grad(ref.square().sum(), (x, w), create_graph=True)
    for a, b in zip(g, gr, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)
    second = torch.autograd.grad(sum(t.sum() for t in g), (x, w))
    expected = torch.autograd.grad(sum(t.sum() for t in gr), (x, w))
    for a, b in zip(second, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=2e-4)
