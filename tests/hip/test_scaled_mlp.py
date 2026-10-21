import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "width,scale",
    [
        (257, 0.0),
        (257, 1e-18),
        (791, 1e-12),
        (791, 1.0),
        (791, 1e10),
    ],
)
def test_scaled_mlp_values_and_gradient_range(width, scale):
    from hip_orb.scaled_mlp import scaled_mlp

    torch.manual_seed(18)
    x = torch.randn(513, width, device="cuda", requires_grad=True)
    w1 = (torch.randn(128, width, device="cuda") / width**0.5).requires_grad_()
    b1 = torch.randn(128, device="cuda", requires_grad=True)
    w2 = (torch.randn(128, 128, device="cuda") / 128**0.5).requires_grad_()
    b2 = torch.randn(128, device="cuda", requires_grad=True)
    w3 = (torch.randn(256, 128, device="cuda") / 128**0.5).requires_grad_()
    b3 = torch.randn(256, device="cuda", requires_grad=True)
    params = (x, w1, b1, w2, b2, w3, b3)
    g = torch.randn(513, 256, device="cuda") * scale
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        out = scaled_mlp(*params)
        ref = torch.nn.functional.linear(
            torch.nn.functional.silu(
                torch.nn.functional.linear(
                    torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
                )
            ),
            w3,
            b3,
        )
        assert float(((out - ref).norm() / ref.norm()).detach()) < 8e-4
        a = torch.autograd.grad(out, params, g)
        b = torch.autograd.grad(ref, params, g)
        for u, v in zip(a, b, strict=True):
            assert torch.isfinite(u).all()
            if scale:
                assert float((u - v).norm() / v.norm()) < 2e-3
            else:
                assert torch.count_nonzero(u) == 0
    finally:
        torch.set_float32_matmul_precision(old)


@pytest.mark.parametrize("rows", [0, 7])
def test_scaled_mlp_higher_derivatives(rows):
    from hip_orb.scaled_mlp import scaled_mlp

    x = torch.randn(rows, 17, device="cuda", requires_grad=True)
    w1 = (torch.randn(32, 17, device="cuda") / 17**0.5).requires_grad_()
    b1 = torch.randn(32, device="cuda", requires_grad=True)
    w2 = (torch.randn(32, 32, device="cuda") / 32**0.5).requires_grad_()
    b2 = torch.randn(32, device="cuda", requires_grad=True)
    w3 = (torch.randn(16, 32, device="cuda") / 32**0.5).requires_grad_()
    b3 = torch.randn(16, device="cuda", requires_grad=True)
    p = (x, w1, b1, w2, b2, w3, b3)
    g = torch.randn(rows, 16, device="cuda") * 1e-12
    out = scaled_mlp(*p)
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
            )
        ),
        w3,
        b3,
    )
    a = torch.autograd.grad(out, p, g, create_graph=True)
    b = torch.autograd.grad(ref, p, g, create_graph=True)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-16, rtol=1e-3)
