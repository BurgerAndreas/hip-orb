import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "width,rows,bias",
    [
        (257, 0, False),
        (791, 0, True),
        (257, 7, True),
        (791, 513, False),
        (1024, 513, True),
        (1024, 7, False),
    ],
)
def test_fused_linear_silu_values_and_gradients(width, rows, bias):
    from hip_orb.fused_linear_silu import linear_silu

    torch.manual_seed(79)
    x = torch.randn(rows, width, device="cuda", requires_grad=True)
    w = (torch.randn(128, width, device="cuda") / width**0.5).requires_grad_()
    b = torch.randn(128, device="cuda", requires_grad=True) if bias else None
    g = torch.randn(rows, 128, device="cuda")
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = linear_silu(x, w, b)
        expected = torch.nn.functional.silu(torch.nn.functional.linear(x, w, b))
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-4)
        inputs = (x, w, b) if bias else (x, w)
        a = torch.autograd.grad(actual, inputs, g)
        r = torch.autograd.grad(expected, inputs, g)
        for u, v in zip(a, r, strict=True):
            torch.testing.assert_close(u, v, atol=4e-5, rtol=3e-4)
        a = torch.autograd.grad(linear_silu(x, w, b), inputs, g, create_graph=True)
        r = torch.autograd.grad(
            torch.nn.functional.silu(torch.nn.functional.linear(x, w, b)),
            inputs,
            g,
            create_graph=True,
        )
        for u, v in zip(a, r, strict=True):
            torch.testing.assert_close(u, v, atol=4e-5, rtol=3e-4)
    finally:
        torch.set_float32_matmul_precision(old)


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_compact_saved_activation_preserves_small_gradients_and_higher_derivatives(scale):
    from hip_orb.fused_linear_silu import linear_silu

    torch.manual_seed(33)
    x = torch.randn(513, 257, device="cuda", requires_grad=True)
    w = (torch.randn(128, 257, device="cuda") / 257**0.5).requires_grad_()
    b = torch.randn(128, device="cuda", requires_grad=True)
    g = torch.randn(513, 128, device="cuda") * scale
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        y = linear_silu(x, w, b, compact=True)
        ref = torch.nn.functional.silu(torch.nn.functional.linear(x, w, b))
        torch.testing.assert_close(y, ref, atol=3e-5, rtol=1e-4)
        ga = torch.autograd.grad(y, (x, w, b), g)
        gb = torch.autograd.grad(ref, (x, w, b), g)
        for a, bv in zip(ga, gb, strict=True):
            assert float((a - bv).norm() / bv.norm()) < 4e-4
        ga = torch.autograd.grad(
            linear_silu(x, w, b, compact=True), (x, w, b), g, create_graph=True
        )
        gb = torch.autograd.grad(
            torch.nn.functional.silu(torch.nn.functional.linear(x, w, b)),
            (x, w, b),
            g,
            create_graph=True,
        )
        for a, bv in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, bv, atol=4e-5 * scale, rtol=3e-4)
    finally:
        torch.set_float32_matmul_precision(old)
