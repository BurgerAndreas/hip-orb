import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "width,scale,compact",
    [
        (257, 0.0, False),
        (257, 1e-18, False),
        (257, 1e-18, True),
        (1024, 1.0, True),
        (257, 1e10, True),
        (1024, 1e-12, False),
        (1024, 0.0, True),
        (1024, 1e10, False),
    ],
)
def test_scaled_silu_preserves_gradient_range(width, scale, compact):
    from hip_orb.scaled_silu import scaled_linear_silu

    torch.manual_seed(41)
    x = torch.randn(513, width, device="cuda", requires_grad=True)
    w = (torch.randn(128, width, device="cuda") / width**0.5).requires_grad_()
    b = torch.randn(128, device="cuda", requires_grad=True)
    g = torch.randn(513, 128, device="cuda") * scale
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = scaled_linear_silu(x, w, b, compact=compact)
        expected = torch.nn.functional.silu(torch.nn.functional.linear(x, w, b))
        assert float(((actual - expected).norm() / expected.norm()).detach()) < 6e-4
        a = torch.autograd.grad(actual, (x, w, b), g)
        r = torch.autograd.grad(expected, (x, w, b), g)
        for u, v in zip(a, r, strict=True):
            assert torch.isfinite(u).all()
            if scale:
                assert float((u - v).norm() / v.norm()) < 1e-3
            else:
                assert torch.count_nonzero(u) == 0
    finally:
        torch.set_float32_matmul_precision(old)


@pytest.mark.parametrize("rows", [0, 7])
def test_scaled_silu_shapes_bias_and_higher_derivatives(rows):
    from hip_orb.scaled_silu import scaled_linear_silu

    x = torch.randn(rows, 257, device="cuda", requires_grad=True)
    w = (torch.randn(128, 257, device="cuda") / 257**0.5).requires_grad_()
    g = torch.randn(rows, 128, device="cuda") * 1e-12
    a = torch.autograd.grad(scaled_linear_silu(x, w, compact=True), (x, w), g, create_graph=True)
    r = torch.autograd.grad(
        torch.nn.functional.silu(torch.nn.functional.linear(x, w)), (x, w), g, create_graph=True
    )
    for u, v in zip(a, r, strict=True):
        torch.testing.assert_close(u, v, atol=1e-16, rtol=1e-3)


def test_scaled_silu_large_input_keeps_finite_values_and_gradients():
    from hip_orb.scaled_silu import scaled_linear_silu

    torch.manual_seed(43)
    x = (torch.randn(513, 257, device="cuda") * 2**18).requires_grad_()
    w = (torch.randn(128, 257, device="cuda") / 257**0.5).requires_grad_()
    b = torch.randn(128, device="cuda", requires_grad=True)
    actual = scaled_linear_silu(x, w, b, compact=True)
    reference = torch.nn.functional.silu(torch.nn.functional.linear(x, w, b))
    assert torch.isfinite(actual).all()
    assert float((actual-reference).norm()/reference.norm()) < 1e-3
    g = torch.randn_like(actual) * 1e-8
    for a, r in zip(torch.autograd.grad(actual, (x,w,b), g),
                    torch.autograd.grad(reference, (x,w,b), g), strict=True):
        assert torch.isfinite(a).all()
        # This range test includes cancellation near the SiLU transition.
        assert float((a-r).norm()/r.norm().clamp_min(1e-30)) < 3e-2
        assert torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0) > 0.999
