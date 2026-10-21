import pytest
import torch

from hip_orb.aligned_linear import aligned_linear


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width,rows", [(257, 0), (791, 7), (641, 513), (256, 513)])
def test_alignment_values_and_gradients(device, width, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(41)
    x = torch.randn(rows, width, device=device, requires_grad=True)
    w = torch.randn(128, width, device=device, requires_grad=True)
    b = torch.randn(128, device=device, requires_grad=True)
    grad = torch.randn(rows, 128, device=device)
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        actual = aligned_linear(x, w, b)
        expected = torch.nn.functional.linear(x, w, b)
        torch.testing.assert_close(actual, expected, atol=8e-5, rtol=2e-4)
        ga = torch.autograd.grad(actual, (x, w, b), grad, create_graph=True)
        gb = torch.autograd.grad(expected, (x, w, b), grad, create_graph=True)
        for a, c in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, c, atol=8e-5, rtol=2e-4)
        # Exercise the ordinary first derivative separately.
        ga = torch.autograd.grad(aligned_linear(x, w, b), (x, w, b), grad)
        gb = torch.autograd.grad(torch.nn.functional.linear(x, w, b), (x, w, b), grad)
        for a, c in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, c, atol=8e-5, rtol=2e-4)
    finally:
        torch.set_float32_matmul_precision(old)
