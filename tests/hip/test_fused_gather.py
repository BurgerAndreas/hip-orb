"""Check missing edges and repeated valid IDs in both differentiation paths."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")


@pytest.mark.parametrize("width", [4, 63, 256, 1024])
@pytest.mark.parametrize("edge_count,pairs", [(0, 17), (3, 0), (3, 2001), (1000, 791)])
def test_masked_edge_gather_values_and_gradients(width, edge_count, pairs):
    from hip_orb.fused_gather import masked_edge_gather
    torch.manual_seed(31)
    edges = torch.randn(edge_count, width, device="cuda", requires_grad=True)
    ids = torch.randint(-1, max(0, edge_count), (pairs,), device="cuda")
    padded = torch.cat((edges, edges.new_zeros((1, width))))
    expected = padded[torch.where(ids >= 0, ids, edge_count)]
    actual = masked_edge_gather(edges, ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    a, = torch.autograd.grad(actual, edges, grad_outputs=upstream)
    b, = torch.autograd.grad(expected, edges, grad_outputs=upstream)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_masked_edge_gather_preserves_second_derivatives():
    from hip_orb.fused_gather import masked_edge_gather
    edges = torch.randn(3, 7, device="cuda", requires_grad=True)
    ids = torch.tensor([-1, 0, 0, 2], device="cuda")
    predicted = masked_edge_gather(edges, ids)
    first, = torch.autograd.grad(predicted.pow(3).sum(), edges, create_graph=True)
    second, = torch.autograd.grad(first.sum(), edges)
    expected = edges.detach() * torch.tensor([12., 0., 6.], device="cuda")[:, None]
    torch.testing.assert_close(second, expected)
