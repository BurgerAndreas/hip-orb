import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "sizes,width,shared,node_grad",
    [
        ((1,), 4, False, False),
        ((1,), 256, True, True),
        ((3, 2, 1), 4, True, False),
        ((3, 2, 1), 256, False, True),
        ((12, 5), 256, True, True),
        ((12, 5), 4, False, False),
    ],
)
def test_pack_values_and_all_gradients(sizes, width, shared, node_grad):
    from hip_orb.batched_pairs import dense_edge_slots
    from hip_orb.dense_pair_inputs import dense_pair_inputs, native_inputs
    from hip_orb.packed_training import compact_indices

    torch.manual_seed(18)
    p = sum(n * (n - 1) // 2 for n in sizes)
    idx = compact_indices(sizes, "cuda")
    i, j = idx["i"], idx["j"]
    fwd, rev = dense_edge_slots(sizes, i, j), dense_edge_slots(sizes, j, i)
    nodes = torch.randn(sum(sizes), width, device="cuda", requires_grad=node_grad)
    edges = torch.randn(2 * p, width, device="cuda", requires_grad=True)
    charges = torch.randn(sum(sizes), device="cuda")
    unit = torch.randn(p, 3, device="cuda")
    r1 = torch.randn(p, 17, device="cuda", requires_grad=True)
    r2 = r1 if shared else torch.randn_like(r1, requires_grad=True)
    expected = native_inputs(nodes, edges, charges, i, j, fwd, rev, unit, r1, r2)
    actual = dense_pair_inputs(nodes, edges, charges, i, j, fwd, rev, unit, r1, r2, sizes)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    pad = (-actual.shape[-1]) % 16
    storage = actual.as_strided((2 * p, actual.shape[-1] + pad), (actual.shape[-1] + pad, 1))
    assert torch.count_nonzero(storage[:, actual.shape[-1] :]) == 0
    grad = torch.randn_like(expected)
    inputs = ([nodes] if node_grad else []) + [edges, r1] + ([] if shared else [r2])
    ga = torch.autograd.grad(actual, inputs, grad)
    gb = torch.autograd.grad(expected, inputs, grad)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def test_aligned_storage_and_second_derivatives():
    from hip_orb.aligned_linear import aligned_linear
    from hip_orb.batched_pairs import dense_edge_slots
    from hip_orb.dense_pair_inputs import dense_pair_inputs, native_inputs
    from hip_orb.packed_training import compact_indices

    sizes = (3, 2)
    idx = compact_indices(sizes, "cuda")
    i, j = idx["i"], idx["j"]
    p = len(i)
    d = 4
    fwd, rev = dense_edge_slots(sizes, i, j), dense_edge_slots(sizes, j, i)
    n = torch.randn(sum(sizes), d, device="cuda", requires_grad=True)
    e = torch.randn(2 * p, d, device="cuda", requires_grad=True)
    q = torch.randn(sum(sizes), device="cuda")
    u = torch.randn(p, 3, device="cuda")
    r = torch.randn(p, 17, device="cuda", requires_grad=True)
    x = dense_pair_inputs(n, e, q, i, j, fwd, rev, u, r, r, sizes)
    ref = native_inputs(n, e, q, i, j, fwd, rev, u, r, r)
    w = torch.randn(128, x.shape[-1], device="cuda", requires_grad=True)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        y = aligned_linear(x, w)
        yr = torch.nn.functional.linear(ref, w)
        torch.testing.assert_close(y, yr, atol=3e-5, rtol=3e-5)
        ga = torch.autograd.grad(y.square().sum(), (n, e, r, w), create_graph=True)
        gb = torch.autograd.grad(yr.square().sum(), (n, e, r, w), create_graph=True)
        for a, b in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
        sa = torch.autograd.grad(sum(g.sum() for g in ga), (n, e, r, w))
        sb = torch.autograd.grad(sum(g.sum() for g in gb), (n, e, r, w))
        for a, b in zip(sa, sb, strict=True):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
    finally:
        torch.set_float32_matmul_precision(old)
