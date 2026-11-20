import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("strided_input", ["nodes", "edges", "senders", "receivers"])
def test_dense_interaction_falls_back_for_strided_inputs(strided_input):
    from hip_orb.dense_interaction import dense_interaction

    class EagerLayer:
        _attention_gate = "sigmoid"
        _node_cond = _edge_cond = None

        def __call__(self, nodes, edges, senders, receivers, cutoff, *, cond_nodes, cond_edges):
            return nodes, edges

    inputs = {
        "nodes": torch.randn(3, 4, device="cuda"),
        "edges": torch.randn(6, 4, device="cuda"),
        "senders": torch.tensor([0, 0, 1, 1, 2, 2], device="cuda"),
        "receivers": torch.tensor([1, 2, 0, 2, 0, 1], device="cuda"),
    }
    inputs[strided_input] = inputs[strided_input].repeat_interleave(2, dim=0)[::2]
    assert not inputs[strided_input].is_contiguous()
    cutoff = torch.ones((6, 1), device="cuda")
    nodes, edges = dense_interaction(EagerLayer(), *inputs.values(), cutoff, sizes=(3,))
    assert nodes is inputs["nodes"] and edges is inputs["edges"]


@pytest.mark.parametrize("sizes", [(1,), (3, 2, 1), (37, 8)])
@pytest.mark.parametrize("d", [4, 256])
def test_dense_message_pack_and_pool_gradients(sizes, d):
    from hip_orb.dense_interaction import packed_messages, pooled_messages
    from hip_orb.orb_extension import dense_message_topology

    s, r = dense_message_topology(sizes, "cuda")
    m = len(s)
    n = torch.randn(sum(sizes), d, device="cuda", requires_grad=True)
    e = torch.randn(m, d, device="cuda", requires_grad=True)
    sa = torch.randn(m, 1, device="cuda", requires_grad=True)
    ra = torch.randn_like(sa, requires_grad=True)
    x = packed_messages(n, e, s, r, sizes)
    ref = torch.cat((e, n[s], n[r]), -1)
    torch.testing.assert_close(x, ref, atol=0, rtol=0)
    g = torch.randn_like(x)
    a = torch.autograd.grad(x, (n, e), g)
    b = torch.autograd.grad(ref, (n, e), g)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, atol=3e-5, rtol=3e-5)
    actual = pooled_messages(e, sa, ra, s, r, sizes)
    reference = (
        e.new_zeros(n.shape).index_add(0, s, e * sa),
        e.new_zeros(n.shape).index_add(0, r, e * ra),
    )
    for u, v in zip(actual, reference):
        torch.testing.assert_close(u, v, atol=3e-5, rtol=3e-5)
    gs, gr = (torch.randn_like(t) for t in actual)
    a = torch.autograd.grad(actual, (e, sa, ra), (gs, gr))
    b = torch.autograd.grad(reference, (e, sa, ra), (gs, gr))
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, atol=3e-5, rtol=3e-5)


def test_dense_message_higher_derivatives():
    from hip_orb.dense_interaction import packed_messages, pooled_messages
    from hip_orb.orb_extension import dense_message_topology

    sizes = (3, 2)
    s, r = dense_message_topology(sizes, "cuda")
    d = 4
    n = torch.randn(5, d, device="cuda", requires_grad=True)
    e = torch.randn(len(s), d, device="cuda", requires_grad=True)
    sa = torch.randn(len(s), 1, device="cuda", requires_grad=True)
    ra = torch.randn_like(sa, requires_grad=True)
    x = packed_messages(n, e, s, r, sizes)
    ref = torch.cat((e, n[s], n[r]), -1)
    ga = torch.autograd.grad(x.square().sum(), (n, e), create_graph=True)
    gb = torch.autograd.grad(ref.square().sum(), (n, e), create_graph=True)
    for u, v in zip(ga, gb):
        torch.testing.assert_close(u, v)
    a = pooled_messages(e, sa, ra, s, r, sizes)
    b = (e.new_zeros(n.shape).index_add(0, s, e * sa), e.new_zeros(n.shape).index_add(0, r, e * ra))
    ga = torch.autograd.grad(sum(t.square().sum() for t in a), (e, sa, ra), create_graph=True)
    gb = torch.autograd.grad(sum(t.square().sum() for t in b), (e, sa, ra), create_graph=True)
    aa = torch.autograd.grad(sum(t.sum() for t in ga), (e, sa, ra))
    bb = torch.autograd.grad(sum(t.sum() for t in gb), (e, sa, ra))
    for u, v in zip(aa, bb):
        torch.testing.assert_close(u, v)
