from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hip_orb.model import Features, HessianHead, HIPOrb
from hip_orb.orb_extension import make_orb_processor, dense_message_topology
from hip_orb.graphs import hessian_pair_topology
from orb_models.common.models.gns import AttentionInteractionNetwork
from orb_models.common.models.nn_util import get_cutoff


def processor():
    return AttentionInteractionNetwork(4, 2, 8, conditioning=("additive", "additive"),
        distance_cutoff=True, attention_gate="sigmoid", activation="silu", mlp_norm="rms_norm")


def test_random_processor_matches_standard_constructor_and_keeps_base_unchanged():
    template = processor()
    with torch.no_grad():
        for p in template.parameters():
            p.fill_(2)
    before = deepcopy(template.state_dict())
    torch.manual_seed(97)
    actual = make_orb_processor(template, "random")
    torch.manual_seed(97)
    expected = processor()
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(template.state_dict()[key], before[key], atol=0, rtol=0)
    assert all(p.requires_grad for p in actual.parameters())


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(4))

    def forward(self, graph):
        return (graph.node_features["latent"] * 0 + self.bias,
                graph.edge_features["vectors"][:, :1].expand(-1, 4) * 0 + self.bias)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn_stacks = nn.ModuleList([processor()])
        self.conditioner = Conditioner()

    def forward(self, graph):
        a, b = self.conditioner(graph)
        nodes, edges = self.gnn_stacks[0](graph.node_features["latent"], graph.edge_features["latent"],
            graph.senders, graph.receivers, get_cutoff(graph.edge_features["vectors"].norm(dim=-1)),
            cond_nodes=a, cond_edges=b)
        return {"node_features": nodes, "edge_features": edges}


class Charges(nn.Module):
    def forward(self, nodes, graph):
        return nodes[:, 0]


def base_and_graph():
    base = nn.Module()
    base.model = Backbone()
    base.heads = nn.ModuleDict({"latent_charges": Charges()})
    base.coulomb_module = SimpleNamespace(coulomb_constant=14.3996, direct_coulomb_erf_damping_sigma=None)
    positions = torch.tensor([[0.,0,0], [2.,0,0], [25.,1,0], [0.,2,0], [3.,2,0]])
    s, r = torch.tensor([0,1,3,4]), torch.tensor([1,0,4,3])
    graph = SimpleNamespace(system_features={"pbc":torch.tensor([False, False])},
        node_features={"positions":positions, "latent":torch.randn(5,4)},
        edge_features={"vectors":positions[r]-positions[s], "latent":torch.randn(4,4)},
        senders=s, receivers=r, n_node=torch.tensor([3,2]), n_edge=torch.tensor([2,2]))
    return base, graph


@pytest.mark.parametrize("message_graph", ["backbone", "hessian"])
@pytest.mark.parametrize("rounds,combine", [(1, False), (2, False), (1, True)])
def test_random_orb_graph_gradients_conditioning_and_checkpoint(message_graph, rounds, combine):
    if combine and message_graph != "hessian":
        pytest.skip("Edge combination only applies to the dense extension")
    torch.set_num_threads(2)
    torch.manual_seed(13)
    base, graph = base_and_graph()
    kwargs=dict(hidden=8, radial_features="bessel", pair_residual_layers=2,
                neighbor_pooling=False, all_to_all=True, extra_orb_round=True,
                orb_initialization="random", orb_message_graph=message_graph,
                orb_rounds=rounds, orb_combine_backbone_edges=combine)
    model = HIPOrb(base, None, **kwargs)
    before = deepcopy(base.state_dict())
    pairs = hessian_pair_topology(graph.senders, graph.receivers, [3,2])
    captured=[]
    hook=model.head.orb_message_layer.register_forward_pre_hook(lambda m,args:captured.append(args))
    f=model.features(graph,sizes=graph.n_node,pairs=pairs)
    hook.remove()
    assert model.head.config["extra_orb_initialization"] == "random"
    if message_graph == "hessian":
        owner=torch.tensor([0,0,0,1,1])
        assert len(f.edges)==8
        assert torch.equal(owner[f.senders],owner[f.receivers])
        assert (f.senders!=f.receivers).all()
        assert len(set(zip(f.senders.tolist(),f.receivers.tolist())))==8
        assert f.pairs is None
        assert model.head.orb_distance_encoder.radial.bessel_weights.requires_grad
        torch.testing.assert_close(captured[0][4],torch.ones(8,1))
        assert graph.n_edge.tolist()==[2,2] and len(graph.senders)==4
    else:
        assert f.senders is graph.senders and f.pairs is pairs
        assert not hasattr(model.head,"orb_distance_encoder")
        torch.testing.assert_close(captured[0][4],get_cutoff(graph.edge_features["vectors"].norm(dim=-1)))
    predicted=model.head(f,residual_only=True)
    sum(h.square().mean() for h in predicted).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    assert all(p.grad is None for p in base.parameters())
    for key,value in before.items():
        torch.testing.assert_close(base.state_dict()[key],value,atol=0,rtol=0)
    restored=HIPOrb(deepcopy(base),None,**kwargs)
    restored.head.load_state_dict(model.head.state_dict())
    for a,b in zip(restored.head(restored.features(graph),residual_only=True),predicted,strict=True):
        torch.testing.assert_close(a,b)


def test_all_to_all_context_uses_far_neighbors_and_learns_independent_radial_features():
    f=Features(torch.randn(3,4),torch.empty(0,4),torch.empty(0,dtype=torch.long),
        torch.empty(0,dtype=torch.long),torch.tensor([[0.,0,0],[2.,0,0],[30.,1,0]]),
        torch.zeros(3),torch.tensor([3]))
    head=HessianHead(4,8,radial_features="bessel",neighbor_pooling=True,all_to_all=True)
    before=head(f,residual_only=True)[0]
    nodes=f.nodes.clone();nodes[2]+=2
    after=head(replace(f,nodes=nodes),residual_only=True)[0]
    assert not torch.allclose(before[:3,:3],after[:3,:3])
    assert head.radial_embedding(torch.tensor([30.])).abs().sum()>0
    before.square().mean().backward()
    assert head.radial.bessel_weights.grad is not None
    with pytest.raises(ValueError,match="cutoff"):
        HessianHead(4,8,cutoff=12,all_to_all=True)


def test_dense_topology_is_cached_and_includes_singletons_without_cross_edges():
    a,b=dense_message_topology((1,3,2),"cpu")
    assert len(a)==8 and torch.equal(a.sort().values,a)
    owner=torch.tensor([0,1,1,1,2,2])
    assert torch.equal(owner[a],owner[b])
    assert a.data_ptr()==dense_message_topology((1,3,2),"cpu")[0].data_ptr()


def test_edge_residual_matches_directed_pairs_without_touching_other_edges():
    from hip_orb.orb_extension import BackboneEdgeResidual
    merger = BackboneEdgeResidual(4).double()
    with torch.no_grad():
        merger.projection.weight.copy_(torch.eye(4, dtype=torch.float64))
    a, b = dense_message_topology((3, 2), "cpu")
    # Deliberately scramble both edge orders.
    order = torch.tensor([7, 0, 3, 1, 5, 2, 6, 4])
    a, b = a[order], b[order]
    s, r = torch.tensor([4, 0, 2]), torch.tensor([3, 1, 0])
    old = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    geometric = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    expected = geometric.clone()
    for k, (i, j) in enumerate(zip(s, r)):
        expected[(a == i) & (b == j)] += merger.norm(old[k])
    result = merger(geometric, old, s, r, a, b, 5)
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    assert old.grad is not None and torch.isfinite(old.grad).all()
    assert merger.projection.weight.grad.abs().sum() > 0
    empty = merger(geometric, old[:0], s[:0], r[:0], a, b, 5)
    torch.testing.assert_close(empty, geometric)


@pytest.mark.parametrize("mode", ["atom", "edge", "both"])
@pytest.mark.parametrize("width", [8, 16])
def test_feature_adapters_learn_elements_geometry_and_restore_checkpoint(mode, width):
    torch.manual_seed(53)
    base, graph = base_and_graph()
    graph.node_features["atomic_numbers"] = torch.tensor([6, 1, 8, 7, 1])
    kwargs = dict(hidden=width, mlp_hidden=32, radial_features="bessel", radial_bases=4,
                  pair_residual_layers=2, all_to_all=True, extra_orb_round=True,
                  orb_initialization="random", orb_message_graph="hessian",
                  orb_combine_backbone_edges=True, orb_feature_adapters=mode)
    model = HIPOrb(base, None, **kwargs)
    original = deepcopy(base.state_dict())
    feature = model.features(graph)
    predicted = model.head(feature, residual_only=True)
    sum(h.square().mean() for h in predicted).backward()
    assert model.head.config["orb_feature_adapters"] == mode
    assert model.head.config["hidden"] == width
    for name, p in model.head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(p.grad is None for p in base.parameters())
    for name, value in original.items():
        torch.testing.assert_close(value, base.state_dict()[name], atol=0, rtol=0)
    adapter = model.head.orb_feature_adapter
    if mode in ("atom", "both"):
        assert adapter.atom_embedding.weight.grad.abs().sum() > 0
    if mode in ("edge", "both"):
        assert adapter.edge_atom_embedding.weight.grad.abs().sum() > 0
        assert adapter.radial.bessel_weights.grad.abs().sum() > 0
    restored = HIPOrb(deepcopy(base), None, **kwargs)
    restored.head.load_state_dict(model.head.state_dict())
    restored.eval()
    with torch.no_grad():
        actual = restored.head(restored.features(graph), residual_only=True)
    for a, b in zip(actual, predicted):
        torch.testing.assert_close(a, b)


def test_adapter_checkpoint_matches_eager_values_gradients_and_element_inputs():
    from hip_orb.orb_extension import OrbFeatureAdapters
    torch.manual_seed(93)
    checkpointed = OrbFeatureAdapters(4, "both", mlp_hidden=16, atom_dim=3, radial_bases=4).double()
    eager = deepcopy(checkpointed).eval()
    numbers = torch.tensor([6, 1, 8])
    positions = torch.tensor([[0., 0, 0], [1., 0.5, 0], [0., 2., 0.3]], dtype=torch.float64)
    senders, receivers = dense_message_topology((3,), "cpu")
    nodes = torch.randn(3, 4, dtype=torch.float64)
    edges = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    edges2 = edges.detach().clone().requires_grad_()
    charges = torch.randn(3, dtype=torch.float64)
    outputs = checkpointed(nodes, edges, numbers, charges, positions, senders, receivers)
    expected = eager(nodes, edges2, numbers, charges, positions, senders, receivers)
    for a, b in zip(outputs, expected):
        torch.testing.assert_close(a, b)
    sum(x.square().sum() for x in outputs).backward()
    sum(x.square().sum() for x in expected).backward()
    torch.testing.assert_close(edges.grad, edges2.grad)
    for p, q in zip(checkpointed.parameters(), eager.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    with torch.no_grad():
        changed = eager(nodes, edges2, torch.tensor([7, 1, 8]), charges, positions, senders, receivers)
        assert not torch.allclose(changed[0], expected[0], atol=1e-8, rtol=0)
        assert not torch.allclose(changed[1], expected[1], atol=1e-8, rtol=0)
        # Residual initialization preserves the input scale.
        assert (outputs[0]-nodes).abs().max() < 0.01
        assert (outputs[1]-edges).abs().max() < 0.01


def test_direct_dense_edge_residual_matches_generic_values_and_gradients():
    from hip_orb.orb_extension import BackboneEdgeResidual
    torch.manual_seed(59)
    sizes = (1, 3, 2)
    a, b = dense_message_topology(sizes, 'cpu')
    selected = torch.tensor([7, 0, 5, 2])
    s, r = a[selected], b[selected]
    generic = BackboneEdgeResidual(4).double()
    direct = deepcopy(generic)
    x = torch.randn(len(a), 4, dtype=torch.float64, requires_grad=True)
    old = torch.randn(len(s), 4, dtype=torch.float64, requires_grad=True)
    xr, oldr = x.detach().clone().requires_grad_(), old.detach().clone().requires_grad_()
    expected = generic(xr, oldr, s, r, a, b, sum(sizes))
    actual = direct(x, old, s, r, a, b, sum(sizes), dense_sizes=sizes)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward();expected.square().sum().backward()
    for u,v in [(x.grad,xr.grad),(old.grad,oldr.grad),
                *[(p.grad,q.grad) for p,q in zip(direct.parameters(),generic.parameters())]]:
        torch.testing.assert_close(u,v,atol=0,rtol=0)

@pytest.mark.parametrize('sizes', [(1,), (1,1), (1,3,2), (2,5,7), (350,)])
def test_vectorized_dense_topology_has_exact_sender_major_order(sizes):
    a,b=dense_message_topology(sizes,'cpu')
    expected=[];offset=0
    for n in sizes:
        expected.extend((i+offset,j+offset) for i in range(n) for j in range(n) if i!=j)
        offset+=n
    assert list(zip(a.tolist(),b.tolist()))==expected
