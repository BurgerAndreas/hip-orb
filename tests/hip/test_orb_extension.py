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
def test_random_orb_graph_gradients_conditioning_and_checkpoint(message_graph):
    torch.set_num_threads(2)
    torch.manual_seed(13)
    base, graph = base_and_graph()
    kwargs=dict(hidden=8, radial_features="bessel", pair_residual_layers=2,
                neighbor_pooling=False, all_to_all=True, extra_orb_round=True,
                orb_initialization="random", orb_message_graph=message_graph)
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
