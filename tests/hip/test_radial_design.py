from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from hip_orb.radial import LogGaussianBasis
from hip_orb.model import HIPOrb, HessianHead, Features
from test_orb_extension import Backbone, base_and_graph, processor
from orb_models.common.models.nn_util import get_cutoff


def test_nonperiodic_channels_remain_informative_beyond_old_cutoff_and_gradcheck():
    basis = LogGaussianBasis(16, 4.0).double()
    distances = torch.tensor([0., .7, 6., 12., 24., 36., 100., 1e6], dtype=torch.float64, requires_grad=True)
    out = basis(distances)
    assert out.shape == (8, 16) and torch.isfinite(out).all()
    torch.testing.assert_close(out[:, -2], torch.ones_like(distances))
    torch.testing.assert_close(out[:, -1], 1 / (1 + distances))
    assert (out[3:6, :-2].sum(1) > 0.1).all()
    assert not torch.allclose(out[3], out[4])
    out.square().sum().backward()
    assert torch.isfinite(distances.grad).all()
    assert basis.centers.grad.abs().sum() > 0
    assert basis.raw_widths.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(basis, (torch.tensor([.8, 12., 80.], dtype=torch.float64, requires_grad=True),))


class StackedBackbone(Backbone):
    def __init__(self):
        super().__init__()
        self.gnn_stacks = nn.ModuleList([processor() for _ in range(3)])

    def forward(self, graph):
        a, b = self.conditioner(graph)
        nodes, edges = graph.node_features['latent'], graph.edge_features['latent']
        for layer in self.gnn_stacks:
            nodes, edges = layer(nodes, edges, graph.senders, graph.receivers,
                                 get_cutoff(graph.edge_features['vectors'].norm(dim=-1)),
                                 cond_nodes=a, cond_edges=b)
        return {'node_features': nodes, 'edge_features': edges}


@pytest.mark.parametrize('variant', ['radial', 'context', 'shared', 'mixture', 'translation', 'combined'])
def test_new_design_gradients_frozen_backbone_hooks_and_checkpoint(variant):
    torch.manual_seed(109)
    base, graph = base_and_graph()
    base.model = StackedBackbone()
    context = variant in ('context', 'combined')
    shared = variant in ('shared', 'combined')
    mixture = variant in ('mixture', 'combined')
    kwargs = dict(hidden=8, mlp_hidden=16, radial_features='log_gaussian', radial_bases=6,
                  radial_log_range=3.5, all_to_all=True, pair_residual_layers=2,
                  extra_orb_round=True, orb_initialization='random', orb_message_graph='hessian',
                  neighbor_pooling=context, checkpoint_context=context, shared_geometry=shared,
                  backbone_layer_mix=3 if mixture else 0, orb_combine_backbone_edges=mixture,
                  translation_projection=variant in ('translation', 'combined'))
    model = HIPOrb(base, None, **kwargs)
    before = deepcopy(base.state_dict())
    f = model.features(graph)
    assert all(not layer._forward_hooks for layer in base.model.gnn_stacks)
    result = model.head(f, residual_only=True)
    sum(x.square().mean() for x in result).backward()
    for key, p in model.head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), key
    assert all(p.grad is None for p in base.parameters())
    for k,v in before.items():torch.testing.assert_close(v, base.state_dict()[k], atol=0, rtol=0)
    if mixture:
        assert model.head.backbone_mixture.node_logits.grad.abs().sum() > 0
        assert model.head.backbone_mixture.edge_logits.grad.abs().sum() > 0
    if variant in ('translation', 'combined'):
        for m in result:
            torch.testing.assert_close(m.reshape(len(m)//3,3,len(m)//3,3).sum(2),torch.zeros_like(m.reshape(len(m)//3,3,len(m)//3,3).sum(2)),atol=2e-6,rtol=0)
    restored = HIPOrb(deepcopy(base), None, **kwargs)
    restored.head.load_state_dict(model.head.state_dict())
    with torch.no_grad():
        expected = restored.head(restored.features(graph), residual_only=True)
    for a,b in zip(result,expected):torch.testing.assert_close(a,b)


def test_shared_encoder_is_called_once_and_receives_both_gradient_paths():
    base, graph = base_and_graph()
    model = HIPOrb(base, None, hidden=8, mlp_hidden=16, all_to_all=True,
                   radial_features='log_gaussian', radial_bases=6,
                   extra_orb_round=True, orb_initialization='random', orb_message_graph='hessian',
                   shared_geometry=True)
    seen=[]
    hook=model.head.shared_geometry_encoder.register_forward_hook(lambda m,a,o: seen.append(o))
    f=model.features(graph)
    params=list(model.head.shared_geometry_encoder.parameters())
    gradients=torch.autograd.grad(f.nodes.square().sum(),params,retain_graph=True)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0
    isolated=replace(f,nodes=f.nodes.detach(),edges=f.edges.detach())
    loss=sum(x.square().sum() for x in model.head(isolated,residual_only=True))
    gradients=torch.autograd.grad(loss,params)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0
    assert len(seen)==1
    hook.remove()


@pytest.mark.parametrize('context',[False,True])
def test_diagonal_depends_on_final_pair_features_only_with_context(context):
    torch.manual_seed(23)
    f=Features(torch.randn(3,4),torch.empty(0,4),torch.empty(0,dtype=torch.long),
               torch.empty(0,dtype=torch.long),torch.randn(3,3),torch.zeros(3),torch.tensor([3]))
    head=HessianHead(4,8,all_to_all=True,radial_features='log_gaussian',radial_bases=6,
                     neighbor_pooling=context,checkpoint_context=context)
    _,diagonal=head._batched_forward(f,residual_only=True,as_tensor='pair-blocks')
    gradient,=torch.autograd.grad(diagonal.square().sum(),head.pair_adapter[0].weight,allow_unused=True)
    assert gradient is not None and gradient.abs().sum()>0 if context else gradient is None


def test_context_checkpoint_matches_eager_backward():
    torch.manual_seed(39)
    f=Features(torch.randn(3,4,dtype=torch.float64),torch.empty(0,4,dtype=torch.float64),
               torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),
               torch.randn(3,3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),torch.tensor([3]))
    model=HessianHead(4,8,all_to_all=True,radial_features='log_gaussian',radial_bases=6,
                      neighbor_pooling=True,checkpoint_context=True).double()
    eager=deepcopy(model);eager.checkpoint_context=False
    a=model(f,residual_only=True)[0];b=eager(f,residual_only=True)[0]
    torch.testing.assert_close(a,b)
    a.square().sum().backward();b.square().sum().backward()
    for p,q in zip(model.parameters(),eager.parameters()):torch.testing.assert_close(p.grad,q.grad)


def test_all_smooth_configs_pass_runner_option_validation(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    from hip_orb import runner
    paths = sorted((Path(__file__).resolve().parents[2] / 'configs').glob('train_2m_smooth_*_balanced_lr_1e-3.json'))
    assert len(paths) == 7
    for path in paths:
        monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--train-index',
            str(tmp_path/'missing_train.npz'), '--validation-index', str(tmp_path/'missing_val.npz'),
            '--output', str(tmp_path/'output')])
        # Reaching dataset opening confirms the CLI accepts every configuration.
        with pytest.raises(FileNotFoundError):runner.main()


@pytest.mark.parametrize('dense', [False, True])
def test_project_geometry_first_matches_predictions_and_all_gradients(dense):
    torch.manual_seed(63)
    n, d = 4, 8
    senders, receivers = torch.where(~torch.eye(n, dtype=torch.bool))
    if not dense:
        senders, receivers = senders[:3], receivers[:3]
    f = Features(torch.randn(n,d,dtype=torch.float64),
                 torch.randn(len(senders),d,dtype=torch.float64),senders,receivers,
                 torch.randn(n,3,dtype=torch.float64),torch.zeros(n,dtype=torch.float64),
                 torch.tensor([n]),geometry=torch.randn(len(senders),d,dtype=torch.float64,requires_grad=True),
                 dense_ordered=dense)
    first=HessianHead(d,16,all_to_all=True,radial_features='log_gaussian',radial_bases=6,
                      shared_geometry=True).double()
    second=deepcopy(first);second._project_geometry_first=True
    a=first(f,residual_only=True)[0];b=second(f,residual_only=True)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(a.square().sum(),(*first.parameters(),f.geometry),allow_unused=True)
    gb=torch.autograd.grad(b.square().sum(),(*second.parameters(),f.geometry),allow_unused=True)
    for u,v in zip(ga,gb,strict=True):
        if u is None:assert v is None
        else:torch.testing.assert_close(u,v,atol=1e-11,rtol=1e-10)
