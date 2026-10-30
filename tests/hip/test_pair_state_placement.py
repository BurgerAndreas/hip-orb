"""Earlier state injection and nonlinear pair readouts retain training contracts."""

import copy
from dataclasses import replace

import pytest
import torch
from test_readout_conditioning import features

from hip_orb.model import HessianHead


@pytest.mark.parametrize("placement", ["adapter", "residual"])
def test_zero_state_projection_preserves_shared_initialization_and_learns(placement):
    f = replace(features(), state_embedding=torch.randn(2, 4))
    options = {"latent_dim": 4, "hidden": 8, "pair_residual_layers": 2, "readout_conditioning": True}
    torch.manual_seed(27)
    reference = HessianHead(**options)
    following = torch.rand(4)
    torch.manual_seed(27)
    head = HessianHead(**options, pair_state_conditioning=placement)
    torch.testing.assert_close(torch.rand(4), following, atol=0, rtol=0)
    for a, b in zip(head(f), reference(f), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    sum(v.square().mean() for v in head(f)).backward()
    modules = [head.pair_adapter] if placement == "adapter" else [v.update for v in head.pair_residual]
    for module in modules:
        g = module[0].weight.grad[:, -4:]
        assert torch.isfinite(g).all() and g.abs().sum() > 0
    with torch.no_grad():
        for module in modules:
            module[0].weight[:, -4:].normal_(std=0.1)
    a = head(f)
    state = f.state_embedding.clone()
    state[0] += 1
    b = head(replace(f, state_embedding=state))
    assert not torch.allclose(a[0][:3, 3:6], b[0][:3, 3:6])
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
    restored = HessianHead(**head.config)
    restored.load_state_dict(head.state_dict())
    for a, b in zip(restored(f), head(f), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_nonlinear_pair_projection_has_nonlinearity_and_trainable_layers():
    f = features()
    head = HessianHead(4, 8, readout_conditioning=True, nonlinear_pair_readout=True)
    x = torch.randn(13, 8)
    a = head.edge_head(x)
    b = head.edge_head(2*x) - head.edge_head(torch.zeros_like(x))
    assert not torch.allclose(b, 2*(a-head.edge_head(torch.zeros_like(x))))
    matrices = head(f)
    sum(v.square().mean() for v in matrices).backward()
    for p in head.edge_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for matrix in matrices:
        torch.testing.assert_close(matrix, matrix.T, atol=0, rtol=0)
    restored = HessianHead(**head.config)
    restored.load_state_dict(head.state_dict())
    for a, b in zip(restored(f), matrices, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("options", [{"pair_state_conditioning": "invalid"},
                                    {"pair_state_conditioning": "adapter"},
                                    {"pair_state_conditioning": "residual", "readout_conditioning": True}])
def test_invalid_pair_state_options_fail(options):
    with pytest.raises(ValueError):
        HessianHead(**options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("placement,nonlinear", [("none",True),("adapter",False),("residual",False)])
def test_fused_execution_matches_complete_eager_head(placement, nonlinear):
    f = features()
    f = replace(f, nodes=f.nodes.cuda(), edges=f.edges.cuda(), senders=f.senders.cuda(),
                receivers=f.receivers.cuda(), positions=f.positions.cuda(), charges=f.charges.cuda(),
                total_charge=f.total_charge.cuda(), spin_multiplicity=f.spin_multiplicity.cuda(),
                state_embedding=torch.randn(2,4,device="cuda"))
    head = HessianHead(4, 512, pair_residual_layers=2, readout_conditioning=True,
                       pair_state_conditioning=placement, nonlinear_pair_readout=nonlinear).cuda()
    torch.nn.init.normal_(head.readout_conditioner[-1].weight,std=0.01)
    reference = copy.deepcopy(head)
    head._fused_affine_readout = True
    head._fused_affine_min_pairs = 0
    a,b = head(f), reference(f)
    for x,y in zip(a,b,strict=True):
        torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-4)
    sum(v.square().mean() for v in a).backward()
    sum(v.square().mean() for v in b).backward()
    for (name,p),(_,q) in zip(head.named_parameters(),reference.named_parameters(),strict=True):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            assert (p.grad-q.grad).norm()/q.grad.norm().clamp_min(1e-12)<2e-3,name
