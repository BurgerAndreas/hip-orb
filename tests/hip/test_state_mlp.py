"""Frozen-state input fusion preserves the existing scaled MLP operands and gradients."""

import copy

import pytest
import torch
from torch.nn import functional as F

from hip_orb.model import Features, HessianHead, mlp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("width", [512, 791])
@pytest.mark.parametrize("gradient_scale", [1.0, 1e-12])
def test_state_input_quantization_and_gradients_match_scaled_mlp(width, gradient_scale):
    from hip_orb.scaled_mlp import _quantize, scaled_mlp
    from hip_orb.state_mlp import _quantize_joined, state_mlp

    torch.manual_seed(137)
    rows, state_dim = 16384, 256
    x = torch.randn(rows, width+9, device="cuda")[:, :width].requires_grad_()
    state = torch.randn(5, state_dim+7, device="cuda")[:, :state_dim]
    owners = torch.arange(rows, device="cuda") * 5 // rows
    joined = torch.cat((x, state[owners]), -1)
    padded = F.pad(joined, (0, (-joined.shape[1]) % 16))
    qa, sa, _ = _quantize(padded)
    qb, sb = _quantize_joined(x, state, owners)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    torch.testing.assert_close(qa, qb, atol=0, rtol=0)
    module = mlp(width+state_dim, 128, 128).cuda()
    args = tuple(p for layer in (module[0], module[2], module[4]) for p in (layer.weight, layer.bias))
    wanted = scaled_mlp(joined, *args)
    got = state_mlp(module, x, state, owners)
    torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    g = torch.randn_like(got) * gradient_scale
    aa = torch.autograd.grad(got, (x, *args), g)
    bb = torch.autograd.grad(wanted, (x, *args), g)
    for a, b in zip(aa, bb, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=2e-5*gradient_scale, rtol=3e-4)


@pytest.mark.parametrize("placement", ["adapter", "residual"])
def test_state_input_fusion_matches_complete_head_and_checkpoint(placement):
    from hip_orb.fused_linear_silu import fused_linear_silu_layers
    from hip_orb.scaled_mlp import scaled_mlp_layers

    torch.manual_seed(29)
    n, count, latent = 130, 2, 4
    node_count = n*count
    receivers = torch.cat([torch.arange(n, device="cuda").roll(-1)+b*n for b in range(count)])
    f = Features(torch.randn(node_count, latent, device="cuda"),
                 torch.randn(node_count, latent, device="cuda"),
                 torch.arange(node_count, device="cuda"), receivers,
                 torch.randn(node_count, 3, device="cuda"), torch.randn(node_count, device="cuda"),
                 torch.tensor([n]*count), total_charge=torch.tensor([0.,2.],device="cuda"),
                 spin_multiplicity=torch.tensor([1.,3.],device="cuda"),
                 state_embedding=torch.randn(count,latent,device="cuda"))
    head = HessianHead(latent, 512, mlp_hidden=32, pair_residual_layers=2, readout_conditioning=True,
                       pair_state_conditioning=placement, radial_features="bessel").cuda()
    modules = [head.pair_adapter] if placement == "adapter" else [layer.update for layer in head.pair_residual]
    with torch.no_grad():
        for module in modules:
            module[0].weight[:, -latent:].normal_(std=0.01)
        head.readout_conditioner[-1].weight.normal_(std=0.01)
    other = copy.deepcopy(head)
    head._join_radial_pairs = other._join_radial_pairs = True
    other._fused_state_inputs = True
    wrappers = []
    outputs = []
    for h in (head, other):
        wrapper = torch.nn.Module()
        wrapper.head = h
        wrappers.append(wrapper)
        with scaled_mlp_layers(wrapper), fused_linear_silu_layers(wrapper, scaled=True, compact=True):
            # Isolate trainable blocks from nondeterministic Coulomb diagonal atomics.
            output = h(f, residual_only=True)
            sum(v.square().mean() for v in output).backward()
            outputs.append(output)
    for a,b in zip(*outputs,strict=True):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for (name,a),(_,b) in zip(head.named_parameters(),other.named_parameters(),strict=True):
        if a.grad is not None:
            assert torch.isfinite(b.grad).all()
            assert (a.grad-b.grad).norm()/a.grad.norm().clamp_min(1e-12)<1e-3,name
    assert head.config==other.config
    head.load_state_dict(other.state_dict())


def test_state_mlp_higher_derivatives_and_frozen_inputs():
    from hip_orb.state_mlp import state_mlp

    x = torch.randn(16384, 4, device="cuda", requires_grad=True)
    state = torch.randn(2, 4, device="cuda")
    owners = torch.arange(len(x),device="cuda")*2//len(x)
    module = mlp(8, 8, 128).cuda()
    a = state_mlp(module,x,state,owners)
    b = module(torch.cat((x,state[owners]),-1))
    params = (x,*module.parameters())
    aa = torch.autograd.grad(a.sum(),params,create_graph=True)
    bb = torch.autograd.grad(b.sum(),params,create_graph=True)
    for av,bv in zip(aa,bb,strict=True):
        torch.testing.assert_close(av,bv)
    assert torch.isfinite(torch.autograd.grad(state_mlp(module,x.detach(),state,owners).sum(),module[0].weight)[0]).all()
