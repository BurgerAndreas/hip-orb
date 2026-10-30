"""Quantize frozen molecular state inputs without materializing repeated FP32 rows."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from hip_orb.scaled_mlp import _activation, _product, _quantize


@triton.jit
def _joined_input(X, E, OWNER, MAXIMUM, OUT, M,
                  KX: tl.constexpr, KE: tl.constexpr, KP: tl.constexpr,
                  XS: tl.constexpr, ES: tl.constexpr,
                  MEASURE: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = ids // KP, ids % KP
    owner = tl.load(OWNER + row, row < M, 0)
    x = tl.load(X + row * XS + col, (row < M) & (col < KX), 0)
    e = tl.load(E + owner * ES + col - KX,
                (row < M) & (col >= KX) & (col < KX + KE), 0)
    value = tl.where(col < KX, x, e)
    if MEASURE:
        tl.store(OUT + tl.program_id(0), tl.max(tl.abs(value), 0))
    else:
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.load(MAXIMUM), 1e-30) / 1024.0)))
        tl.store(OUT + ids, value / scale, row < M)


def _quantize_joined(x, state, owners):
    width = x.shape[1] + state.shape[1]
    padded = triton.cdiv(width, 16) * 16
    blocks = triton.cdiv(len(x) * padded, 1024)
    maxima = x.new_empty((blocks,))
    _joined_input[(blocks,)](x, state, owners, None, maxima, len(x), x.shape[1], state.shape[1],
                             padded, x.stride(0), state.stride(0), True, 1024)
    maximum = maxima.amax()
    quantized = torch.empty((len(x), padded), device=x.device, dtype=torch.float16)
    _joined_input[(blocks,)](x, state, owners, maximum, quantized, len(x), x.shape[1], state.shape[1],
                             padded, x.stride(0), state.stride(0), False, 1024)
    return quantized, maximum


class _StateMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, state, owners, w1, b1, w2, b2, w3, b3, fused_reductions, fused_activations):
        ctx.fused_reductions = fused_reductions
        a, sa = _quantize_joined(x, state, owners)
        pad = a.shape[1] - w1.shape[1]
        wh1 = F.pad(w1.half(), (0, pad)) if pad else w1.half()
        wh2, wh3 = w2.half(), w3.half()
        h1, p1, s1 = _activation(a, wh1, b1, sa, fused=fused_activations)
        h2, p2, s2 = _activation(h1, wh2, b2, s1, fused=fused_activations)
        y = _product(h2, wh3.T, s2)
        if b3 is not None:
            y.add_(b3)
        ctx.save_for_backward(x, state, owners, w1, b1, w2, b2, w3, b3,
                              a, wh1, wh2, wh3, h1, p1, h2, p2, sa, s1, s2)
        return y

    @staticmethod
    def backward(ctx, g):
        (x, state, owners, w1, b1, w2, b2, w3, b3,
         a, wh1, wh2, wh3, h1, p1, h2, p2, sa, sh1, sh2) = ctx.saved_tensors
        needs = tuple(ctx.needs_input_grad[i] for i in (0, 3, 4, 5, 6, 7, 8))
        if torch.is_grad_enabled():
            with torch.enable_grad():
                value = torch.cat((x, state[owners]), dim=-1)
                y = F.linear(F.silu(F.linear(F.silu(F.linear(value, w1, b1)), w2, b2)), w3, b3)
                params = (x, w1, b1, w2, b2, w3, b3)
                gradients = iter(torch.autograd.grad(
                    y, [p for p, need in zip(params, needs, strict=True) if need], g, create_graph=True))
                dx, dw1, db1, dw2, db2, dw3, db3 = (next(gradients) if need else None for need in needs)
            return dx, None, None, dw1, db1, dw2, db2, dw3, db3, None, None
        q3, s3, db3 = _quantize(g.contiguous(), need_bias=needs[6], fused_parts=ctx.fused_reductions)
        dw3 = _product(q3.T, h2, s3, sh2) if needs[5] else None
        raw2 = torch.mm(q3, wh3, out_dtype=torch.float32)
        del q3
        q2, s2, db2 = _quantize(raw2, p2, s3, needs[4])
        del raw2
        dw2 = _product(q2.T, h1, s2, sh1) if needs[3] else None
        raw1 = torch.mm(q2, wh2, out_dtype=torch.float32)
        del q2
        q1, s1, db1 = _quantize(raw1, p1, s2, needs[2])
        del raw1
        dw1 = _product(q1.T, a, s1, sa)[:, :w1.shape[1]].contiguous() if needs[1] else None
        # State encodings are frozen. Do not compute their discarded row gradients.
        px = triton.cdiv(x.shape[1], 16) * 16
        dx = _product(q1, wh1[:, :px], s1)[:, :x.shape[1]] if needs[0] else None
        return dx, None, None, dw1, db1, dw2, db2, dw3, db3, None, None


def state_mlp(module, x, state, owners, *, fused_reductions=None, fused_activations=None):
    """Retain the scaled MLP policy and the existing concatenated-weight checkpoint layout."""
    supported = (x.is_cuda and x.dtype == state.dtype == module[0].weight.dtype == torch.float32
                 and len(x) >= 16384 and not state.requires_grad
                 and module[4].out_features >= 128
                 and x.stride(1) == state.stride(1) == 1 and owners.is_contiguous())
    if supported:
        if fused_reductions is None:
            fused_reductions = getattr(module, "_state_fused_reductions", True)
        if fused_activations is None:
            fused_activations = getattr(module, "_state_fused_activations", True)
        a, b, c = module[0], module[2], module[4]
        return _StateMLP.apply(x, state, owners, a.weight, a.bias, b.weight, b.bias,
                               c.weight, c.bias, fused_reductions, fused_activations)
    return module(torch.cat((x, state[owners]), dim=-1))
