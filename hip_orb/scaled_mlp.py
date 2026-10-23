"""Compact MLP activations and locally scaled FP16 matrix products."""

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from hip_orb.compact_silu import _save
from hip_orb.scaled_silu import _restore, _scale_cast


@triton.jit
def _gradient(
    PRE,
    G,
    MAXIMUM,
    DZ,
    PART,
    MAX,
    M,
    N: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    RM: tl.constexpr,
    CN: tl.constexpr,
):
    row = tl.program_id(0) * RM + tl.arange(0, RM)
    col = tl.program_id(1) * CN + tl.arange(0, CN)
    ids = row[:, None] * N + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < N)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.load(MAXIMUM), 1.0e-30) / 1024.0)))
    z = tl.load(PRE + ids, mask, 0).to(tl.float32)
    g = tl.load(G + ids, mask, 0) * scale
    sig = tl.sigmoid(z)
    v = g * sig * (1.0 + z * (1.0 - sig))
    tl.store(DZ + ids, v, mask)
    tl.store(
        MAX + tl.program_id(0) * tl.cdiv(N, CN) + tl.program_id(1), tl.max(tl.max(tl.abs(v), 0), 0)
    )
    if WRITE_BIAS:
        tl.store(PART + tl.program_id(0) * N + col, tl.sum(v, 0), col < N)


def _quantize(g, pre=None, previous=None, need_bias=False, *, fused_parts=False):
    m, n = g.shape
    if pre is None and fused_parts:
        from hip_orb.gradient_quantization import quantize_gradient
        return quantize_gradient(g, need_bias=need_bias)
    if pre is None:
        maximum = g.abs().amax()
        bias = g.sum(0) if need_bias else None
    else:
        maxima = g.new_empty((triton.cdiv(m, 32), triton.cdiv(n, 128)))
        partial = g.new_empty((triton.cdiv(m, 32), n)) if need_bias else None
        _gradient[(triton.cdiv(m, 32), triton.cdiv(n, 128))](
            pre, g, previous, g, partial, maxima, m, n, need_bias, 32, 128
        )
        maximum = maxima.amax()
        bias = partial.sum(0) if need_bias else None
    q = torch.empty_like(g, dtype=torch.float16)
    _scale_cast[(triton.cdiv(g.numel(), 1024),)](g, maximum, q, g.numel(), 1024)
    return q, maximum, bias


def _product(a, b, maximum, activation_maximum=None):
    out = torch.mm(a, b, out_dtype=torch.float32)
    if out.numel():
        _restore[(triton.cdiv(out.numel(), 1024),)](out, maximum, out, out.numel(), 1024)
    if out.numel() and activation_maximum is not None:
        _restore[(triton.cdiv(out.numel(), 1024),)](
            out, activation_maximum, out, out.numel(), 1024
        )
    return out


def _activation(a, w, b, maximum):
    pre = _product(a, w.T, maximum)
    h = torch.empty_like(pre)
    saved = torch.empty_like(pre, dtype=torch.float16)
    _save[(triton.cdiv(pre.numel(), 1024),)](
        pre, b, h, saved, pre.numel(), pre.shape[1], b is not None, 1024
    )
    quantized, maximum, _ = _quantize(h)
    return quantized, saved, maximum


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, w3, b3, fused_reductions):
        ctx.fused_reductions = fused_reductions
        width = x.shape[-1]
        pad = (-width) % 16
        if (
            getattr(x, "_hip_zero_padded", False)
            and x.ndim == 2
            and x.stride() == (width + pad, 1)
            and x.storage_offset() == 0
        ):
            a = x.as_strided((len(x), width + pad), (width + pad, 1))
        else:
            a = x.reshape(-1, width)
            if pad:
                a = F.pad(a, (0, pad))
        a, sa, _ = _quantize(a.contiguous())
        wh1 = F.pad(w1.half(), (0, pad)) if pad else w1.half()
        wh2 = w2.half()
        wh3 = w3.half()
        h1, p1, s1 = _activation(a, wh1, b1, sa)
        h2, p2, s2 = _activation(h1, wh2, b2, s1)
        y = _product(h2, wh3.T, s2)
        if b3 is not None:
            y.add_(b3)
        ctx.save_for_backward(x, w1, b1, w2, b2, w3, b3, a, wh1, wh2, wh3, h1, p1, h2, p2, sa, s1, s2)
        return y.reshape(*x.shape[:-1], len(w3))

    @staticmethod
    def backward(ctx, g):
        x, w1, b1, w2, b2, w3, b3, a, wh1, wh2, wh3, h1, p1, h2, p2, sa, sh1, sh2 = ctx.saved_tensors
        needs = ctx.needs_input_grad[:7]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = F.linear(F.silu(F.linear(F.silu(F.linear(x, w1, b1)), w2, b2)), w3, b3)
                params = (x, w1, b1, w2, b2, w3, b3)
                values = iter(
                    torch.autograd.grad(
                        y,
                        [p for p, need in zip(params, needs, strict=True) if need],
                        g,
                        create_graph=True,
                    )
                )
            return tuple(next(values) if need else None for need in needs) + (None,)
        q3, s3, db3 = _quantize(g.reshape(-1, len(w3)).contiguous(), need_bias=needs[6],
                                  fused_parts=ctx.fused_reductions)
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
        dw1 = _product(q1.T, a, s1, sa)[:, : x.shape[-1]].contiguous() if needs[1] else None
        dx = _product(q1, wh1, s1)[:, : x.shape[-1]].reshape(x.shape) if needs[0] else None
        return dx, dw1, db1, dw2, db2, dw3, db3, None


def scaled_mlp(x, w1, b1, w2, b2, w3, b3, *, fused_reductions=True):
    if x.is_cuda and x.dtype == w1.dtype == torch.float32 and x.numel():
        return _MLP.apply(x, w1, b1, w2, b2, w3, b3, fused_reductions)
    return F.linear(F.silu(F.linear(F.silu(F.linear(x, w1, b1)), w2, b2)), w3, b3)


@contextmanager
def scaled_mlp_layers(model, *, fused_reductions=True):
    changed = []
    try:
        for name, module in model.named_modules():
            if not name.startswith("head.") or not isinstance(module, torch.nn.Sequential):
                continue
            layers = list(module.children())
            if len(layers) == 6 and isinstance(layers[-1], torch.nn.Identity):
                layers = layers[:-1]
            if len(layers) != 5 or not all(
                isinstance(layers[i], torch.nn.Linear) for i in (0, 2, 4)
            ):
                continue
            if (
                not all(isinstance(layers[i], torch.nn.SiLU) for i in (1, 3))
                or layers[-1].out_features < 128
            ):
                continue
            changed.append((module, module.forward))
            a, b, c = layers[0], layers[2], layers[4]

            def forward(self, x, a=a, b=b, c=c, original=module.forward):
                if x.numel() // x.shape[-1] < 16384:
                    return original(x)
                return scaled_mlp(x, a.weight, a.bias, b.weight, b.bias, c.weight, c.bias,
                                  fused_reductions=fused_reductions)

            module.forward = MethodType(forward, module)
        yield
    finally:
        for m, f in reversed(changed):
            m.forward = f
