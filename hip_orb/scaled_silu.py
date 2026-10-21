"""Local gradient scaling for FP16 MLP products with FP32 accumulation."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _scale_cast(G, S, Q, SIZE, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    maximum = tl.maximum(tl.load(S), 1.0e-30)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 1024.0)))
    g = tl.load(G + ids, ids < SIZE, 0)
    tl.store(Q + ids, g / scale, ids < SIZE)


@triton.jit
def _restore(P, S, Y, SIZE, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.load(S), 1.0e-30) / 1024.0)))
    value = tl.load(P + ids, ids < SIZE, 0)
    tl.store(Y + ids, value * scale, ids < SIZE)


@triton.jit
def _derivative(
    PRE,
    BIAS,
    G,
    DZ,
    PART,
    MAX,
    M,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    RM: tl.constexpr,
    CN: tl.constexpr,
):
    rows = tl.program_id(0) * RM + tl.arange(0, RM)
    cols = tl.program_id(1) * CN + tl.arange(0, CN)
    ids = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    z = tl.load(PRE + ids, mask, 0).to(tl.float32)
    if HAS_BIAS:
        z += tl.load(BIAS + cols, cols < N, 0)[None, :]
    g = tl.load(G + ids, mask, 0)
    sig = tl.sigmoid(z)
    v = g * sig * (1.0 + z * (1.0 - sig))
    tl.store(DZ + ids, v, mask)
    maximum = tl.max(tl.max(tl.abs(v), 0), 0)
    tl.store(MAX + tl.program_id(0) * tl.cdiv(N, CN) + tl.program_id(1), maximum)
    if WRITE_BIAS:
        tl.store(PART + tl.program_id(0) * N + cols, tl.sum(v, 0), cols < N)


class _ScaledSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, half_forward, compact):
        from hip_orb.precision_kernels import _silu_forward

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
            a = F.pad(x.reshape(-1, width), (0, pad)) if pad else x.reshape(-1, width)
        wp = F.pad(w, (0, pad)) if pad else w
        ah, wh = a.half(), wp.half()
        pre = torch.mm(ah, wh.T, out_dtype=torch.float32) if half_forward else a @ wp.T
        if compact:
            from hip_orb.compact_silu import save_activation

            y, pre = save_activation(pre, b)
        else:
            y = torch.empty_like(pre)
            if y.numel():
                _silu_forward[(triton.cdiv(y.numel(), 1024),)](
                    pre, b, y, y.numel(), y.shape[1], b is not None, 1024
                )
        ctx.compact = compact
        ctx.save_for_backward(x, w, b, ah, wh, pre)
        return y.reshape(*x.shape[:-1], len(w))

    @staticmethod
    def backward(ctx, g):
        x, w, b, a, wh, pre = ctx.saved_tensors
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = F.silu(F.linear(x, w, b))
                params = (x, w, b)
                vals = iter(
                    torch.autograd.grad(
                        y,
                        [
                            p
                            for p, need in zip(params, ctx.needs_input_grad[:3], strict=True)
                            if need
                        ],
                        g,
                        create_graph=True,
                    )
                )
            return tuple(next(vals) if need else None for need in ctx.needs_input_grad[:3]) + (
                None,
                None,
            )
        g = g.contiguous().reshape_as(pre)
        m, n = g.shape
        partial = g.new_empty((triton.cdiv(m, 32), n)) if ctx.needs_input_grad[2] else None
        dz = torch.empty_like(g)
        maxima = g.new_empty((triton.cdiv(m, 32), triton.cdiv(n, 128)))
        if m:
            _derivative[(triton.cdiv(m, 32), triton.cdiv(n, 128))](
                pre,
                None if ctx.compact else b,
                g,
                dz,
                partial,
                maxima,
                m,
                n,
                b is not None and not ctx.compact,
                ctx.needs_input_grad[2],
                32,
                128,
            )
        maximum = maxima.amax() if m else dz.new_ones(())
        q = torch.empty_like(dz, dtype=torch.float16)
        if m:
            _scale_cast[(triton.cdiv(dz.numel(), 1024),)](dz, maximum, q, dz.numel(), 1024)
        del dz

        def product(left, right):
            v = torch.mm(left, right, out_dtype=torch.float32)
            out = v
            if v.numel():
                _restore[(triton.cdiv(v.numel(), 1024),)](v, maximum, out, v.numel(), 1024)
            return out

        dx = product(q, wh)[:, : x.shape[-1]].reshape(x.shape) if ctx.needs_input_grad[0] else None
        dw = product(q.T, a)[:, : x.shape[-1]].contiguous() if ctx.needs_input_grad[1] else None
        db = partial.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dw, db, None, None


def scaled_linear_silu(x, weight, bias=None, *, compact=False):
    if x.is_cuda and x.dtype == weight.dtype == torch.float32 and x.numel():
        return _ScaledSiLU.apply(x, weight, bias, True, compact)
    return F.silu(F.linear(x, weight, bias))
