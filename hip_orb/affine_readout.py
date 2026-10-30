"""Molecular affine conditioning fused into an FP32 readout projection."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _forward(X, S, T, OWNER, W, B, Y, PRE, M,
             K: tl.constexpr, N: tl.constexpr,
             XS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
             BIAS: tl.constexpr, SILU: tl.constexpr, COMPACT: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    owner = tl.load(OWNER + rows, rows < M, 0)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        mask = (rows[:, None] < M) & (k[None, :] < K)
        x = tl.load(X + rows[:, None] * XS + k[None, :], mask, 0)
        scale = tl.load(S + owner[:, None] * SS + k[None, :], mask, 0)
        shift = tl.load(T + owner[:, None] * TS + k[None, :], mask, 0)
        a = x * (1.0 + scale) + shift
        a = tl.where(mask, a, 0.0)
        w = tl.load(W + cols[None, :] * K + k[:, None],
                    (k[:, None] < K) & (cols[None, :] < N), 0)
        acc += tl.dot(a, w, input_precision="tf32x3")
    if BIAS:
        acc += tl.load(B + cols, cols < N, 0)[None, :]
    ids = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    if SILU:
        saved = tl.minimum(tl.maximum(acc, -80.0), 80.0) if COMPACT else acc
        tl.store(PRE + ids, saved, mask)
        acc = acc * tl.sigmoid(acc)
    tl.store(Y + ids, acc, mask)


@triton.jit
def _condition(X, S, T, OWNER, A, M, K: tl.constexpr,
               XS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
               BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = ids // K, ids % K
    owner = tl.load(OWNER + row, row < M, 0)
    x = tl.load(X + row * XS + col, row < M, 0)
    s = tl.load(S + owner * SS + col, row < M, 0)
    t = tl.load(T + owner * TS + col, row < M, 0)
    tl.store(A + ids, x * (1.0 + s) + t, row < M)


@triton.jit
def _backward_affine(X, S, OWNER, DA, DX, DS, DT, M, K: tl.constexpr,
                     XS: tl.constexpr, SS: tl.constexpr,
                     NEED_X: tl.constexpr, NEED_S: tl.constexpr, NEED_T: tl.constexpr,
                     RM: tl.constexpr, CK: tl.constexpr):
    row = tl.program_id(0) * RM + tl.arange(0, RM)
    col = tl.program_id(1) * CK + tl.arange(0, CK)
    mask = (row[:, None] < M) & (col[None, :] < K)
    owner = tl.load(OWNER + row, row < M, 0)
    g = tl.load(DA + row[:, None] * K + col[None, :], mask, 0)
    if NEED_X:
        scale = tl.load(S + owner[:, None] * SS + col[None, :], mask, 0)
        tl.store(DX + row[:, None] * K + col[None, :], g * (1.0 + scale), mask)
    if NEED_S or NEED_T:
        # A batch tile usually contains one molecule. Reduce before atomic writes.
        first = tl.min(tl.where(row < M, owner, 2147483647), 0)
        last = tl.max(tl.where(row < M, owner, -1), 0)
        if NEED_S:
            x = tl.load(X + row[:, None] * XS + col[None, :], mask, 0)
        for molecule in range(first, last + 1):
            selected = (row < M) & (owner == molecule)
            if NEED_S:
                value = tl.sum(tl.where(selected[:, None], g * x, 0.0), 0)
                tl.atomic_add(DS + molecule * K + col, value, col < K, sem="relaxed")
            if NEED_T:
                value = tl.sum(tl.where(selected[:, None], g, 0.0), 0)
                tl.atomic_add(DT + molecule * K + col, value, col < K, sem="relaxed")


def _reference(x, scale, shift, owners, weight, bias, silu):
    y = F.linear(x * (1 + scale[owners]) + shift[owners], weight, bias)
    return F.silu(y) if silu else y


class _AffineReadout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, owners, weight, bias, silu, compact):
        m, k = x.shape
        n = len(weight)
        y = x.new_empty((m, n))
        pre = x.new_empty((m, n), dtype=torch.float16 if compact else x.dtype) if silu else None
        _forward[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            x, scale, shift, owners, weight, bias, y, pre, m, k, n,
            x.stride(0), scale.stride(0), shift.stride(0), bias is not None,
            silu, compact, 32, 32, 32, enable_fp_fusion=False)
        ctx.save_for_backward(x, scale, shift, owners, weight, bias, pre)
        ctx.silu = silu
        return y

    @staticmethod
    def backward(ctx, grad):
        x, scale, shift, owners, weight, bias, pre = ctx.saved_tensors
        need_x, need_s, need_t, _, need_w, need_b = ctx.needs_input_grad[:6]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = _reference(x, scale, shift, owners, weight, bias, ctx.silu)
                params = (x, scale, shift, weight, bias)
                needs = (need_x, need_s, need_t, need_w, need_b)
                values = iter(torch.autograd.grad(y, [p for p, n in zip(params, needs, strict=True) if n],
                                                 grad, create_graph=True))
                dx, ds, dt, dw, db = (next(values) if n else None for n in needs)
            return dx, ds, dt, None, dw, db, None, None
        dz = grad.contiguous()
        db = None
        if ctx.silu:
            from hip_orb.precision_kernels import _silu_backward

            partial = dz.new_empty((triton.cdiv(len(x), 32), len(weight))) if need_b else None
            derived = torch.empty_like(dz)
            _silu_backward[(triton.cdiv(len(x), 32), triton.cdiv(len(weight), 128))](
                pre, None, dz, partial, derived, len(x), len(weight), False, need_b, 32, 128)
            dz = derived
            db = partial.sum(0) if need_b else None
        elif need_b:
            db = dz.sum(0)
        dw = None
        if need_w:
            conditioned = torch.empty_like(x, memory_format=torch.contiguous_format)
            _condition[(triton.cdiv(x.numel(), 1024),)](
                x, scale, shift, owners, conditioned, len(x), x.shape[1],
                x.stride(0), scale.stride(0), shift.stride(0), 1024, enable_fp_fusion=False)
            dw = dz.T @ conditioned
        dx = ds = dt = None
        if need_x or need_s or need_t:
            da = dz @ weight
            dx = torch.empty_like(x, memory_format=torch.contiguous_format) if need_x else None
            ds = scale.new_zeros(scale.shape) if need_s else None
            dt = shift.new_zeros(shift.shape) if need_t else None
            _backward_affine[(triton.cdiv(len(x), 64), triton.cdiv(x.shape[1], 128))](
                x, scale, owners, da, dx, ds, dt, len(x), x.shape[1], x.stride(0), scale.stride(0),
                need_x, need_s, need_t, 64, 128, enable_fp_fusion=False)
        return dx, ds, dt, None, dw, db, None, None


def affine_readout(x, scale, shift, owners, weight, bias=None, *, silu=False, compact=False):
    """Owners must be sorted molecule IDs; weights and channel dimensions stay unchanged."""
    supported = (x.is_cuda and x.dtype == scale.dtype == shift.dtype == weight.dtype == torch.float32
                 and x.numel() and x.ndim == 2 and x.stride(1) == 1
                 and scale.stride(1) == shift.stride(1) == 1 and weight.is_contiguous()
                 and owners.is_contiguous() and not torch.are_deterministic_algorithms_enabled())
    if supported:
        return _AffineReadout.apply(x, scale, shift, owners, weight, bias, silu, compact)
    return _reference(x, scale, shift, owners, weight, bias, silu)
