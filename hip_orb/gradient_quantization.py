"""Fuse gradient maxima and bias reductions before local FP16 conversion."""

import torch
import triton
import triton.language as tl

from hip_orb.scaled_silu import _scale_cast


@triton.jit
def _parts(
    G, MAX, BIAS, M, N: tl.constexpr, WRITE_BIAS: tl.constexpr, RM: tl.constexpr, CN: tl.constexpr
):
    rows = tl.program_id(0) * RM + tl.arange(0, RM)
    cols = tl.program_id(1) * CN + tl.arange(0, CN)
    value = tl.load(
        G + rows[:, None] * N + cols[None, :], (rows[:, None] < M) & (cols[None, :] < N), 0
    )
    tl.store(
        MAX + tl.program_id(0) * tl.cdiv(N, CN) + tl.program_id(1),
        tl.max(tl.max(tl.abs(value), 0), 0),
    )
    if WRITE_BIAS:
        tl.store(BIAS + tl.program_id(0) * N + cols, tl.sum(value, 0), cols < N)


def quantize_gradient(g, *, need_bias=False):
    m, n = g.shape
    maxima = g.new_empty((triton.cdiv(m, 32), triton.cdiv(n, 128)))
    parts = g.new_empty((triton.cdiv(m, 32), n)) if need_bias else None
    _parts[(triton.cdiv(m, 32), triton.cdiv(n, 128))](g, maxima, parts, m, n, need_bias, 32, 128)
    maximum = maxima.amax()
    q = torch.empty_like(g, dtype=torch.float16)
    _scale_cast[(triton.cdiv(g.numel(), 1024),)](g, maximum, q, g.numel(), 1024)
    return q, maximum, parts.sum(0) if need_bias else None
