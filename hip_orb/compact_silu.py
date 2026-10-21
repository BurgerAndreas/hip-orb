"""Bounded FP16 saved preactivations with FP32 outputs and gradients."""

import torch
import triton
import triton.language as tl


@triton.jit
def _save(PRE, BIAS, Y, SAVED, SIZE, N: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(PRE + ids, ids < SIZE, 0)
    if HAS_BIAS:
        z += tl.load(BIAS + ids % N, ids < SIZE, 0)
    tl.store(Y + ids, z * tl.sigmoid(z), ids < SIZE)
    tl.store(SAVED + ids, tl.minimum(tl.maximum(z, -80.0), 80.0), ids < SIZE)


def save_activation(pre, bias):
    out = pre
    saved = torch.empty_like(pre, dtype=torch.float16)
    if pre.numel():
        _save[(triton.cdiv(pre.numel(), 1024),)](
            pre, bias, out, saved, pre.numel(), pre.shape[-1], bias is not None, 1024
        )
    return out, saved
