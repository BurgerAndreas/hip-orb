"""Native FP32 linear forwards with locally scaled FP16 backward products."""

from contextlib import contextmanager
from functools import lru_cache
from types import MethodType

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from hip_orb.scaled_silu import _restore, _scale_cast


@triton.jit
def _max_blocks(G, PART, SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(G + i, i < SIZE, 0)
    tl.store(PART + tl.program_id(0), tl.max(tl.abs(value), 0))


def quantize_gradient(g):
    partial = g.new_empty((triton.cdiv(g.numel(), 4096),))
    _max_blocks[(len(partial),)](g, partial, g.numel(), 4096)
    maximum = partial.amax()
    q = torch.empty_like(g, dtype=torch.float16)
    _scale_cast[(triton.cdiv(g.numel(), 1024),)](g, maximum, q, g.numel(), 1024)
    return q, maximum


def scaled_product(a, b, maximum):
    result = torch.mm(a, b, out_dtype=torch.float32)
    if result.numel():
        _restore[(triton.cdiv(result.numel(), 1024),)](
            result, maximum, result, result.numel(), 1024
        )
    return result


class _BackwardLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, original):
        ctx.save_for_backward(x, weight, bias)
        return original(x)

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = F.linear(x, weight, bias)
                inputs = [p for p, need in zip((x, weight, bias), needs, strict=True) if need]
                values = iter(torch.autograd.grad(y, inputs, grad, create_graph=True))
            return tuple(next(values) if need else None for need in needs) + (None,)
        g = grad.reshape(-1, len(weight)).contiguous()
        if not g.numel():
            return (
                torch.zeros_like(x) if needs[0] else None,
                torch.zeros_like(weight) if needs[1] else None,
                torch.zeros_like(bias) if needs[2] else None,
                None,
            )
        q, maximum = quantize_gradient(g)
        width = x.shape[-1]
        pad = (-width) % 16
        dx = dw = None
        if needs[0]:
            w = weight.half()
            if pad:
                w = F.pad(w, (0, pad))
            dx = scaled_product(q, w, maximum)[:, :width].reshape(x.shape)
        if needs[1]:
            a = x.reshape(-1, width).half()
            if pad:
                a = F.pad(a, (0, pad))
            dw = scaled_product(q.T, a, maximum)[:, :width].contiguous()
        db = g.sum(0) if needs[2] else None
        return dx, dw, db, None


@contextmanager
def backward_precision(model, *, minimum_rows=16384, square=False):
    changed = []
    try:
        for name, module in model.named_modules():
            if not name.startswith("head.") or not isinstance(module, torch.nn.Linear):
                continue
            if module.out_features < 128:
                continue
            if square and (module.in_features != module.out_features or module.out_features < 512):
                continue
            original = module.forward
            changed.append((module, original))

            def forward(self, x, original=original):
                if (
                    not x.is_cuda
                    or x.dtype != torch.float32
                    or x.numel() // x.shape[-1] < minimum_rows
                ):
                    return original(x)
                return _BackwardLinear.apply(x, self.weight, self.bias, original)

            module.forward = MethodType(forward, module)
        yield
    finally:
        for module, original in reversed(changed):
            module.forward = original
