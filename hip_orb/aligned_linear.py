"""Align FP32 linear inputs for Tensor Core kernels without model padding."""

import torch
from torch.nn import functional as F


class _AlignedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        width = x.shape[-1]
        padding = (-width) % 16
        if (getattr(x, "_hip_zero_padded", False) and x.ndim == 2
                and x.stride() == (width + padding, 1) and x.storage_offset() == 0):
            a = x.as_strided((x.shape[0], width + padding), (width + padding, 1))
        else:
            a = F.pad(x.reshape(-1, width), (0, padding))
        w = F.pad(weight, (0, padding))
        y = F.linear(a, w, bias)
        ctx.save_for_backward(a, w, x, weight, bias)
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        a, w, x, weight, bias = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = F.linear(x, weight, bias)
                inputs = [v for v, need in zip((x, weight, bias), ctx.needs_input_grad, strict=True) if need]
                grads = iter(torch.autograd.grad(y, inputs, grad, create_graph=True))
            return tuple(next(grads) if need else None for need in ctx.needs_input_grad)
        g = grad.reshape(-1, weight.shape[0]).contiguous()
        dx = (g @ w)[:, :x.shape[-1]].reshape(x.shape) if need_x else None
        dw = (g.T @ a)[:, :x.shape[-1]].contiguous() if need_w else None
        db = g.sum(0) if need_b else None
        return dx, dw, db


def aligned_linear(x, weight, bias=None):
    """Pad only internal matrix storage. Parameters and results keep their shapes."""
    if (x.is_cuda and x.dtype == weight.dtype == torch.float32
            and x.shape[-1] % 16 and weight.shape[0] >= 128):
        return _AlignedLinear.apply(x, weight, bias)
    return F.linear(x, weight, bias)
