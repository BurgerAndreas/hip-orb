"""FP32 matrix products with fused SiLU and bias-gradient reductions."""

from contextlib import contextmanager
from itertools import pairwise
from types import MethodType

import torch
from torch.nn import functional as F


class _LinearSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, compact):
        import triton

        from hip_orb.precision_kernels import _silu_forward

        width = x.shape[-1]
        padding = (-width) % 16
        if (
            getattr(x, "_hip_zero_padded", False)
            and x.ndim == 2
            and x.stride() == (width + padding, 1)
            and x.storage_offset() == 0
        ):
            a = x.as_strided((len(x), width + padding), (width + padding, 1))
        else:
            a = F.pad(x.reshape(-1, width), (0, padding)) if padding else x.reshape(-1, width)
        w = F.pad(weight, (0, padding)) if padding else weight
        pre = torch.mm(a, w.T)
        y = torch.empty_like(pre)
        if y.numel() and not compact:
            _silu_forward[(triton.cdiv(y.numel(), 1024),)](
                pre, bias, y, y.numel(), y.shape[1], bias is not None, 1024
            )
        if compact and pre.numel():
            from hip_orb.compact_silu import save_activation

            y, saved = save_activation(pre, bias)
            pre = saved
        ctx.compact = bool(compact and pre.numel())
        ctx.save_for_backward(x, weight, bias, a, w, pre)
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias, a, w, pre = ctx.saved_tensors
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = F.silu(F.linear(x, weight, bias))
                inputs = [
                    v
                    for v, need in zip((x, weight, bias), ctx.needs_input_grad[:3], strict=True)
                    if need
                ]
                values = iter(torch.autograd.grad(y, inputs, grad, create_graph=True))
            return tuple(next(values) if need else None for need in ctx.needs_input_grad[:3]) + (
                None,
            )
        import triton

        from hip_orb.precision_kernels import _silu_backward

        g = grad.contiguous().reshape_as(pre)
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        partial = g.new_empty((triton.cdiv(len(g), 32), g.shape[1])) if need_b else None
        dz = torch.empty_like(g)
        if g.numel():
            _silu_backward[(triton.cdiv(len(g), 32), triton.cdiv(g.shape[1], 128))](
                pre,
                None if ctx.compact else bias,
                g,
                partial,
                dz,
                g.shape[0],
                g.shape[1],
                bias is not None and not ctx.compact,
                need_b,
                32,
                128,
            )
        dx = (dz @ w)[:, : x.shape[-1]].reshape(x.shape) if need_x else None
        dw = (dz.T @ a)[:, : x.shape[-1]].contiguous() if need_w else None
        db = partial.sum(0) if need_b else None
        return dx, dw, db, None


def linear_silu(x, weight, bias=None, *, compact=False):
    if x.is_cuda and x.dtype == weight.dtype == torch.float32:
        return _LinearSiLU.apply(x, weight, bias, compact)
    return F.silu(F.linear(x, weight, bias))


@contextmanager
def fused_linear_silu_layers(model, *, backbone=False, compact=False, scaled=False):
    originals = []
    try:
        for name, parent in model.named_modules():
            if not isinstance(parent, torch.nn.Sequential):
                continue
            if not name.startswith("head.") and not (backbone and name.startswith("base.model.")):
                continue
            for layer, activation in pairwise(list(parent.children())):
                if not isinstance(layer, torch.nn.Linear) or not isinstance(
                    activation, torch.nn.SiLU
                ):
                    continue
                originals.append((layer, layer.forward))
                originals.append((activation, activation.forward))

                def forward(self, x):
                    if scaled and x.numel() // x.shape[-1] >= 16384:
                        from hip_orb.scaled_silu import scaled_linear_silu

                        return scaled_linear_silu(x, self.weight, self.bias, compact=compact)
                    return linear_silu(x, self.weight, self.bias, compact=compact)

                layer.forward = MethodType(forward, layer)
                activation.forward = lambda x: x
        yield
    finally:
        for module, forward in reversed(originals):
            module.forward = forward
