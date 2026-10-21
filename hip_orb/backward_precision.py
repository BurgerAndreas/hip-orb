"""Native FP32 MLP forwards with locally scaled FP16 backward products."""

from contextlib import contextmanager
from functools import lru_cache
from types import MethodType

import torch
from torch.nn import functional as F


@lru_cache(maxsize=8)
def _unit_scale(device):
    return torch.tensor(1024.0, device=device, dtype=torch.float32)


class _BackwardMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, w1, b1, w2, b2, w3, b3, operations, fused_reductions=False, weight_only=False
    ):
        ctx.fused_reductions = fused_reductions
        ctx.weight_only = weight_only
        p1 = operations[0](x)
        h1 = operations[1](p1)
        p2 = operations[2](h1)
        h2 = operations[3](p2)
        y = operations[4](h2)
        ctx.save_for_backward(x, w1, b1, w2, b2, w3, b3, p1, h1, p2, h2)
        return y

    @staticmethod
    def backward(ctx, g):
        x, w1, b1, w2, b2, w3, b3, p1, h1, p2, h2 = ctx.saved_tensors
        needs = ctx.needs_input_grad[:7]
        extra = (None,) * (len(ctx.needs_input_grad) - 7)
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
            return tuple(next(values) if need else None for need in needs) + extra
        if not g.numel():
            params = (x, w1, b1, w2, b2, w3, b3)
            return (
                tuple(
                    torch.zeros_like(p) if need else None
                    for p, need in zip(params, needs, strict=True)
                )
                + extra
            )
        from hip_orb.scaled_mlp import _product, _quantize

        q3, s3, db3 = _quantize(
            g.reshape(-1, len(w3)).contiguous(),
            need_bias=needs[6],
            fused_parts=ctx.fused_reductions,
        )
        dw3 = _product(q3.T, h2.half(), s3) if needs[5] else None
        raw2 = (
            torch.mm(g.reshape(-1, len(w3)), w3)
            if ctx.weight_only
            else torch.mm(q3, w3.half(), out_dtype=torch.float32)
        )
        del q3
        q2, s2, db2 = _quantize(
            raw2, p2, _unit_scale(str(x.device)) if ctx.weight_only else s3, needs[4]
        )
        raw1 = torch.mm(raw2, w2) if ctx.weight_only else None
        del raw2
        dw2 = _product(q2.T, h1.half(), s2) if needs[3] else None
        if not ctx.weight_only:
            raw1 = torch.mm(q2, w2.half(), out_dtype=torch.float32)
        del q2
        q1, s1, db1 = _quantize(
            raw1, p1, _unit_scale(str(x.device)) if ctx.weight_only else s2, needs[2]
        )
        dx = torch.mm(raw1, w1).reshape(x.shape) if ctx.weight_only and needs[0] else None
        del raw1
        width = x.shape[-1]
        pad = (-width) % 16
        a = x.reshape(-1, width).half()
        if pad:
            a = F.pad(a, (0, pad))
        dw1 = _product(q1.T, a, s1)[:, :width].contiguous() if needs[1] else None
        if needs[0] and not ctx.weight_only:
            wp = F.pad(w1.half(), (0, pad)) if pad else w1.half()
            dx = _product(q1, wp, s1)[:, :width].reshape(x.shape)
        return (dx, dw1, db1, dw2, db2, dw3, db3) + extra


@contextmanager
def backward_mlp_precision(
    model, *, minimum_rows=16384, maximum_rows=1000000, fused_reductions=True, weight_only=False
):
    changed = []
    try:
        for name, module in model.named_modules():
            if not name.startswith("head.") or not isinstance(module, torch.nn.Sequential):
                continue
            layers = list(module.children())
            if len(layers) == 6 and isinstance(layers[-1], torch.nn.Identity):
                layers = layers[:-1]
            if (
                len(layers) != 5
                or not all(isinstance(layers[i], torch.nn.Linear) for i in (0, 2, 4))
                or not all(isinstance(layers[i], torch.nn.SiLU) for i in (1, 3))
                or layers[4].out_features < 128
            ):
                continue
            original = module.forward
            operations = tuple(layer.forward for layer in layers)
            a, b, c = layers[0], layers[2], layers[4]
            changed.append((module, original))

            def forward(self, x, a=a, b=b, c=c, original=original, operations=operations):
                rows = x.numel() // x.shape[-1]
                if (
                    not x.is_cuda
                    or x.dtype != torch.float32
                    or x.ndim != 2
                    or not minimum_rows <= rows <= maximum_rows
                ):
                    return original(x)
                return _BackwardMLP.apply(
                    x,
                    a.weight,
                    a.bias,
                    b.weight,
                    b.bias,
                    c.weight,
                    c.bias,
                    operations,
                    fused_reductions,
                    weight_only,
                )

            module.forward = MethodType(forward, module)
        yield
    finally:
        for module, original in reversed(changed):
            module.forward = original
