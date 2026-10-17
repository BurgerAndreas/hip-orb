"""Opt-in precision experiments with FP32 master parameters and residual paths.

FP8 uses actual Hopper scaled matrix multiplications, not simulated casts.
Its custom backward implements a straight-through quantization surrogate.
Selected policies also store internal MLP activations in FP16.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import pairwise
from types import MethodType

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LinearPolicy:
    dtype: str = "fp32"
    backward: str = "bf16"
    output_dtype: str = "fp32"
    selection: str = "all"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    fuse_silu: bool = False
    activation_dtype: str = "fp32"
    input_alignment: int = 1


def quantize_fp8(value, dtype=torch.float8_e4m3fn):
    """Return quantized data and the FP32 dequantization scale."""
    value = value.float()
    scale = (value.abs().amax() / torch.finfo(dtype).max).clamp_min(torch.finfo(torch.float32).tiny)
    return (value / scale).to(dtype), scale.reshape(())


def pad_matrix(value, rows, columns):
    if value.shape == (rows, columns):
        return value.contiguous()
    return F.pad(value, (0, columns - value.shape[1], 0, rows - value.shape[0])).contiguous()


def scaled_mm(a, b, scale_a, scale_b):
    """B must have column-major strides for the native CUDA FP8 kernel."""
    return torch._scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32, use_fast_accum=False
    )


class _HalfLinearFloatOutput(torch.autograd.Function):
    """Half inputs and FP32 GEMM outputs, including both backward GEMMs."""

    @staticmethod
    def forward(ctx, x, weight, bias, dtype, cached_weight):
        ctx.shape = x.shape
        ctx.dtypes = (x.dtype, weight.dtype, None if bias is None else bias.dtype)
        a = x.reshape(-1, x.shape[-1]).to(dtype)
        w = weight.to(dtype) if cached_weight is None else cached_weight
        if a.shape[1] != w.shape[1]:
            a = F.pad(a, (0, w.shape[1] - a.shape[1]))
        y = torch.mm(a, w.T, out_dtype=torch.float32)
        if bias is not None:
            y = y + bias.float()
        ctx.save_for_backward(a, w)
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("Half-input FP32-output GEMMs support first-order gradients only")
        a, w = ctx.saved_tensors
        g = grad.reshape(-1, w.shape[0]).to(a.dtype)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = torch.mm(g, w, out_dtype=torch.float32)[:, :ctx.shape[-1]].reshape(ctx.shape).to(ctx.dtypes[0])
        if ctx.needs_input_grad[1]:
            dw = torch.mm(g.T, a, out_dtype=torch.float32)[:, :ctx.shape[-1]].contiguous().to(ctx.dtypes[1])
        if ctx.needs_input_grad[2]:
            db = grad.reshape(-1, w.shape[0]).float().sum(0).to(ctx.dtypes[2])
        return dx, dw, db, None, None


def cached_half_weight(weight, dtype, cache, alignment=1):
    if alignment < 1:
        raise ValueError("Input alignment must be positive")
    if cache is None and alignment == 1:
        return None
    if cache is None:
        cache = {}
    key = (weight.data_ptr(), weight._version, dtype, alignment)
    if cache.get("key") != key:
        cache.clear()
        half = weight.detach().to(dtype)
        padding = (-weight.shape[1]) % alignment
        if padding:
            half = F.pad(half, (0, padding))
        cache.update(key=key, weight=half)
    return cache["weight"]


def half_linear_float_output(x, weight, bias=None, *, dtype=torch.bfloat16, cache=None, alignment=1):
    return _HalfLinearFloatOutput.apply(
        x, weight, bias, dtype, cached_half_weight(weight, dtype, cache, alignment)
    )


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, qweight, weight_scale, backward):
        rows, inputs = x.shape
        outputs = weight.shape[0]
        m, k = ((v + 15) // 16 * 16 for v in (rows, inputs))
        qx, sx = quantize_fp8(pad_matrix(x, m, k))
        y = scaled_mm(qx, qweight.T, sx, weight_scale)[:rows, :outputs]
        if bias is not None:
            y = y + bias.float()
        ctx.shape = (rows, inputs, outputs)
        ctx.dtypes = (x.dtype, weight.dtype, None if bias is None else bias.dtype)
        ctx.backward_precision = backward
        if any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(qx, qweight, sx, weight_scale)
        return y

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("Native FP8 supports first-order gradients only")
        qx, qw, sx, sw = ctx.saved_tensors
        rows, inputs, outputs = ctx.shape
        m, _ = qx.shape
        n = qw.shape[0]
        g = pad_matrix(grad.float(), m, n)
        dx = dw = db = None
        if ctx.backward_precision == "bf16":
            # Backward uses the quantized forward operands, dequantized to BF16.
            x = (qx.float() * sx).to(torch.bfloat16)
            w = (qw.float() * sw).to(torch.bfloat16)
            g = g.to(torch.bfloat16)
            if ctx.needs_input_grad[0]:
                dx = (g @ w)[:rows, :inputs].to(ctx.dtypes[0])
            if ctx.needs_input_grad[1]:
                dw = (g.T @ x)[:outputs, :inputs].to(ctx.dtypes[1])
        elif ctx.backward_precision in ("fp8_e4m3", "fp8_e5m2"):
            dtype = (
                torch.float8_e5m2 if ctx.backward_precision == "fp8_e5m2" else torch.float8_e4m3fn
            )
            qg, sg = quantize_fp8(g, dtype)
            if ctx.needs_input_grad[0]:
                dx = scaled_mm(qg, qw.T.contiguous().T, sg, sw)[:rows, :inputs].to(ctx.dtypes[0])
            if ctx.needs_input_grad[1]:
                dw = scaled_mm(qg.T.contiguous(), qx.T.contiguous().T, sg, sx)[
                    :outputs, :inputs
                ].to(ctx.dtypes[1])
        else:
            raise ValueError("Invalid FP8 backward precision")
        if ctx.needs_input_grad[2]:
            db = grad.float().sum(0).to(ctx.dtypes[2])
        return dx, dw, db, None, None, None


def fp8_linear(x, weight, bias=None, *, backward="bf16", cache=None):
    if x.device.type != "cuda":
        raise ValueError("Native FP8 linear requires CUDA")
    if not x.numel():
        return F.linear(x, weight, bias)
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    outputs, inputs = weight.shape
    n, k = ((v + 15) // 16 * 16 for v in (outputs, inputs))
    key = (weight._version, weight.data_ptr(), tuple(weight.shape))
    if cache is None or cache.get("key") != key:
        qw, sw = quantize_fp8(pad_matrix(weight.detach(), n, k))
        if cache is not None:
            cache.clear()
            cache.update(key=key, qweight=qw, scale=sw)
    else:
        qw, sw = cache["qweight"], cache["scale"]
    y = _FP8Linear.apply(x, weight, bias, qw, sw, backward)
    return y.reshape(*shape[:-1], outputs)


@contextmanager
def linear_precision(model, *, head=LinearPolicy(), backbone=LinearPolicy(), include_outputs=False):
    """Temporarily replace linear forwards without changing parameters or state keys.

    The backbone selection covers base.model. The charge head, geometry,
    Coulomb correction, pooling, normalization, and losses keep their own dtype.
    """
    changed = []
    previous_caches = getattr(model, "_hip_precision_trainable_caches", None)
    model._hip_precision_trainable_caches = list(previous_caches or ())
    stats = {
        "head_layers": [],
        "backbone_layers": [],
        "calls": {
            key: 0
            for key in (
                "bf16",
                "fp16",
                "fp8",
                "bf16_f32",
                "fp16_f32",
                "fp16_scaled",
                "fp8_row",
                "bf16_compensated",
            )
        },
    }
    try:
        followed_by_silu = {}
        for parent in model.modules():
            if isinstance(parent, torch.nn.Sequential):
                children = list(parent.children())
                for a, b in pairwise(children):
                    if isinstance(a, torch.nn.Linear) and isinstance(b, torch.nn.SiLU):
                        followed_by_silu[id(a)] = b
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            region = (
                "head"
                if name.startswith("head.")
                else "backbone"
                if name.startswith("base.model.")
                else None
            )
            if region is None:
                continue
            policy = head if region == "head" else backbone
            if policy.include and not any(fnmatchcase(name, pat) for pat in policy.include):
                continue
            if any(fnmatchcase(name, pat) for pat in policy.exclude):
                continue
            if policy.selection not in ("all", "square"):
                raise ValueError("Unknown linear selection")
            if policy.selection == "square" and (
                module.in_features != module.out_features or module.in_features < 512
            ):
                continue
            if (
                region == "head"
                and module.out_features == 9
                and not include_outputs
                and head.dtype != "fp32"
                and (head.output_dtype == "bf16" or head.activation_dtype == "fp16")
            ):
                original = module.forward

                def full_output(self, x):
                    return F.linear(x.float(), self.weight, self.bias)

                module.forward = MethodType(full_output, module)
                changed.append((module, original, {}))
                continue
            if policy.dtype == "fp32" or (
                region == "head" and module.out_features == 9 and not include_outputs
            ):
                continue
            if policy.dtype not in stats["calls"]:
                raise ValueError("Unknown linear precision")
            cache = {}
            if module.weight.requires_grad:
                model._hip_precision_trainable_caches.append(cache)
            original = module.forward
            fuse = policy.fuse_silu and id(module) in followed_by_silu
            if fuse:
                if policy.dtype != "fp16_f32":
                    raise ValueError("SiLU fusion requires fp16_f32 linear precision")
                activation = followed_by_silu[id(module)]
                changed.append((activation, activation.forward, {}))
                activation.forward = lambda x: x

            def forward(self, x, policy=policy, cache=cache, fuse=fuse, region=region, name=name):
                stats["calls"][policy.dtype] += 1
                if fuse:
                    from hip_orb.precision_kernels import fused_half_silu

                    return fused_half_silu(
                        x,
                        self.weight,
                        self.bias,
                        half_output=policy.activation_dtype == "fp16",
                        cache=cache,
                        alignment=policy.input_alignment,
                        input_grad_columns=getattr(self, "_hip_input_grad_columns", None),
                    )
                if policy.dtype == "fp8":
                    y = fp8_linear(x, self.weight, self.bias, backward=policy.backward, cache=cache)
                elif policy.dtype in ("fp16_scaled", "fp8_row", "bf16_compensated"):
                    from hip_orb.precision_scaled import scaled_linear

                    y = scaled_linear(x, self.weight, self.bias, mode=policy.dtype, cache=cache)
                elif policy.dtype in ("bf16_f32", "fp16_f32"):
                    y = half_linear_float_output(
                        x,
                        self.weight,
                        self.bias,
                        dtype=torch.bfloat16 if policy.dtype == "bf16_f32" else torch.float16,
                        cache=cache,
                        alignment=policy.input_alignment,
                    )
                else:
                    dtype = torch.bfloat16 if policy.dtype == "bf16" else torch.float16
                    # Cache only frozen weights. Trainable casts retain the native autograd graph.
                    key = (self.weight._version, self.weight.data_ptr(), dtype)
                    if self.weight.requires_grad:
                        w = self.weight.to(dtype)
                    else:
                        if cache.get("key") != key:
                            cache.clear()
                            cache.update(key=key, weight=self.weight.to(dtype))
                        w = cache["weight"]
                    bias = None if self.bias is None else self.bias.to(dtype)
                    y = F.linear(x.to(dtype), w, bias)
                dtype = torch.bfloat16 if policy.output_dtype == "bf16" else torch.float32
                # Orb's NN-2 output enters LayerNorm. Keep that boundary FP32;
                # NN-0 and NN-1 can keep their hidden activations in FP16.
                if (
                    region == "backbone"
                    and policy.activation_dtype == "fp16"
                    and name.endswith(("NN-0", "NN-1"))
                ):
                    dtype = torch.float16
                return y.to(dtype)

            module.forward = MethodType(forward, module)
            changed.append((module, original, cache))
            stats[region + "_layers"].append(name)
        yield stats
    finally:
        for module, original, cache in reversed(changed):
            module.forward = original
            cache.clear()
        if previous_caches is None:
            del model._hip_precision_trainable_caches
        else:
            model._hip_precision_trainable_caches = previous_caches


def invalidate_trainable_precision_caches(model):
    """Invalidate trainable casts after updates, including versionless fused AdamW."""
    for cache in getattr(model, "_hip_precision_trainable_caches", ()):
        cache.clear()
