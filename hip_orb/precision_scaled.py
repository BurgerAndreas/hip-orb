"""Experimental row scaling and compensated low-precision linear operations."""

import torch


def row_quantize(value, dtype, *, padded=False, fused=True):
    """Scale each row independently. FP16 uses exact powers of two."""
    value = value.contiguous()
    if fused and value.is_cuda and value.shape[1] <= 4096:
        from hip_orb.precision_kernels import quantize_rows

        return quantize_rows(value, dtype, padded=padded)
    m, k = value.shape
    mp, kp = (((v + 15) // 16 * 16) if padded else v for v in (m, k))
    maximum = value.float().abs().amax(1, keepdim=True).clamp_min(1e-30)
    if dtype == torch.float16:
        scale = torch.exp2(torch.ceil(torch.log2(maximum / 1024)))
    else:
        scale = maximum / torch.finfo(dtype).max
    q = (value.float() / scale).to(dtype)
    if padded:
        q = torch.nn.functional.pad(q, (0, kp - k, 0, mp - m))
        scale = torch.nn.functional.pad(scale, (0, 0, 0, mp - m), value=1.0)
    return q, scale.contiguous()


def row_mm(a, bt, dtype=torch.float16):
    qa, sa = row_quantize(a, dtype)
    qb, sb = row_quantize(bt, dtype)
    return torch.mm(qa, qb.T, out_dtype=torch.float32) * sa * sb.T


def compensated_mm(a, bt):
    """Three BF16 products retain the leading correction terms of FP32 inputs."""
    ah, bh = a.bfloat16(), bt.bfloat16()
    al, bl = (a.float() - ah.float()).bfloat16(), (bt.float() - bh.float()).bfloat16()
    y = torch.mm(ah, bh.T, out_dtype=torch.float32)
    y.add_(torch.mm(ah, bl.T, out_dtype=torch.float32))
    y.add_(torch.mm(al, bh.T, out_dtype=torch.float32))
    return y


class _ScaledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mode, cache):
        ctx.shape = x.shape
        ctx.dtypes = (x.dtype, weight.dtype, None if bias is None else bias.dtype)
        ctx.mode = mode
        a = x.reshape(-1, x.shape[-1])
        if mode == "bf16_compensated":
            y = compensated_mm(a, weight)
            if any(ctx.needs_input_grad[:3]):
                ctx.save_for_backward(a, weight)
        else:
            dtype = torch.float16 if mode == "fp16_scaled" else torch.float8_e4m3fn
            padded = mode == "fp8_row"
            qa, sa = row_quantize(a, dtype, padded=padded)
            key = (weight.data_ptr(), weight._version, mode)
            if cache.get("key") != key:
                qw, sw = row_quantize(weight.detach(), dtype, padded=padded)
                cache.update(key=key, qw=qw, sw=sw)
            qw, sw = cache["qw"], cache["sw"]
            if padded:
                # Torch 2.8 row-wise scaled_mm only returns BF16. Use scalar
                # unit scales, then apply row scales without rounding to BF16.
                if "one" not in cache:
                    cache["one"] = torch.ones((), device=x.device)
                product = torch._scaled_mm(
                    qa,
                    qw.T,
                    scale_a=cache["one"],
                    scale_b=cache["one"],
                    out_dtype=torch.float32,
                    use_fast_accum=False,
                )
                from hip_orb.precision_kernels import rescale_rows

                y = rescale_rows(product, sa, sw, len(a), len(weight))
            else:
                y = torch.mm(qa, qw.T, out_dtype=torch.float32) * sa * sw.T
            if any(ctx.needs_input_grad[:3]):
                ctx.save_for_backward(qa, qw, sa, sw)
        if bias is not None:
            y = y + bias.float()
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("Scaled linear operations support first-order gradients only")
        if ctx.mode == "bf16_compensated":
            a, w = ctx.saved_tensors
            mm = compensated_mm
        else:
            qa, qw, sa, sw = ctx.saved_tensors
            a = (qa.float() * sa)[: grad.numel() // grad.shape[-1], : ctx.shape[-1]]
            w = (qw.float() * sw)[: grad.shape[-1], : ctx.shape[-1]]
            mm = row_mm
        g = grad.reshape(-1, w.shape[0]).float()
        dx = mm(g, w.T).reshape(ctx.shape).to(ctx.dtypes[0]) if ctx.needs_input_grad[0] else None
        dw = mm(g.T, a.T).to(ctx.dtypes[1]) if ctx.needs_input_grad[1] else None
        db = g.sum(0).to(ctx.dtypes[2]) if ctx.needs_input_grad[2] else None
        return dx, dw, db, None, None


def scaled_linear(x, weight, bias=None, *, mode="fp16_scaled", cache=None):
    if mode not in ("fp16_scaled", "fp8_row", "bf16_compensated"):
        raise ValueError("Unknown scaled linear mode")
    if not x.numel():
        return torch.nn.functional.linear(
            x.float(), weight.float(), None if bias is None else bias.float()
        )
    return _ScaledLinear.apply(x, weight, bias, mode, {} if cache is None else cache)
