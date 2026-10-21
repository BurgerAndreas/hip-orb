"""Fused row reduction, scaling, padding, and low-precision conversion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    X,
    Q,
    S,
    M,
    K: tl.constexpr,
    MP,
    KP: tl.constexpr,
    HALF: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, BK)
    v = tl.load(
        X + rows[:, None] * K + cols[None, :], (rows[:, None] < M) & (cols[None, :] < K), 0
    ).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(v), 1), 1e-30)
    if HALF:
        scale = tl.exp2(tl.ceil(tl.log2(maximum / 1024.0)))
    else:
        scale = maximum / 448.0
    # Padded rows have a benign scale and zero data.
    scale = tl.where(rows < M, scale, 1.0)
    tl.store(S + rows, scale, rows < MP)
    tl.store(
        Q + rows[:, None] * KP + cols[None, :],
        v / scale[:, None],
        (rows[:, None] < MP) & (cols[None, :] < KP),
    )


def quantize_rows(value, dtype, *, padded=False):
    m, k = value.shape
    mp, kp = (((v + 15) // 16 * 16) if padded else v for v in (m, k))
    q = torch.empty((mp, kp), device=value.device, dtype=dtype)
    scales = torch.empty((mp, 1), device=value.device, dtype=torch.float32)
    if not mp:
        return q, scales
    _quantize_rows[(triton.cdiv(mp, 4),)](
        value,
        q,
        scales,
        m,
        k,
        mp,
        kp,
        dtype == torch.float16,
        triton.next_power_of_2(kp),
        num_warps=4,
    )
    return q, scales


@triton.jit
def _rescale_rows(P, SA, SB, Y, M, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = ids // N, ids % N
    v = tl.load(P + rows * STRIDE + cols, rows < M, 0)
    sa = tl.load(SA + rows, rows < M, 1.0)
    sb = tl.load(SB + cols, rows < M, 1.0)
    tl.store(Y + ids, v * sa * sb, rows < M)


def rescale_rows(product, sa, sb, m, n):
    y = torch.empty((m, n), device=product.device, dtype=torch.float32)
    _rescale_rows[(triton.cdiv(m * n, 1024),)](product, sa, sb, y, m, n, product.stride(0), 1024)
    return y


@triton.jit
def _silu_forward(P, B, Y, SIZE, N: tl.constexpr, BIAS: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(P + ids, ids < SIZE, 0)
    if BIAS:
        z += tl.load(B + ids % N, ids < SIZE, 0)
    y = z * tl.sigmoid(z)
    tl.store(Y + ids, y, ids < SIZE)


@triton.jit
def _silu_backward(
    P,
    B,
    G,
    PARTIAL,
    GH,
    M,
    N: tl.constexpr,
    BIAS: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    RM: tl.constexpr,
    CN: tl.constexpr,
):
    rows = tl.program_id(0) * RM + tl.arange(0, RM)
    cols = tl.program_id(1) * CN + tl.arange(0, CN)
    ids = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    z = tl.load(P + ids, mask, 0).to(tl.float32)
    if BIAS:
        z += tl.load(B + cols, cols < N, 0)[None, :]
    g = tl.load(G + ids, mask, 0).to(tl.float32)
    s = tl.sigmoid(z)
    dz = g * s * (1.0 + z * (1.0 - s))
    tl.store(GH + ids, dz, mask)
    if WRITE_BIAS:
        partial = tl.sum(dz, 0)
        tl.store(PARTIAL + tl.program_id(0) * N + cols, partial, cols < N)


class _FusedHalfSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, half_output, cached_weight, input_grad_columns, fused_input_cast, radial_source):
        ctx.shape = x.shape
        ctx.input_grad_columns = input_grad_columns
        ctx.radial_dtype = None if radial_source is None else radial_source.dtype
        w = weight.half() if cached_weight is None else cached_weight
        if fused_input_cast:
            from hip_orb.head_fusion import cast_half_padded
            a = cast_half_padded(x.reshape(-1, x.shape[-1]), w.shape[1])
        else:
            a = x.reshape(-1, x.shape[-1]).half()
            if a.shape[1] != w.shape[1]:
                a = torch.nn.functional.pad(a, (0, w.shape[1] - a.shape[1]))
        pre = torch.mm(a, w.T, out_dtype=torch.float32)
        y = torch.empty_like(pre, dtype=torch.float16 if half_output else torch.float32)
        _silu_forward[(triton.cdiv(y.numel(), 1024),)](
            pre, bias, y, y.numel(), y.shape[1], bias is not None, 1024,
        )
        ctx.save_for_backward(a, w, pre, bias)
        ctx.dtypes = x.dtype, weight.dtype, None if bias is None else bias.dtype
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("Fused half SiLU supports first-order gradients only")
        a, w, pre, bias = ctx.saved_tensors
        g = grad.contiguous().reshape_as(pre)
        partial = torch.empty(
            (triton.cdiv(g.shape[0], 32), g.shape[1]) if ctx.needs_input_grad[2] else (1,),
            device=g.device,
        )
        gh = torch.empty_like(pre, dtype=torch.float16)
        _silu_backward[(triton.cdiv(g.shape[0], 32), triton.cdiv(g.shape[1], 128))](
            pre,
            bias,
            g,
            partial,
            gh,
            g.shape[0],
            g.shape[1],
            bias is not None,
            ctx.needs_input_grad[2],
            32,
            128,
        )
        dx = dradial = None
        if ctx.needs_input_grad[7]:
            lo, hi = ctx.input_grad_columns
            selected = torch.nn.functional.pad(w[:, lo:hi], (0, (-(hi - lo)) % 16))
            dradial = torch.mm(gh, selected, out_dtype=torch.float32)[:, :hi-lo].to(ctx.radial_dtype)
        if ctx.needs_input_grad[0]:
            if ctx.input_grad_columns is None:
                dx = torch.mm(gh, w, out_dtype=torch.float32)[:, :ctx.shape[-1]].reshape(ctx.shape).to(ctx.dtypes[0])
            else:
                lo, hi = ctx.input_grad_columns
                # Other channels originate from frozen features and cannot receive a gradient.
                selected = torch.nn.functional.pad(w[:, lo:hi], (0, (-(hi - lo)) % 16))
                small = torch.mm(gh, selected, out_dtype=torch.float32)[:, :hi-lo]
                dx = torch.zeros(ctx.shape, device=g.device, dtype=ctx.dtypes[0])
                dx.reshape(-1, ctx.shape[-1])[:, lo:hi] = small.to(ctx.dtypes[0])
        dw = (
            torch.mm(gh.T, a, out_dtype=torch.float32)[:, :ctx.shape[-1]].contiguous().to(ctx.dtypes[1])
            if ctx.needs_input_grad[1]
            else None
        )
        db = partial.sum(0).to(ctx.dtypes[2]) if ctx.needs_input_grad[2] else None
        return dx, dw, db, None, None, None, None, dradial


def fused_half_silu(x, weight, bias=None, *, half_output=False, cache=None, alignment=1, input_grad_columns=None,
                    fused_input_cast=False, radial_source=None):
    """Use a direct radial input only for matching columns of detached x."""
    from hip_orb.mixed_precision import cached_half_weight

    if radial_source is not None and (
            input_grad_columns is None or x.requires_grad
            or radial_source.shape != (x.numel() // x.shape[-1], input_grad_columns[1] - input_grad_columns[0])):
        raise ValueError("Direct radial backward requires detached inputs and matching radial columns")
    if not x.numel():
        # Preserve a zero connection to the trainable radial input for empty pair sets.
        if radial_source is not None:
            x = x + radial_source.sum() * 0
        y = torch.nn.functional.silu(
            torch.nn.functional.linear(
                x.float(),
                weight.float(),
                None if bias is None else bias.float(),
            )
        )
        return y.half() if half_output else y
    return _FusedHalfSiLU.apply(
        x, weight, bias, half_output, cached_half_weight(weight, torch.float16, cache, alignment),
        input_grad_columns, fused_input_cast, radial_source
    )
