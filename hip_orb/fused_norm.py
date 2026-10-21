"""FP32 RMS normalization with fused row reductions and exact epsilon semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, W, Y, R, M, N: tl.constexpr, EPS: tl.constexpr,
             AFFINE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    y = x * inv
    if AFFINE:
        y *= tl.load(W + col, col < N, 0)
    tl.store(Y + row * N + col, y, col < N)
    tl.store(R + row, inv)


@triton.jit
def _backward(X, W, G, R, DX, PART, M, N: tl.constexpr,
              AFFINE: tl.constexpr, NEED_X: tl.constexpr, NEED_W: tl.constexpr,
              B: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, B)
    valid = (row[:, None] < M) & (col[None, :] < N)
    x = tl.load(X + row[:, None] * N + col[None, :], valid, 0)
    g = tl.load(G + row[:, None] * N + col[None, :], valid, 0)
    r = tl.load(R + row, row < M, 0)
    xhat = x * r[:, None]
    if NEED_W:
        dw = tl.sum(g * xhat, 0)
        tl.store(PART + tl.program_id(0) * N + col, dw, col < N)
    if NEED_X:
        if AFFINE:
            w = tl.load(W + col, col < N, 0)
            g *= w[None, :]
        dot = tl.sum(g * xhat, 1) / N
        dx = r[:, None] * (g - xhat * dot[:, None])
        tl.store(DX + row[:, None] * N + col[None, :], dx, valid)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        y = torch.empty_like(x)
        n, m = x.shape[-1], x.numel() // x.shape[-1]
        inv = x.new_empty(m)
        if m:
            _forward[(m,)](x, weight, y, inv, m, n, eps, weight is not None,
                            triton.next_power_of_2(n))
        ctx.save_for_backward(x, weight, inv)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve higher derivatives for downstream Hessian analysis.
            with torch.enable_grad():
                y = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, ctx.eps)
                inputs = [v for v in (x, weight) if v is not None and v.requires_grad]
                grads = iter(torch.autograd.grad(y, inputs, grad, create_graph=True))
            return (next(grads) if x.requires_grad else None,
                    next(grads) if weight is not None and weight.requires_grad else None, None)
        grad = grad.contiguous()
        n, m = x.shape[-1], x.numel() // x.shape[-1]
        need_x, need_w = ctx.needs_input_grad[:2]
        dx = torch.empty_like(x) if need_x else None
        parts = x.new_empty((triton.cdiv(m, 16), n)) if need_w else None
        if m:
            _backward[(triton.cdiv(m, 16),)](
                x, weight, grad, inv, dx, parts, m, n, weight is not None,
                need_x, need_w, triton.next_power_of_2(n), 16, num_warps=8)
        dw = parts.sum(0) if need_w else None
        return dx, dw, None


def rms_norm(x, shape, weight=None, eps=None):
    """Use FP32 kernels for contiguous CUDA rows; retain the native fallback."""
    if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
            and len(shape) == 1 and x.shape[-1] == shape[0] and 0 < shape[0] <= 4096
            and (weight is None or (weight.is_contiguous() and weight.dtype == x.dtype
                                   and weight.device == x.device))):
        eps = torch.finfo(x.dtype).eps if eps is None else eps
        return _RMSNorm.apply(x, weight, eps)
    return torch.nn.functional.rms_norm(x, shape, weight, eps)
