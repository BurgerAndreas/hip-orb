"""Optional head fusion candidates; keep native matrix kernels as the default."""

import torch
import triton
import triton.language as tl


@triton.jit
def _cast_pad_half(X, Y, M, K: tl.constexpr, KP: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = ids // KP, ids % KP
    value = tl.load(X + row * S0 + col * S1, (row < M) & (col < K), 0)
    tl.store(Y + ids, value, row < M)


def cast_half_padded(x, columns):
    """Convert and zero-pad a matrix in one pass, including strided inputs."""
    if x.ndim != 2 or columns < x.shape[1]:
        raise ValueError("Require a matrix and a nondecreasing column count")
    if x.dtype == torch.float16 and columns == x.shape[1]:
        return x
    y = torch.empty((x.shape[0], columns), dtype=torch.float16, device=x.device)
    if y.numel():
        _cast_pad_half[(triton.cdiv(y.numel(), 1024),)](
            x, y, x.shape[0], x.shape[1], columns, *x.stride(), 1024)
    return y


@triton.jit
def _pack_pair_inputs(NODES, EDGES, CHARGES, A, B, IDS, DIRECTION, RADIAL, OUT,
                       PAIRS, LATENT: tl.constexpr, RADII: tl.constexpr,
                       WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = ids // WIDTH, ids % WIDTH
    valid = row < PAIRS
    a = tl.load(A + row, valid, 0)
    b = tl.load(B + row, valid, 0)
    edge = tl.load(IDS + row, valid, -1)
    na = tl.load(NODES + a * LATENT + col,
                 valid & (col < LATENT), 0)
    nb = tl.load(NODES + b * LATENT + col - LATENT,
                 valid & (col >= LATENT) & (col < 2 * LATENT), 0)
    ev = tl.load(EDGES + edge * LATENT + col - 2 * LATENT,
                 valid & (edge >= 0) & (col >= 2 * LATENT) & (col < 3 * LATENT), 0)
    uv = tl.load(DIRECTION + row * 3 + col - 3 * LATENT,
                 valid & (col >= 3 * LATENT) & (col < 3 * LATENT + 3), 0)
    rstart = 3 * LATENT + 3
    rv = tl.load(RADIAL + row * RADII + col - rstart,
                 valid & (col >= rstart) & (col < rstart + RADII), 0)
    qa = tl.load(CHARGES + a, valid & (col == rstart + RADII), 0)
    qb = tl.load(CHARGES + b, valid & (col == rstart + RADII + 1), 0)
    present = (col == WIDTH - 1) & (edge >= 0)
    tl.store(OUT + ids, na + nb + ev + uv + rv + qa + qb + present, valid)


class _FrozenPairInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, charges, a, b, ids, direction, radii):
        latent, radial = nodes.shape[1], radii.shape[1]
        width = 3 * latent + 6 + radial
        ctx.start, ctx.radial = 3 * latent + 3, radial
        out = nodes.new_empty((a.numel(), width))
        if out.numel():
            _pack_pair_inputs[(triton.cdiv(out.numel(), 1024),)](
                nodes, edges, charges, a, b, ids, direction, radii, out,
                a.numel(), latent, radial, width, 1024)
        return out

    @staticmethod
    def backward(ctx, grad):
        # Only the radial embedding is trainable on this guarded path.
        return (None,) * 7 + (grad[:, ctx.start:ctx.start + ctx.radial].contiguous(),)


def frozen_pair_inputs(nodes, edges, charges, a, b, ids, direction, radii):
    """Fuse feature gathers and concatenation while retaining radial gradients."""
    frozen = (nodes, edges, charges, direction)
    if any(t.requires_grad for t in frozen):
        raise ValueError("Pair-input fusion requires frozen nonradial features")
    if any(t.dtype != torch.float32 for t in (*frozen, radii)):
        raise ValueError("Pair-input fusion requires FP32 features")
    if any(not t.is_contiguous() for t in (*frozen, a, b, ids, radii)):
        raise ValueError("Pair-input fusion requires contiguous inputs")
    return _FrozenPairInputs.apply(nodes, edges, charges, a, b, ids, direction, radii)
