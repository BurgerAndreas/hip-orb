"""Gather absent message edges as zero without a shared dummy-row gradient."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(Edges, Ids, Output, P, D: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair, column = offset // D, offset % D
    edge = tl.load(Ids + pair, pair < P, other=-1)
    value = tl.load(Edges + edge * D + column, (pair < P) & (edge >= 0), other=0)
    tl.store(Output + offset, value, pair < P)


@triton.jit
def _scatter(Grad, Ids, Output, P, D: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair, column = offset // D, offset % D
    edge = tl.load(Ids + pair, pair < P, other=-1)
    value = tl.load(Grad + offset, pair < P, other=0)
    # Ignore absent edges before accumulation. They cannot share a hot row.
    tl.atomic_add(Output + edge * D + column, value, (pair < P) & (edge >= 0), sem="relaxed")


class _MaskedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, ids):
        output = edges.new_empty((ids.numel(), edges.shape[-1]))
        if output.numel():
            _gather[(triton.cdiv(output.numel(), 1024),)](
                edges, ids, output, ids.numel(), D=edges.shape[-1], BLOCK=1024)
        ctx.save_for_backward(ids)
        ctx.edge_shape = edges.shape
        return output

    @staticmethod
    def backward(ctx, grad):
        ids, = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve reverse-over-reverse differentiation for analysis. The
            # ordinary training backward uses the faster kernel below.
            rows, width = ctx.edge_shape
            padded = grad.new_zeros((rows + 1, width))
            return padded.index_add(0, torch.where(ids >= 0, ids, rows), grad)[:rows], None
        grad = grad.contiguous()
        output = grad.new_zeros(ctx.edge_shape)
        if grad.numel():
            _scatter[(triton.cdiv(grad.numel(), 1024),)](
                grad, ids, output, ids.numel(), D=grad.shape[-1], BLOCK=1024)
        return output, None


def masked_edge_gather(edges, ids):
    """Indices use minus one for absent edges; repeated valid IDs accumulate."""
    from hip_orb.block_layout import require
    if edges.ndim != 2 or ids.ndim != 1 or ids.device != edges.device:
        raise ValueError("Edge features and one-dimensional indices must share a device")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Message-edge indices must be integers")
    require((ids < len(edges)).all(), "Message-edge index exceeds the edge count")
    if (edges.is_cuda and edges.dtype == torch.float32 and edges.is_contiguous()
            and ids.is_contiguous()):
        return _MaskedGather.apply(edges, ids)
    padded = torch.cat((edges, edges.new_zeros((1, edges.shape[-1]))))
    return padded[torch.where(ids >= 0, ids, edges.shape[0])]
