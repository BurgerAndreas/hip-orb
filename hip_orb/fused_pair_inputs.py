"""One output write per feature; inputs are detached frozen-backbone tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_join(Nodes, Edges, Charges, I, J, Forward, Reverse, Delta, Distance, Output,
                 pairs, node_stride, edge_stride, charge_stride, delta_stride,
                 D: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    row = (tl.program_id(0) * ROWS + tl.arange(0, ROWS))[:, None]
    valid = row < 2*pairs
    columns = tl.arange(0, BLOCK)[None, :]
    backwards = row >= pairs
    pair = row % pairs
    i, j = tl.load(I + pair, valid, other=0), tl.load(J + pair, valid, other=0)
    a, b = tl.where(backwards, j, i), tl.where(backwards, i, j)
    f, r = tl.load(Forward + pair, valid, other=-1), tl.load(Reverse + pair, valid, other=-1)
    edge = tl.where(backwards, r, f)
    first = tl.load(Nodes + a * node_stride + columns, valid & (columns < D), other=0)
    second = tl.load(Nodes + b * node_stride + columns-D,
                     valid & (columns >= D) & (columns < 2*D), other=0)
    latent = tl.load(Edges + tl.maximum(edge, 0) * edge_stride + columns-2*D,
                     valid & (columns >= 2*D) & (columns < 3*D) & (edge >= 0), other=0)
    distance = tl.load(Distance + pair, valid, other=1.)
    direction = tl.load(Delta + pair * delta_stride + columns-3*D,
                        valid & (columns >= 3*D) & (columns < 3*D+3), other=0)
    direction = direction / distance * tl.where(backwards, -1., 1.)
    radial = tl.where(columns == 3*D+3, tl.log(1.+distance), 0.)
    charge_a = tl.where(columns == 3*D+4, tl.load(Charges + a*charge_stride, valid, other=0.), 0.)
    charge_b = tl.where(columns == 3*D+5, tl.load(Charges + b*charge_stride, valid, other=0.), 0.)
    present = tl.where(columns == 3*D+6, (edge >= 0).to(tl.float32), 0.)
    joined = first + second + latent + direction + radial + charge_a + charge_b + present
    tl.store(Output + row * WIDTH + columns, joined, valid & (columns < WIDTH))


def fused_directed_pair_inputs(nodes, edges, charges, i, j, forward, reverse, delta, distance):
    tensors = (nodes, edges, charges, delta, distance)
    if any(x.requires_grad for x in tensors):
        raise ValueError('Fused pair inputs require detached frozen features')
    if nodes.dtype != torch.float32 or any(x.dtype != nodes.dtype for x in tensors):
        raise ValueError('Fused pair inputs require float32 features')
    if any(x.stride(-1) != 1 for x in (nodes, edges, delta, distance)):
        raise ValueError('Fused pair inputs require contiguous feature columns')
    p, d = i.numel(), nodes.shape[1]
    width = 3*d+7
    output = nodes.new_empty((2*p, width))
    if p:
        _gather_join[(triton.cdiv(2*p, 4),)](
            nodes, edges, charges, i, j, forward, reverse, delta, distance, output,
            p, nodes.stride(0), edges.stride(0), charges.stride(0), delta.stride(0),
            D=d, WIDTH=width, BLOCK=triton.next_power_of_2(width), ROWS=4, num_warps=8)
    return output
