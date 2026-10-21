"""Pack both dense pair directions once and reduce feature gradients without atomics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    NODES,
    EDGES,
    CHARGES,
    PAIR_I,
    J,
    FWD,
    REV,
    UNIT,
    R1,
    R2,
    OUT,
    P,
    D: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = t // KP, t % KP
    valid = row < 2 * P
    backwards = row >= P
    pair = tl.where(backwards, row - P, row)
    i = tl.load(PAIR_I + pair, valid, 0)
    j = tl.load(J + pair, valid, 0)
    a, b = tl.where(backwards, j, i), tl.where(backwards, i, j)
    f = tl.load(FWD + pair, valid, 0)
    r = tl.load(REV + pair, valid, 0)
    edge = tl.where(backwards, r, f)
    na = tl.load(NODES + a * D + col, valid & (col < D), 0)
    nb = tl.load(NODES + b * D + col - D, valid & (col >= D) & (col < 2 * D), 0)
    ev = tl.load(EDGES + edge * D + col - 2 * D, valid & (col >= 2 * D) & (col < 3 * D), 0)
    uv = tl.load(UNIT + pair * 3 + col - 3 * D, valid & (col >= 3 * D) & (col < 3 * D + 3), 0)
    uv *= tl.where(backwards, -1.0, 1.0)
    rc = col - 3 * D - 3
    r1 = tl.load(R1 + pair * R + rc, valid & ~backwards & (rc >= 0) & (rc < R), 0)
    r2 = tl.load(R2 + pair * R + rc, valid & backwards & (rc >= 0) & (rc < R), 0)
    qa = tl.load(CHARGES + a, valid & (col == 3 * D + 3 + R), 0)
    qb = tl.load(CHARGES + b, valid & (col == 3 * D + 4 + R), 0)
    present = (col == K - 1).to(tl.float32)
    tl.store(OUT + t, na + nb + ev + uv + r1 + r2 + qa + qb + present, valid)


@triton.jit
def _edge_radial_backward(
    G,
    FWD,
    REV,
    EDGE_GRAD,
    RAD_GRAD,
    P,
    D: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    GS: tl.constexpr,
    NEED_EDGE: tl.constexpr,
    NEED_RAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = t // C, t % C
    valid = row < 2 * P
    if NEED_EDGE:
        backwards = row >= P
        pair = tl.where(backwards, row - P, row)
        f = tl.load(FWD + pair, valid, 0)
        r = tl.load(REV + pair, valid, 0)
        edge = tl.where(backwards, r, f)
        value = tl.load(G + row * GS + 2 * D + col, valid & (col < D), 0)
        tl.store(EDGE_GRAD + edge * D + col, value, valid & (col < D))
    if NEED_RAD:
        radial = tl.load(G + row * GS + 3 * D + 3 + col, valid & (col < R), 0)
        tl.store(RAD_GRAD + row * R + col, radial, valid & (col < R))


@triton.jit
def _node_backward(
    G,
    ORIGIN,
    EDGE_ORIGIN,
    WIDTH,
    OUT,
    P,
    D: tl.constexpr,
    GS: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    COLS: tl.constexpr,
):
    node = tl.program_id(0)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    neighbor = tl.arange(0, NEIGHBORS)
    origin = tl.load(ORIGIN + node)
    n = tl.load(WIDTH + node)
    base = tl.load(EDGE_ORIGIN + node) // 2
    local = node - origin
    a, b = tl.minimum(local, neighbor), tl.maximum(local, neighbor)
    pair = base + a * (2 * n - a - 1) // 2 + b - a - 1
    valid = (neighbor < n) & (neighbor != local)
    # Broadcast the neighbor dimension explicitly for column coalescing.
    first_col = tl.where((local < neighbor)[:, None], col[None, :], D + col[None, :])
    second_col = tl.where((local < neighbor)[:, None], D + col[None, :], col[None, :])
    mask = valid[:, None] & (col[None, :] < D)
    a_grad = tl.load(G + pair[:, None] * GS + first_col, mask, 0)
    b_grad = tl.load(G + (pair[:, None] + P) * GS + second_col, mask, 0)
    value = tl.sum(a_grad + b_grad, 0)
    tl.store(OUT + node * D + col, value, col < D)


def native_inputs(nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2):
    def one(a, b, ids, direction, radial):
        return torch.cat(
            (
                nodes[a],
                nodes[b],
                edges[ids],
                direction,
                radial,
                charges[a, None],
                charges[b, None],
                nodes.new_ones((len(a), 1)),
            ),
            -1,
        )

    return torch.cat((one(i, j, forward, unit, radial1), one(j, i, reverse, -unit, radial2)))


class _DenseInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2, sizes):
        p, d, r = len(i), nodes.shape[1], radial1.shape[1]
        k = 3 * d + 6 + r
        kp = triton.cdiv(k, 16) * 16
        out = nodes.new_empty((2 * p, kp))
        if p:
            _pack[(triton.cdiv(out.numel(), 1024),)](
                nodes,
                edges,
                charges,
                i,
                j,
                forward,
                reverse,
                unit,
                radial1,
                radial2,
                out,
                p,
                d,
                r,
                k,
                kp,
                1024,
            )
        ctx.save_for_backward(nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2)
        ctx.sizes = tuple(sizes)
        return out[:, :k]

    @staticmethod
    def backward(ctx, gradient):
        nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2 = ctx.saved_tensors
        if torch.is_grad_enabled():
            with torch.enable_grad():
                vals = [v.view_as(v) for v in (nodes, edges, radial1, radial2)]
                out = native_inputs(
                    vals[0], vals[1], charges, i, j, forward, reverse, unit, vals[2], vals[3]
                )
                needs = [ctx.needs_input_grad[k] for k in (0, 1, 8, 9)]
                inputs = [v for v, need in zip(vals, needs, strict=True) if need]
                values = iter(torch.autograd.grad(out, inputs, gradient, create_graph=True))
                grads = [next(values) if need else None for need in needs]
            return (grads[0], grads[1]) + (None,) * 6 + (grads[2], grads[3], None)
        from hip_orb.batched_pairs import dense_node_metadata

        if gradient.stride(-1) != 1:
            gradient = gradient.contiguous()
        p, d, r = len(i), nodes.shape[1], radial1.shape[1]
        dn = torch.empty_like(nodes) if ctx.needs_input_grad[0] else None
        de = torch.empty_like(edges) if ctx.needs_input_grad[1] else None
        need_r = ctx.needs_input_grad[8] or ctx.needs_input_grad[9]
        dr = radial1.new_empty((2 * p, r)) if need_r else None
        if p and (de is not None or need_r):
            _edge_radial_backward[(triton.cdiv(2 * p * max(d, r), 1024),)](
                gradient,
                forward,
                reverse,
                de,
                dr,
                p,
                d,
                r,
                max(d, r),
                gradient.stride(0),
                de is not None,
                need_r,
                1024,
            )
        if dn is not None:
            origin, edge_origin, width = dense_node_metadata(ctx.sizes, str(nodes.device))
            _node_backward[(len(nodes), triton.cdiv(d, 32))](
                gradient,
                origin,
                edge_origin,
                width,
                dn,
                p,
                d,
                gradient.stride(0),
                triton.next_power_of_2(max(ctx.sizes)),
                32,
                num_warps=8,
            )
        dr1 = dr[:p] if ctx.needs_input_grad[8] else None
        dr2 = dr[p:] if ctx.needs_input_grad[9] else None
        return (dn, de) + (None,) * 6 + (dr1, dr2, None)


def dense_pair_inputs(nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2, sizes):
    """Require canonical complete dense edges and frozen geometry and charges."""
    tensors = (nodes, edges, charges, i, j, forward, reverse, unit, radial1, radial2)
    if (
        nodes.is_cuda
        and nodes.dtype == torch.float32
        and not charges.requires_grad
        and not unit.requires_grad
        and len(edges) == 2 * len(i)
        and all(t.is_contiguous() for t in tensors)
        and all(t.dtype == torch.float32 for t in (edges, charges, unit, radial1, radial2))
    ):
        result = _DenseInputs.apply(*tensors, tuple(sizes))
        result._hip_zero_padded = True
        return result
    return native_inputs(*tensors)
