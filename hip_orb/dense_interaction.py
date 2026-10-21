"""Dense molecular message packing and deterministic gated neighbor sums."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(N, E, S, R, Y, M, D: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = ids // (3 * D), ids % (3 * D)
    valid = row < M
    s = tl.load(S + row, valid, 0)
    r = tl.load(R + row, valid, 0)
    e = tl.load(E + row * D + col, valid & (col < D), 0)
    a = tl.load(N + s * D + col - D, valid & (col >= D) & (col < 2 * D), 0)
    b = tl.load(N + r * D + col - 2 * D, valid & (col >= 2 * D), 0)
    tl.store(Y + ids, e + a + b, valid)


@triton.jit
def _node_grad(
    G, ORIGIN, START, WIDTH, DN, D: tl.constexpr, NEIGHBORS: tl.constexpr, COLS: tl.constexpr
):
    node = tl.program_id(0)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    neighbor = tl.arange(0, NEIGHBORS)
    origin = tl.load(ORIGIN + node)
    start = tl.load(START + node)
    n = tl.load(WIDTH + node)
    local = node - origin
    outgoing = start + local * (n - 1) + neighbor - (neighbor > local).to(tl.int32)
    incoming = start + neighbor * (n - 1) + local - (local > neighbor).to(tl.int32)
    mask = (neighbor[:, None] < n) & (neighbor[:, None] != local) & (col[None, :] < D)
    a = tl.load(G + outgoing[:, None] * 3 * D + D + col[None, :], mask, 0)
    b = tl.load(G + incoming[:, None] * 3 * D + 2 * D + col[None, :], mask, 0)
    tl.store(DN + node * D + col, tl.sum(a + b, 0), col < D)


@triton.jit
def _pool(
    E,
    SA,
    RA,
    ORIGIN,
    START,
    WIDTH,
    SENT,
    RECEIVED,
    D: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    COLS: tl.constexpr,
):
    node = tl.program_id(0)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    neighbor = tl.arange(0, NEIGHBORS)
    origin = tl.load(ORIGIN + node)
    start = tl.load(START + node)
    n = tl.load(WIDTH + node)
    local = node - origin
    outgoing = start + local * (n - 1) + neighbor - (neighbor > local).to(tl.int32)
    incoming = start + neighbor * (n - 1) + local - (local > neighbor).to(tl.int32)
    valid = (neighbor < n) & (neighbor != local)
    mask = valid[:, None] & (col[None, :] < D)
    a = tl.load(E + outgoing[:, None] * D + col[None, :], mask, 0)
    b = tl.load(E + incoming[:, None] * D + col[None, :], mask, 0)
    sa = tl.load(SA + outgoing, valid, 0)
    ra = tl.load(RA + incoming, valid, 0)
    tl.store(SENT + node * D + col, tl.sum(a * sa[:, None], 0), col < D)
    tl.store(RECEIVED + node * D + col, tl.sum(b * ra[:, None], 0), col < D)


@triton.jit
def _pool_grad(
    E,
    SA,
    RA,
    S,
    R,
    GS,
    GR,
    DE,
    DS,
    DR,
    M,
    D: tl.constexpr,
    B: tl.constexpr,
    NE: tl.constexpr,
    NS: tl.constexpr,
    NR: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    s = tl.load(S + row)
    r = tl.load(R + row)
    gs = tl.load(GS + s * D + col, col < D, 0)
    gr = tl.load(GR + r * D + col, col < D, 0)
    if NE:
        sa = tl.load(SA + row)
        ra = tl.load(RA + row)
        tl.store(DE + row * D + col, gs * sa + gr * ra, col < D)
    if NS or NR:
        e = tl.load(E + row * D + col, col < D, 0)
        if NS:
            tl.store(DS + row, tl.sum(gs * e, 0))
        if NR:
            tl.store(DR + row, tl.sum(gr * e, 0))


def _metadata(sizes, device):
    from hip_orb.batched_pairs import dense_node_metadata

    return dense_node_metadata(tuple(sizes), str(device))


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, edges, s, r, sizes):
        d = nodes.shape[1]
        out = nodes.new_empty((len(edges), 3 * d))
        if out.numel():
            _pack[(triton.cdiv(out.numel(), 1024),)](nodes, edges, s, r, out, len(edges), d, 1024)
        ctx.save_for_backward(nodes, edges, s, r)
        ctx.sizes = sizes
        return out

    @staticmethod
    def backward(ctx, g):
        n, e, s, r = ctx.saved_tensors
        d = n.shape[1]
        g = g.contiguous()
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = torch.cat((e, n[s], n[r]), -1)
                inputs = [
                    p for p, need in zip((n, e), ctx.needs_input_grad[:2], strict=True) if need
                ]
                grads = iter(torch.autograd.grad(y, inputs, g, create_graph=True))
            return tuple(next(grads) if need else None for need in ctx.needs_input_grad[:2]) + (
                None,
                None,
                None,
            )
        dn = None
        if ctx.needs_input_grad[0]:
            dn = torch.empty_like(n)
            origin, start, width = _metadata(ctx.sizes, n.device)
            if len(n):
                _node_grad[(len(n), triton.cdiv(d, 32))](
                    g,
                    origin,
                    start,
                    width,
                    dn,
                    d,
                    triton.next_power_of_2(max(ctx.sizes)),
                    32,
                    num_warps=8,
                )
        de = g[:, :d].contiguous() if ctx.needs_input_grad[1] else None
        return dn, de, None, None, None


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, sa, ra, s, r, sizes):
        count = sum(sizes)
        d = edges.shape[1]
        sent = edges.new_empty((count, d))
        received = torch.empty_like(sent)
        origin, start, width = _metadata(sizes, edges.device)
        if count:
            _pool[(count, triton.cdiv(d, 32))](
                edges,
                sa,
                ra,
                origin,
                start,
                width,
                sent,
                received,
                d,
                triton.next_power_of_2(max(sizes)),
                32,
                num_warps=8,
            )
        ctx.save_for_backward(edges, sa, ra, s, r)
        ctx.sizes = sizes
        return sent, received

    @staticmethod
    def backward(ctx, gs, gr):
        e, sa, ra, s, r = ctx.saved_tensors
        gs = gs.contiguous()
        gr = gr.contiguous()
        needs = ctx.needs_input_grad[:3]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                count = sum(ctx.sizes)
                sent = e.new_zeros((count, e.shape[1])).index_add(0, s, e * sa)
                received = e.new_zeros((count, e.shape[1])).index_add(0, r, e * ra)
                inputs = [p for p, need in zip((e, sa, ra), needs, strict=True) if need]
                values = iter(
                    torch.autograd.grad((sent, received), inputs, (gs, gr), create_graph=True)
                )
            return tuple(next(values) if need else None for need in needs) + (None, None, None)
        de = torch.empty_like(e) if needs[0] else None
        ds = torch.empty_like(sa) if needs[1] else None
        dr = torch.empty_like(ra) if needs[2] else None
        if len(e):
            _pool_grad[(len(e),)](
                e,
                sa,
                ra,
                s,
                r,
                gs,
                gr,
                de,
                ds,
                dr,
                len(e),
                e.shape[1],
                triton.next_power_of_2(e.shape[1]),
                *needs,
            )
        return de, ds, dr, None, None, None


def packed_messages(nodes, edges, s, r, sizes):
    return _Pack.apply(nodes, edges, s, r, tuple(sizes))


def pooled_messages(edges, sa, ra, s, r, sizes):
    return _Pool.apply(edges, sa, ra, s, r, tuple(sizes))


def dense_interaction(
    layer, nodes, edges, s, r, cutoff, *, cond_nodes=None, cond_edges=None, sizes
):
    if (
        not nodes.is_cuda
        or nodes.dtype != torch.float32
        or edges.dtype != torch.float32
        or layer._attention_gate != "sigmoid"
        or layer._node_cond == "concatenative"
        or layer._edge_cond == "concatenative"
    ):
        return layer(nodes, edges, s, r, cutoff, cond_nodes=cond_nodes, cond_edges=cond_edges)
    if layer._node_cond == "additive" and cond_nodes is not None:
        nodes = nodes + layer._cond_node_proj(cond_nodes)
    if layer._edge_cond == "additive" and cond_edges is not None:
        edges = edges + layer._cond_edge_proj(cond_edges)
    ra = layer._receive_attn(edges).sigmoid()
    sa = layer._send_attn(edges).sigmoid()
    if layer._distance_cutoff:
        ra = ra * cutoff
        sa = sa * cutoff
    updated_edges = layer._edge_mlp(packed_messages(nodes, edges, s, r, sizes))
    sent, received = pooled_messages(updated_edges, sa, ra, s, r, sizes)
    updated_nodes = layer._node_mlp(torch.cat((nodes, received, sent), -1))
    return nodes + updated_nodes, edges + updated_edges
