"""FP32 block losses with fused rotations, reductions, and prediction gradients.

The fast path supports the two supervised views used by packed training. It
never rounds predictions, targets, losses, or prediction gradients to FP16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compose_rotations(R1, R2, OUT):
    ids = tl.arange(0, 16)
    row, col = ids // 3, ids % 3
    value = tl.full((16,), 0, tl.float32)
    for k in tl.static_range(3):
        a = tl.load(R2 + row * 3 + k, ids < 9, 0)
        b = tl.load(R1 + col * 3 + k, ids < 9, 0)
        value += a * b
    tl.store(OUT + ids, value, ids < 9)


def compose_rotations(r1, r2):
    """Form R2 R1.T in FP32 even when model GEMMs permit TF32."""
    if (
        r1.is_cuda
        and r1.dtype == r2.dtype == torch.float32
        and not (r1.requires_grad or r2.requires_grad)
    ):
        result = torch.empty_like(r1, memory_format=torch.contiguous_format)
        _compose_rotations[(1,)](
            r1.contiguous(), r2.contiguous(), result, num_warps=1, enable_fp_fusion=False
        )
        return result
    return (
        r2[:, 0, None] * r1[None, :, 0]
        + r2[:, 1, None] * r1[None, :, 1]
        + r2[:, 2, None] * r1[None, :, 2]
    )


def _reference_rotate(value, rotation):
    # Elementwise products keep the reference independent of TF32 GEMM policy.
    left = torch.stack(
        [
            rotation[i, 0] * value[:, 0, :]
            + rotation[i, 1] * value[:, 1, :]
            + rotation[i, 2] * value[:, 2, :]
            for i in range(3)
        ],
        dim=1,
    )
    return torch.stack(
        [
            left[:, :, 0] * rotation[i, 0]
            + left[:, :, 1] * rotation[i, 1]
            + left[:, :, 2] * rotation[i, 2]
            for i in range(3)
        ],
        dim=2,
    )


@triton.jit
def _load_block(P, ids, mask):
    return (
        tl.load(P + ids * 9 + 0, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 1, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 2, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 3, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 4, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 5, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 6, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 7, mask, 0).to(tl.float32),
        tl.load(P + ids * 9 + 8, mask, 0).to(tl.float32),
    )


@triton.jit
def _rotate(a, R, TRANSPOSE: tl.constexpr = False):
    # Explicit 3x3 products avoid batched GEMM dispatch and intermediate arrays.
    r = (
        tl.load(R + 0),
        tl.load(R + (3 if TRANSPOSE else 1)),
        tl.load(R + (6 if TRANSPOSE else 2)),
        tl.load(R + (1 if TRANSPOSE else 3)),
        tl.load(R + 4),
        tl.load(R + (7 if TRANSPOSE else 5)),
        tl.load(R + (2 if TRANSPOSE else 6)),
        tl.load(R + (5 if TRANSPOSE else 7)),
        tl.load(R + 8),
    )
    left = (
        r[0] * a[0] + r[1] * a[3] + r[2] * a[6],
        r[0] * a[1] + r[1] * a[4] + r[2] * a[7],
        r[0] * a[2] + r[1] * a[5] + r[2] * a[8],
        r[3] * a[0] + r[4] * a[3] + r[5] * a[6],
        r[3] * a[1] + r[4] * a[4] + r[5] * a[7],
        r[3] * a[2] + r[4] * a[5] + r[5] * a[8],
        r[6] * a[0] + r[7] * a[3] + r[8] * a[6],
        r[6] * a[1] + r[7] * a[4] + r[8] * a[7],
        r[6] * a[2] + r[7] * a[5] + r[8] * a[8],
    )
    return (
        left[0] * r[0] + left[1] * r[1] + left[2] * r[2],
        left[0] * r[3] + left[1] * r[4] + left[2] * r[5],
        left[0] * r[6] + left[1] * r[7] + left[2] * r[8],
        left[3] * r[0] + left[4] * r[1] + left[5] * r[2],
        left[3] * r[3] + left[4] * r[4] + left[5] * r[5],
        left[3] * r[6] + left[4] * r[7] + left[5] * r[8],
        left[6] * r[0] + left[7] * r[1] + left[8] * r[2],
        left[6] * r[3] + left[7] * r[4] + left[8] * r[5],
        left[6] * r[6] + left[7] * r[7] + left[8] * r[8],
    )


@triton.jit
def _norm(a):
    value = tl.full(a[0].shape, 0, tl.float32)
    for k in tl.static_range(9):
        value += a[k] * a[k]
    return tl.sqrt(value)


@triton.jit
def _errors(A, B, U, IDS, R1, R2, REL, ids, mask, SUPERVISED: tl.constexpr):
    first, second = _load_block(A, ids, mask), _load_block(B, ids, mask)
    transformed = _rotate(first, REL)
    consistency = (
        second[0] - transformed[0],
        second[1] - transformed[1],
        second[2] - transformed[2],
        second[3] - transformed[3],
        second[4] - transformed[4],
        second[5] - transformed[5],
        second[6] - transformed[6],
        second[7] - transformed[7],
        second[8] - transformed[8],
    )
    if SUPERVISED:
        target = (
            tl.load(U + tl.load(IDS + ids * 9 + 0, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 1, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 2, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 3, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 4, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 5, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 6, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 7, mask, 0), mask, 0),
            tl.load(U + tl.load(IDS + ids * 9 + 8, mask, 0), mask, 0),
        )
        t1, t2 = _rotate(target, R1), _rotate(target, R2)
        e1 = (
            first[0] - t1[0],
            first[1] - t1[1],
            first[2] - t1[2],
            first[3] - t1[3],
            first[4] - t1[4],
            first[5] - t1[5],
            first[6] - t1[6],
            first[7] - t1[7],
            first[8] - t1[8],
        )
        e2 = (
            second[0] - t2[0],
            second[1] - t2[1],
            second[2] - t2[2],
            second[3] - t2[3],
            second[4] - t2[4],
            second[5] - t2[5],
            second[6] - t2[6],
            second[7] - t2[7],
            second[8] - t2[8],
        )
    else:
        e1, e2 = consistency, consistency
    return e1, e2, consistency


@triton.jit(do_not_specialize=["N"])
def _loss_forward(
    A,
    B,
    U,
    IDS,
    W,
    R1,
    R2,
    REL,
    OUT,
    N,
    CW: tl.constexpr,
    SUPERVISED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < N
    e1, e2, ec = _errors(A, B, U, IDS, R1, R2, REL, ids, mask, SUPERVISED)
    value = CW * _norm(ec)
    if SUPERVISED:
        value += 0.5 * (_norm(e1) + _norm(e2))
    weight = tl.load(W + ids, mask, 0)
    tl.store(OUT + tl.program_id(0), tl.sum(value / 3.0 * weight, 0))


@triton.jit(do_not_specialize=["N"])
def _loss_backward(
    A,
    B,
    U,
    IDS,
    W,
    R1,
    R2,
    REL,
    UPSTREAM,
    DA,
    DB,
    N,
    CW: tl.constexpr,
    SUPERVISED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < N
    e1, e2, ec = _errors(A, B, U, IDS, R1, R2, REL, ids, mask, SUPERVISED)
    nc = _norm(ec)
    factor = tl.load(UPSTREAM) * tl.load(W + ids, mask, 0) / 3.0
    # Match vector_norm's zero subgradient without dividing by zero.
    sc = tl.where(nc > 0, factor * CW / tl.where(nc > 0, nc, 1.0), 0.0)
    gc = (
        ec[0] * sc,
        ec[1] * sc,
        ec[2] * sc,
        ec[3] * sc,
        ec[4] * sc,
        ec[5] * sc,
        ec[6] * sc,
        ec[7] * sc,
        ec[8] * sc,
    )
    reverse = _rotate(gc, REL, TRANSPOSE=True)
    if SUPERVISED:
        n1, n2 = _norm(e1), _norm(e2)
        s1 = tl.where(n1 > 0, 0.5 * factor / tl.where(n1 > 0, n1, 1.0), 0.0)
        s2 = tl.where(n2 > 0, 0.5 * factor / tl.where(n2 > 0, n2, 1.0), 0.0)
    for k in tl.static_range(9):
        ga, gb = -reverse[k], gc[k]
        if SUPERVISED:
            ga += e1[k] * s1
            gb += e2[k] * s2
        tl.store(DA + ids * 9 + k, ga, mask)
        tl.store(DB + ids * 9 + k, gb, mask)


def reference_loss(a, ad, b, bd, upper, r1, r2, relative, indices, weight):
    """Differentiable fallback, including rotation and target gradients."""

    def terms(first, second, ids, weights):
        consistency = (
            torch.linalg.vector_norm(second - _reference_rotate(first, relative), dim=(-2, -1))
            * weight
        )
        if upper is not None:
            target = upper[ids]
            consistency = consistency + 0.5 * (
                torch.linalg.vector_norm(first - _reference_rotate(target, r1), dim=(-2, -1))
                + torch.linalg.vector_norm(second - _reference_rotate(target, r2), dim=(-2, -1))
            )
        return (consistency / 3 * weights).sum()

    return terms(a, b, indices.get("pair_upper"), indices["pair_weight"]) + terms(
        ad, bd, indices.get("diag_upper"), indices["diag_weight"]
    )


class _BlockLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, ad, b, bd, upper, r1, r2, relative, indices, weight):
        ctx.indices, ctx.weight = indices, weight
        ctx.save_for_backward(a, ad, b, bd, upper, r1, r2, relative)
        groups = ((a, b, "pair"), (ad, bd, "diag"))
        count = sum(triton.cdiv(len(first), 128) for first, _, _ in groups)
        parts = a.new_empty(count)
        offset = 0
        for first, second, name in groups:
            if len(first):
                _loss_forward[(triton.cdiv(len(first), 128),)](
                    first,
                    second,
                    upper,
                    indices.get(name + "_upper"),
                    indices[name + "_weight"],
                    r1,
                    r2,
                    relative,
                    parts[offset:],
                    len(first),
                    weight,
                    upper is not None,
                    128,
                    enable_fp_fusion=False,
                )
                offset += triton.cdiv(len(first), 128)
        return parts.sum()

    @staticmethod
    def backward(ctx, upstream):
        a, ad, b, bd, upper, r1, r2, relative = ctx.saved_tensors
        if torch.is_grad_enabled():
            with torch.enable_grad():
                loss = reference_loss(
                    a, ad, b, bd, upper, r1, r2, relative, ctx.indices, ctx.weight
                )
                inputs = (a, ad, b, bd)
                needed = [
                    x for x, need in zip(inputs, ctx.needs_input_grad[:4], strict=True) if need
                ]
                grads = iter(torch.autograd.grad(loss, needed, upstream, create_graph=True))
            return (
                tuple(next(grads) if need else None for need in ctx.needs_input_grad[:4])
                + (None,) * 6
            )
        gradients = tuple(torch.empty_like(x) for x in (a, ad, b, bd))
        for first, second, da, db, name in (
            (a, b, gradients[0], gradients[2], "pair"),
            (ad, bd, gradients[1], gradients[3], "diag"),
        ):
            if len(first):
                _loss_backward[(triton.cdiv(len(first), 128),)](
                    first,
                    second,
                    upper,
                    ctx.indices.get(name + "_upper"),
                    ctx.indices[name + "_weight"],
                    r1,
                    r2,
                    relative,
                    upstream,
                    da,
                    db,
                    len(first),
                    ctx.weight,
                    upper is not None,
                    128,
                    enable_fp_fusion=False,
                )
        return gradients + (None,) * 6


def fused_block_loss(a, ad, b, bd, upper, r1, r2, relative, indices, weight):
    values = (a, ad, b, bd, upper, r1, r2, relative)
    supported = all(x is None or (x.is_cuda and x.dtype == torch.float32) for x in values)
    supported &= not any(x is not None and x.requires_grad for x in (upper, r1, r2, relative))
    if not supported:
        return reference_loss(a, ad, b, bd, upper, r1, r2, relative, indices, weight)
    values = tuple(x.contiguous() if x is not None else None for x in values)
    return _BlockLoss.apply(*values, indices, float(weight))


def two_view_block_loss(a, ad, b, bd, upper, r1, r2, indices, weight=0.1):
    """Average both supervised losses and add the weighted consistency loss."""
    return fused_block_loss(a, ad, b, bd, upper, r1, r2, compose_rotations(r1, r2), indices, weight)


def consistency_terms(a, ad, b, bd, relative, indices):
    return fused_block_loss(a, ad, b, bd, None, None, None, relative, indices, 1.0)
