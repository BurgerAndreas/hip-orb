"""Read atom coordinates once per edge and write its displacement directly."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _reuse_margin(
    P,
    OWNER,
    SIZES,
    OFFSETS,
    UNSAFE,
    RADIUS2: tl.constexpr,
    MAXIMUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    atom = tl.program_id(0)
    owner = tl.load(OWNER + atom)
    n, start = tl.load(SIZES + owner), tl.load(OFFSETS + owner)
    local = tl.arange(0, BLOCK)
    valid = local < n
    x = tl.load(P + (start + local) * 3, valid, 0)
    y = tl.load(P + (start + local) * 3 + 1, valid, 0)
    z = tl.load(P + (start + local) * 3 + 2, valid, 0)
    extent = tl.max(tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z)), 0)
    finite = (tl.abs(x) <= 1e6) & (tl.abs(y) <= 1e6) & (tl.abs(z) <= 1e6)
    bad = tl.min(finite.to(tl.int32), 0) == 0
    # A larger bound covers comparisons in either rotated FP32 frame.
    tolerance = (384.0 * 1.1920928955078125e-7) * (extent + 1.0) * (extent + 1.0)
    dx = x - tl.load(P + atom * 3)
    dy = y - tl.load(P + atom * 3 + 1)
    dz = z - tl.load(P + atom * 3 + 2)
    squared = tl.where(valid & (start + local != atom), dx * dx + dy * dy + dz * dz, float("inf"))
    bad |= tl.min(tl.abs(squared - RADIUS2), 0) <= tolerance
    if MAXIMUM < BLOCK - 1:  # noqa: SIM102 - Remove the sort at compile time when no rank check is needed.
        if n - 1 > MAXIMUM:
            ordered = tl.sort(squared, descending=False)
            boundary = tl.sum(tl.where(local == MAXIMUM - 1, ordered, 0.0), 0)
            following = tl.sum(tl.where(local == MAXIMUM, ordered, 0.0), 0)
            bad |= (boundary < RADIUS2 + tolerance) & (following - boundary <= 2 * tolerance)
    if bad:
        tl.atomic_or(UNSAFE, 1, sem="relaxed")


def reuse_margin_safe(graph, sizes, radius, maximum):
    positions = graph.node_features["positions"]
    counts, offsets = _shape_metadata(tuple(sizes), str(positions.device))
    unsafe = torch.zeros((), device=positions.device, dtype=torch.int32)
    _reuse_margin[(len(positions),)](
        positions.contiguous(),
        graph.node_batch_index.to(positions.device),
        counts,
        offsets,
        unsafe,
        radius * radius,
        maximum,
        triton.next_power_of_2(max(sizes)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return not bool(unsafe.item())


@lru_cache(maxsize=8)
def _shape_metadata(sizes, device):
    offsets = [0]
    for n in sizes[:-1]:
        offsets.append(offsets[-1] + n)
    return (
        torch.tensor(sizes, device=device, dtype=torch.int64),
        torch.tensor(offsets, device=device, dtype=torch.int64),
    )


@triton.jit
def _vectors(P, S, R, V, N: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge, axis = ids // 3, ids % 3
    mask = edge < N
    a = tl.load(S + edge, mask, 0)
    b = tl.load(R + edge, mask, 0)
    value = tl.load(P + b * 3 + axis, mask, 0) - tl.load(P + a * 3 + axis, mask, 0)
    tl.store(V + ids, value, mask)


def edge_vectors(positions, senders, receivers):
    if positions.requires_grad:
        return positions[receivers] - positions[senders]
    output = positions.new_empty((len(senders), 3))
    if output.numel():
        _vectors[(triton.cdiv(output.numel(), 512),)](
            positions.contiguous(),
            senders.contiguous(),
            receivers.contiguous(),
            output,
            len(senders),
            512,
        )
    return output
