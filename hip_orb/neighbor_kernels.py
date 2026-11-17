"""CUDA sorted-cell search with bounded candidate tiles and no neighbour cap."""

import itertools

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _tile(
    X,
    Order,
    Starts,
    Counts,
    Left,
    Right,
    Tasks,
    TaskStarts,
    Cumulative,
    OutI,
    OutJ,
    TileCounts,
    Cutoff: tl.constexpr,
    B: tl.constexpr,
    Fill: tl.constexpr,
):
    task = tl.program_id(0)
    pair = tl.load(Tasks + task)
    left, right = tl.load(Left + pair), tl.load(Right + pair)
    ni, nj = tl.load(Counts + left), tl.load(Counts + right)
    local = task - tl.load(TaskStarts + pair)
    columns = tl.cdiv(nj, B)
    lane = tl.arange(0, B * B)
    a = (local // columns) * B + lane // B
    b = (local % columns) * B + lane % B
    i = tl.load(Order + tl.load(Starts + left) + a, a < ni, other=0)
    j = tl.load(Order + tl.load(Starts + right) + b, b < nj, other=0)
    valid = (a < ni) & (b < nj) & ((left != right) | (i < j))
    dx = tl.load(X + 3 * i, valid, other=0) - tl.load(X + 3 * j, valid, other=0)
    dy = tl.load(X + 3 * i + 1, valid, other=0) - tl.load(X + 3 * j + 1, valid, other=0)
    dz = tl.load(X + 3 * i + 2, valid, other=0) - tl.load(X + 3 * j + 2, valid, other=0)
    keep = valid & (libdevice.sqrt_rn((dx * dx + dz * dz) + dy * dy) < Cutoff)
    if Fill:
        offset = tl.load(Cumulative + task)
        slot = offset + tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(OutI + slot, tl.minimum(i, j).to(tl.int32), keep)
        tl.store(OutJ + slot, tl.maximum(i, j).to(tl.int32), keep)
    else:
        tl.store(TileCounts + task, tl.sum(keep.to(tl.int32), 0))


def cell_plan(positions, sizes, cutoff, tile=32):
    """Keep occupied cells only; metadata stays O(N), independent of box volume."""
    # Float64 bin assignment prevents float32 division from skipping a cell.
    width = float(torch.tensor(cutoff, dtype=torch.float32)) * (1 + 8 * 2**-23)
    cells = torch.floor((positions.double() - positions.double().amin(0)) / width).long()
    dims = tuple(int(x) + 1 for x in cells.amax(0).tolist())
    volume = dims[0] * dims[1] * dims[2]
    if volume * max(len(sizes), 1) >= 2**63:
        raise ValueError("Coordinate span exceeds int64 cell keys")
    owner = torch.repeat_interleave(
        torch.arange(len(sizes), device=positions.device),
        torch.tensor(sizes, device=positions.device),
    )
    keys = ((owner * dims[0] + cells[:, 0]) * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    ordered, order = keys.sort()
    unique, counts = torch.unique_consecutive(ordered, return_counts=True)
    starts = counts.cumsum(0) - counts
    xyz = cells[order[starts]]
    shifts = [x for x in itertools.product((-1, 0, 1), repeat=3) if x >= (0, 0, 0)]
    shifts = torch.tensor(shifts, device=positions.device)
    target = xyz[:, None] + shifts
    inside = ((target >= 0) & (target < torch.tensor(dims, device=positions.device))).all(-1)
    delta = (shifts[:, 0] * dims[1] + shifts[:, 1]) * dims[2] + shifts[:, 2]
    queries = unique[:, None] + delta
    right = torch.searchsorted(unique, queries.reshape(-1)).reshape_as(queries)
    inside &= (right < len(unique)) & (unique[right.clamp_max(len(unique) - 1)] == queries)
    left = torch.arange(len(unique), device=positions.device)[:, None].expand_as(right)
    left, right = left[inside], right[inside]
    task_counts = tr.cdiv(counts[left], tile) * tr.cdiv(counts[right], tile)
    task_starts = task_counts.cumsum(0) - task_counts
    tasks = torch.repeat_interleave(torch.arange(len(left), device=positions.device), task_counts)
    return order, starts, counts, left, right, tasks, task_starts, tile


def cell_pairs(positions, cutoff, plan):
    order, starts, counts, left, right, tasks, task_starts, tile = plan
    if not len(tasks):
        return tuple(torch.empty(0, dtype=torch.int32, device=positions.device) for _ in range(2))
    tile_counts = torch.empty(len(tasks), dtype=torch.int32, device=positions.device)
    args = (positions, order, starts, counts, left, right, tasks, task_starts)
    _tile[(len(tasks),)](
        *args,
        tile_counts,
        tile_counts,
        tile_counts,
        tile_counts,
        cutoff,
        tile,
        False,
        enable_fp_fusion=False,
    )
    cumulative = tile_counts.cumsum(0)
    size = int(cumulative[-1])
    offsets = cumulative - tile_counts
    i = torch.empty(size, dtype=torch.int32, device=positions.device)
    j = torch.empty_like(i)
    _tile[(len(tasks),)](
        *args, offsets, i, j, tile_counts, cutoff, tile, True, enable_fp_fusion=False
    )
    return i, j


def cell_chunks(positions, cutoff, plan, pair_chunk):
    order, starts, counts, left, right, tasks, task_starts, tile = plan
    # One tile emits at most tile**2 pairs. Bound search scratch space as well
    # as output; never materialize all candidate atom-pair products.
    task_chunk = max(1, pair_chunk // (tile * tile))
    for start in range(0, len(tasks), task_chunk):
        selected = tasks[start : start + task_chunk]
        adjusted = task_starts - start
        tile_counts = torch.empty(len(selected), dtype=torch.int32, device=positions.device)
        args = (positions, order, starts, counts, left, right, selected, adjusted)
        _tile[(len(selected),)](
            *args,
            tile_counts,
            tile_counts,
            tile_counts,
            tile_counts,
            cutoff,
            tile,
            False,
            enable_fp_fusion=False,
        )
        cumulative = tile_counts.cumsum(0)
        size = int(cumulative[-1])
        if not size:
            continue
        offsets = cumulative - tile_counts
        i = torch.empty(size, dtype=torch.int32, device=positions.device)
        j = torch.empty_like(i)
        _tile[(len(selected),)](
            *args, offsets, i, j, tile_counts, cutoff, tile, True, enable_fp_fusion=False
        )
        for part in range(0, size, pair_chunk):
            yield i[part : part + pair_chunk], j[part : part + pair_chunk]
