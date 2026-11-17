"""Exact, uncapped molecular radius pairs for sparse Hessian blocks.

Search is discrete and detached. Pair geometry is recomputed by the head, so
the choice of search backend does not change the trainable model.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.spatial import cKDTree

CPU_PAIR_ARRAY_BUDGET = 4 * 1024**3


def _validate(positions, sizes, cutoff):
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (N, 3)")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff must be finite and positive")
    cutoff32 = float(torch.tensor(cutoff, dtype=torch.float32))
    if not math.isfinite(cutoff32) or cutoff32 <= 0:
        raise ValueError("Cutoff must be representable in float32")
    raw_sizes = torch.as_tensor(sizes)
    if raw_sizes.ndim != 1:
        raise ValueError("Molecule sizes must be a vector")
    values = raw_sizes.tolist()
    sizes = tuple(int(n) for n in values)
    if any(n != raw for n, raw in zip(sizes, values, strict=True)):
        raise ValueError("Molecule sizes must be integers")
    if any(n < 0 for n in sizes) or sum(sizes) != len(positions):
        raise ValueError("Molecule sizes must cover the positions")
    if len(positions) >= 2**31:
        raise ValueError("Pair indices require fewer than 2**31 atoms")
    positions = positions.detach().float().contiguous()
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("Positions must be finite")
    return positions, sizes


def _empty(device):
    return tuple(torch.empty(0, dtype=torch.int32, device=device) for _ in range(2))


def _inside(positions, i, j, cutoff):
    # A three-coordinate CUDA cdist/norm reduction adds x,z before y. Keep
    # that order on CPU as well: changing it can move a boundary by one ULP.
    delta = positions[i.long()] - positions[j.long()]
    squared = delta[:, 0].square() + delta[:, 2].square()
    return (squared + delta[:, 1].square()).sqrt() < cutoff


class RadiusPairs:
    """Reusable pair enumeration; chunks never exceed ``pair_chunk`` pairs."""

    def __init__(self, positions, sizes, cutoff, backend="auto", rows=256):
        self.positions, self.sizes = _validate(positions, sizes, cutoff)
        self.cutoff, self.rows = float(cutoff), rows
        if backend == "auto":
            backend = "brute" if len(positions) <= 512 else "gpu" if positions.is_cuda else "cpu"
        if backend not in ("brute", "cpu", "gpu"):
            raise ValueError(f"Unknown radius search backend: {backend}")
        if backend == "gpu" and not positions.is_cuda:
            raise ValueError("GPU radius search requires CUDA positions")
        if rows < 1:
            raise ValueError("Search rows must be positive")
        self.backend = backend
        self._plan = None
        if backend == "gpu" and len(positions):
            from hip_orb.neighbor_kernels import cell_plan

            self._plan = cell_plan(self.positions, self.sizes, self.cutoff)

    def chunks(self, pair_chunk=262144):
        if pair_chunk < 1:
            raise ValueError("Pair chunk must be positive")
        if not len(self.positions):
            return
        if self.backend == "gpu":
            from hip_orb.neighbor_kernels import cell_chunks

            source = cell_chunks(self.positions, self.cutoff, self._plan, pair_chunk)
        elif self.backend == "cpu":
            source = self._cpu_chunks()
        else:
            source = self._brute_chunks()
        for i, j in source:
            for start in range(0, len(i), pair_chunk):
                yield i[start : start + pair_chunk], j[start : start + pair_chunk]

    def materialize(self):
        if self.backend == "cpu" and self.positions.is_cuda:
            pairs = RadiusPairs(
                self.positions.cpu(), self.sizes, self.cutoff, "cpu", self.rows
            ).materialize()
            return tuple(x.pin_memory().to(self.positions.device, non_blocking=True) for x in pairs)
        if self.backend == "gpu" and self._plan is not None:
            from hip_orb.neighbor_kernels import cell_pairs

            return cell_pairs(self.positions, self.cutoff, self._plan)
        parts = list(self.chunks())
        if not parts:
            return _empty(self.positions.device)
        return tuple(torch.cat([part[k] for part in parts]) for k in range(2))

    def _brute_chunks(self):
        positions, offset = self.positions, 0
        for n in self.sizes:
            for start in range(offset, offset + n, self.rows):
                stop = min(start + self.rows, offset + n)
                a = torch.arange(start, stop, device=positions.device)
                b = torch.arange(start, offset + n, device=positions.device)
                distance = torch.cdist(
                    positions[a], positions[b], compute_mode="donot_use_mm_for_euclid_dist"
                )
                radius = float(np.float32(self.cutoff)) * (1 + 8 * np.finfo(np.float32).eps)
                x, y = ((distance < radius) & (a[:, None] < b)).nonzero(as_tuple=True)
                i, j = a[x].int(), b[y].int()
                keep = _inside(positions, i, j, self.cutoff)
                yield i[keep], j[keep]
            offset += n

    def _cpu_chunks(self):
        # The radius is conservative; final membership uses float32 arithmetic.
        positions = self.positions.cpu()
        radius = float(np.float32(self.cutoff)) * (1 + 8 * np.finfo(np.float32).eps)
        offset = 0
        for n in self.sizes:
            xyz = positions[offset : offset + n].numpy()
            if n < 2:
                offset += n
                continue
            tree = cKDTree(xyz)
            upper = n * (n - 1) // 2
            if upper * 16 > CPU_PAIR_ARRAY_BUDGET:
                upper = (int(tree.count_neighbors(tree, radius)) - n) // 2
            if upper * 16 <= CPU_PAIR_ARRAY_BUDGET:
                pairs = tree.query_pairs(radius, eps=0, output_type="ndarray")
                for start in range(0, len(pairs), 1048576):
                    part = pairs[start : start + 1048576]
                    i = torch.from_numpy(part[:, 0].copy() + offset).int()
                    j = torch.from_numpy(part[:, 1].copy() + offset).int()
                    keep = _inside(positions, i, j, self.cutoff)
                    yield i[keep].to(self.positions.device), j[keep].to(self.positions.device)
                offset += n
                continue
            for start in range(0, n, self.rows):
                lists = tree.query_ball_point(
                    xyz[start : start + self.rows], radius, eps=0, workers=1, return_sorted=False
                )
                lengths = np.fromiter((len(x) for x in lists), dtype=np.int64)
                i = np.repeat(np.arange(start, start + len(lists)), lengths)
                j = np.concatenate(lists).astype(np.int64, copy=False)
                keep = i < j
                i = torch.from_numpy(i[keep] + offset).int()
                j = torch.from_numpy(j[keep] + offset).int()
                keep = _inside(positions, i, j, self.cutoff)
                yield i[keep].to(self.positions.device), j[keep].to(self.positions.device)
            offset += n


def neighbor_pairs(positions, sizes, cutoff, rows=256, *, backend="auto"):
    """Return uncapped int32 pairs, i < j, without cross-molecule pairs."""
    return RadiusPairs(positions, sizes, cutoff, backend, rows).materialize()


class AsyncCPUPairs:
    """Build CPU pairs while the frozen GPU backbone runs.

    Submit final CPU coordinates before graph transfer/forward. Coordinates are
    copied at submission, so loader buffer reuse cannot change the search.
    Consume the result with ``result(device)``; use this object as a context
    manager to release its worker even when inference fails.
    """

    def __init__(self, positions, sizes, cutoff):
        if positions.device.type != "cpu":
            raise ValueError("Async CPU search requires original CPU coordinates")
        positions, sizes = _validate(positions, sizes, cutoff)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hip-radius")
        self._future = self._executor.submit(
            neighbor_pairs, positions.clone(), sizes, cutoff, backend="cpu"
        )

    def result(self, device):
        pairs = self._future.result()
        if torch.device(device).type == "cuda":
            pairs = tuple(x.pin_memory().to(device, non_blocking=True) for x in pairs)
        return pairs

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
