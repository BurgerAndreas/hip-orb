"""Reuse molecular connectivity across rotations with a conservative margin guard."""

from copy import copy

import numpy as np
import torch


def topology_reuse_safe(batch, adapter, graph=None):
    """Reject cutoff boundaries, neighbor-rank ties, and nonfinite coordinates.

    Rotation and FP32 coordinate conversion can change boundary decisions.
    Check a generous error envelope before reusing the first view's graph.
    CUDA checks use the first view's positions without a second neighbor list.
    CPU checks use the original coordinates.
    """
    radius, maximum = float(adapter.radius), int(adapter.max_num_neighbors)
    if not np.isfinite(radius) or radius <= 0 or maximum < 1:
        return False
    if (
        graph is not None
        and graph.node_features["positions"].is_cuda
        and graph.node_features["positions"].dtype == torch.float32
    ):
        from hip_orb.two_view_graph_kernels import reuse_margin_safe

        return reuse_margin_safe(graph, tuple(batch["sizes"].tolist()), radius, maximum)
    coords = batch["coords"].numpy()
    offset = 0
    for n in batch["sizes"].tolist():
        positions = np.asarray(coords[offset : offset + n], dtype=np.float64)
        offset += n
        if not np.isfinite(positions).all():
            return False
        extent = float(np.abs(positions).max(initial=0))
        if extent > 1e6:
            return False
        # Covers FP32 rotations, coordinate casts, subtraction, and norm error.
        tolerance = 128 * np.finfo(np.float32).eps * (extent + 1) ** 2
        if maximum >= n - 1 and radius**2 > 12 * extent**2 + tolerance:
            continue
        delta = positions[:, None, :] - positions[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        np.fill_diagonal(squared, np.inf)
        if np.any(np.abs(squared - radius**2) <= tolerance):
            return False
        if 0 < maximum < n - 1:
            nearest = np.partition(squared, (maximum - 1, maximum), axis=1)
            inside = nearest[:, maximum - 1] < radius**2 + tolerance
            if np.any(inside & (nearest[:, maximum] - nearest[:, maximum - 1] <= 2 * tolerance)):
                return False
    return True


def reuse_graph(graph, rotated_batch, *, fused=True):
    """Share immutable indices and state; give the second view its own geometry."""
    result = copy(graph)
    positions = rotated_batch["rotated_coords"].to(
        device=graph.senders.device, dtype=graph.node_features["positions"].dtype
    )
    result.node_features = dict(graph.node_features, positions=positions)
    result.edge_features = dict(graph.edge_features)
    result.system_features = dict(graph.system_features)
    result.node_targets, result.edge_targets = dict(graph.node_targets), dict(graph.edge_targets)
    result.system_targets = dict(graph.system_targets)
    if fused and positions.is_cuda and positions.dtype == torch.float32:
        from hip_orb.two_view_graph_kernels import edge_vectors

        vectors = edge_vectors(positions, graph.senders, graph.receivers)
    else:
        vectors = positions[graph.receivers] - positions[graph.senders]
    result.edge_features["vectors"] = vectors
    return result
