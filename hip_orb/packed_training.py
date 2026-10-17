"""Opt-in packed molecular training. Live equal-size training stays unchanged."""

import hashlib
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from hip_orb.block_layout import layout, require
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors


@lru_cache(maxsize=8)
def packed_indices(sizes):
    """Prepare geometry-only indices on CPU; loader workers can do this once per batch."""
    indices = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in layout(sizes, "cpu").items()
    }
    order, upper, row, col, owners, weights, dense_a, dense_b = ([] for _ in range(8))
    transpose_blocks = []
    atom_offset = dense_offset = upper_offset = 0
    for b, n in enumerate(sizes):
        d = 3 * n
        order.append(
            torch.arange(d * d).reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1) + dense_offset
        )
        a, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        transpose_blocks.append((c * n + a).reshape(-1) + dense_offset // 9)
        row.append(a.reshape(-1) + atom_offset)
        col.append(c.reshape(-1) + atom_offset)
        owners.append(torch.full((n * n,), b, dtype=torch.long))
        weights.append(torch.full((n * n,), 1.0 / (len(sizes) * n * n)))
        xyz = torch.arange(3)
        r = (3 * a[..., None, None] + xyz[None, None, :, None]).expand(n, n, 3, 3)
        c = (3 * c[..., None, None] + xyz[None, None, None, :]).expand(n, n, 3, 3)
        lo, hi = torch.minimum(r, c), torch.maximum(r, c)
        upper.append((lo * d - lo * (lo - 1) // 2 + hi - lo + upper_offset).reshape(-1))
        ids = torch.arange(n)
        da, db = torch.meshgrid(ids, ids, indexing="ij")
        keep = da != db
        dense_a.append(da[keep] + atom_offset)
        dense_b.append(db[keep] + atom_offset)
        atom_offset += n
        dense_offset += d * d
        upper_offset += d * (d + 1) // 2
    indices.update(
        block_order=torch.cat(order),
        transpose_blocks=torch.cat(transpose_blocks),
        upper_ids=torch.cat(upper),
        block_row=torch.cat(row),
        block_col=torch.cat(col),
        block_owner=torch.cat(owners),
        block_weight=torch.cat(weights),
        dense_a=torch.cat(dense_a),
        dense_b=torch.cat(dense_b),
    )
    return indices


def collate_packed(samples, *, indices=True):
    sizes = tuple(len(s["coords"]) for s in samples)
    result = {
        key: torch.from_numpy(np.concatenate([np.asarray(s[key]) for s in samples]))
        for key in ("coords", "atomic_numbers", "hessian_upper")
    }
    result.update(
        {
            key: torch.from_numpy(np.stack([s[key] for s in samples]))
            for key in ("charge", "spin", "omol_index")
        }
    )
    result["sizes"] = torch.tensor(sizes)
    if indices:
        result["indices"] = packed_indices(sizes)
    return result


def prepare_packed(batch, seed=7):
    identity = hashlib.sha256(batch["omol_index"].numpy().tobytes() + str(seed).encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(identity[:8], "little") % (2**63 - 1))
    rotation, triangular = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    rotation *= torch.where(triangular.diagonal() < 0, -1.0, 1.0)[None, :]
    rotation[:, 0] *= torch.linalg.det(rotation)
    return dict(
        batch,
        rotation=rotation,
        rotated_coords=torch.from_numpy(batch["coords"].numpy() @ rotation.numpy().T),
    )


def packed_graph(batch, adapter, device):
    """Use Orb's native neighbor list with arbitrary per-geometry atom offsets."""
    from orb_models.common.atoms import graph_featurization as gf
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs

    cpu_sizes = batch["sizes"]
    sizes_tuple = tuple(cpu_sizes.tolist())
    b, total = len(sizes_tuple), sum(sizes_tuple)
    if b == 1:
        from ase import Atoms

        atoms = Atoms(
            numbers=batch["atomic_numbers"].numpy(), positions=batch["rotated_coords"].numpy()
        )
        atoms.info.update(charge=int(batch["charge"][0]), spin=int(batch["spin"][0]))
        graph = adapter.from_ase_atoms_list([atoms], device=device)
        graph._hip_known_molecular = True
        return refresh_message_vectors(canonical_message_graph(graph))
    positions = batch["rotated_coords"].to(device=device, dtype=torch.float32)
    sizes = cpu_sizes.to(device)
    owners = torch.arange(b, device=device).repeat_interleave(sizes, output_size=total)
    cells = torch.zeros(b, 3, 3, device=device)
    pbc = torch.zeros(b, 3, dtype=torch.bool, device=device)
    pointers = F.pad(cpu_sizes.cumsum(0).int(), (1, 0)).to(device)
    radius = float(torch.tensor([adapter.radius]).item())
    matrix, counts, shifts = gf._compute_neighbor_list_with_fallback(
        positions=positions.contiguous(),
        cell=cells,
        pbc=pbc,
        cutoff=radius,
        batch_ptr=pointers,
        fill_value=-1,
    )
    maximum = int(adapter.max_num_neighbors)
    if max(sizes_tuple) - 1 > maximum:
        matrix, counts, shifts = gf.select_k_neighbors_from_alchemi_neighbor_matrix(
            positions,
            cells[owners],
            matrix,
            counts,
            shifts,
            max_number_neighbors=maximum,
            fill_value=-1,
        )
    edges, edge_ptr, unit_shifts = gf.get_neighbor_list_from_neighbor_matrix(
        matrix, num_neighbors=counts, neighbor_shift_matrix=shifts, fill_value=-1
    )
    degrees = edge_ptr[1:] - edge_ptr[:-1]
    edge_counts = torch.zeros(b, device=device, dtype=degrees.dtype).index_add_(0, owners, degrees)
    numbers = batch["atomic_numbers"].to(device=device, dtype=torch.long)
    atom_ids = torch.arange(total, device=device) - pointers.long()[owners]
    graph = AtomGraphs(
        senders=edges[0].long(),
        receivers=edges[1].long(),
        n_node=cpu_sizes,
        n_edge=edge_counts,
        node_features={
            "positions": positions,
            "atomic_numbers": numbers,
            "atomic_numbers_embedding": F.one_hot(numbers, num_classes=118).float(),
            "atom_identity": atom_ids,
        },
        edge_features={
            "vectors": positions[edges[1].long()] - positions[edges[0].long()],
            "unit_shifts": unit_shifts,
        },
        system_features={
            "cell": cells,
            "pbc": pbc,
            "total_charge": batch["charge"].to(device=device, dtype=torch.float32),
            "spin_multiplicity": batch["spin"].to(device=device, dtype=torch.float32),
        },
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=torch.zeros(total, device=device),
        radius=adapter.radius,
        max_num_neighbors=torch.full_like(sizes, maximum),
    )
    graph.n_node, graph.node_batch_index = sizes, owners
    graph._hip_known_molecular = True
    return refresh_message_vectors(canonical_message_graph(graph))


def _segmented_projection(blocks, indices, sizes):
    row, col, owner = (indices[k] for k in ("block_row", "block_col", "block_owner"))
    node_owner = indices["node_owner"]
    n = sizes.to(blocks.device)
    lengths = n[node_owner]
    row_mean = torch.segment_reduce(blocks, "mean", lengths=lengths, unsafe=True)
    transposed = blocks[indices["transpose_blocks"]]
    col_mean = torch.segment_reduce(transposed, "mean", lengths=lengths, unsafe=True)
    grand = torch.segment_reduce(row_mean, "mean", lengths=n, unsafe=True)
    return blocks - row_mean[row] - col_mean[col] + grand[owner]


class _PackedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, indices, sizes):
        ctx.indices, ctx.sizes = indices, sizes
        return _segmented_projection(blocks, indices, sizes)

    @staticmethod
    def backward(ctx, gradient):
        # H -> P H P is linear and self-adjoint. Apply the same projection to dH.
        return _segmented_projection(gradient, ctx.indices, ctx.sizes), None, None


def project_packed(blocks, indices, sizes):
    """Project translations with segmented reductions and an atomic-free backward."""
    return _PackedProjection.apply(blocks, indices, sizes)


def packed_blocks(model, feature, indices, sizes):
    feature.packed_layout = indices
    flat = model.head._batched_forward(feature, as_tensor="packed")
    blocks = flat[indices["block_order"]].reshape(-1, 3, 3)
    if model.head.translation_projection:
        blocks = project_packed(blocks, indices, sizes)
    return blocks


def packed_loss(blocks, target, indices, loss_type="block_rmse"):
    error = blocks - target
    if loss_type == "block_rmse":
        values = torch.linalg.vector_norm(error, dim=(1, 2)) / 3
    elif loss_type == "mse":
        values = error.square().mean((1, 2))
    elif loss_type == "mae":
        values = error.abs().mean((1, 2))
    else:
        raise ValueError("Packed experiment supports block_rmse, mse, and mae")
    return (values * indices["block_weight"]).sum()


def packed_update(model, optimizer, batch, device="cuda", seed=7, loss_type="block_rmse", *, scaler=None):
    with torch.profiler.record_function("packed_prepare"):
        batch = prepare_packed(batch, seed)
    with torch.profiler.record_function("packed_graph_and_indices"):
        graph = packed_graph(batch, model.adapter, device)
        compact = (getattr(model, "_compact_pair_loss", False)
                   and not model.head.translation_projection
                   and not hasattr(model.head, "orb_distance_encoder"))
        if compact:
            indices = compact_indices(tuple(batch["sizes"].tolist()), str(device))
        elif "indices" in batch:
            indices = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch["indices"].items()
            }
        else:
            indices = device_indices(batch["sizes"], device)
        if "dense_a" in indices:
            graph._hip_dense_topology = indices["dense_a"], indices["dense_b"]
    with torch.profiler.record_function("packed_frozen_backbone"):
        feature = model.features(graph, sizes=batch["sizes"])
    with torch.profiler.record_function("packed_head_and_loss"):
        upper = batch["hessian_upper"].to(device, non_blocking=True)
        r = batch["rotation"].to(device, non_blocking=True)
        if compact:
            feature.packed_layout = indices
            blocks, diagonal = model.head._batched_forward(feature, as_tensor="pair-blocks")
            loss = compact_loss(blocks, diagonal, upper, r, indices, loss_type)
        else:
            target = upper[indices["upper_ids"]].reshape(-1, 3, 3)
            target = r @ target @ r.T
            blocks = packed_blocks(model, feature, indices, batch["sizes"])
            loss = packed_loss(blocks, target, indices, loss_type)
    require(torch.isfinite(loss), "Packed training loss is not finite")
    with torch.profiler.record_function("packed_backward_and_optimizer"):
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        from hip_orb.mixed_precision import invalidate_trainable_precision_caches
        invalidate_trainable_precision_caches(model)
    return float(loss.detach()), len(batch["sizes"])


def collate_packed_compact(samples):
    """Transfer coordinates, labels, and small size metadata; build indices on device."""
    return collate_packed(samples, indices=False)


def device_indices(cpu_sizes, device):
    """Reuse bounded geometry-only layouts across repeated batch shapes."""
    return _device_indices(tuple(cpu_sizes.tolist()), str(device))


def clear_device_indices():
    _device_indices.cache_clear()


@lru_cache(maxsize=8)
def _device_indices(values, device):
    """Vectorized offset indexing without per-geometry GPU launches or padding."""
    cpu_sizes = torch.tensor(values, dtype=torch.long)
    total_nodes = sum(values)
    total_pairs = sum(n * (n - 1) // 2 for n in values)
    total_blocks = sum(n * n for n in values)
    sizes = cpu_sizes.to(device)
    owner_ids = torch.arange(len(values), device=device)
    offsets = F.pad(sizes.cumsum(0), (1, 0))
    dense_offsets = F.pad((9 * sizes * sizes).cumsum(0), (1, 0))
    upper_offsets = F.pad(((3 * sizes) * (3 * sizes + 1) // 2).cumsum(0), (1, 0))
    node_owner = owner_ids.repeat_interleave(sizes, output_size=total_nodes)
    counts = sizes * (sizes - 1) // 2
    owner = owner_ids.repeat_interleave(counts, output_size=total_pairs)
    pair_offsets = F.pad(counts.cumsum(0), (1, 0))
    t = torch.arange(total_pairs, device=device) - pair_offsets[owner]
    n = sizes[owner]
    a = n - 2 - torch.floor((torch.sqrt((4 * n * (n - 1) - 8 * t - 7).double()) - 1) / 2).long()
    start = a * (2 * n - a - 1) // 2
    b = t - start + a + 1
    i, j = a + offsets[owner], b + offsets[owner]
    xyz = torch.arange(3, device=device)
    row = 3 * a[:, None, None] + xyz[None, :, None]
    col = 3 * b[:, None, None] + xyz[None, None, :]
    d = 3 * n[:, None, None]
    origin = dense_offsets[owner, None, None]
    forward = (origin + row * d + col).reshape(-1)
    reverse = (origin + col * d + row).reshape(-1)
    local_node = torch.arange(total_nodes, device=device) - offsets[node_owner]
    row = 3 * local_node[:, None, None] + xyz[None, :, None]
    col = 3 * local_node[:, None, None] + xyz[None, None, :]
    diag = (
        dense_offsets[node_owner, None, None] + row * (3 * sizes[node_owner, None, None]) + col
    ).reshape(-1)
    owner = owner_ids.repeat_interleave(sizes * sizes, output_size=total_blocks)
    block_offsets = F.pad((sizes * sizes).cumsum(0), (1, 0))
    t = torch.arange(total_blocks, device=device) - block_offsets[owner]
    n = sizes[owner]
    a, b = t // n, t % n
    transpose_blocks = block_offsets[owner] + b * n + a
    row = 3 * a[:, None, None] + xyz[None, :, None]
    col = 3 * b[:, None, None] + xyz[None, None, :]
    row, col = row.expand(-1, 3, 3), col.expand(-1, 3, 3)
    order = (dense_offsets[owner, None, None] + row * (3 * n[:, None, None]) + col).reshape(-1)
    lo, hi = torch.minimum(row, col), torch.maximum(row, col)
    upper = (
        upper_offsets[owner, None, None]
        + lo * (3 * n[:, None, None])
        - lo * (lo - 1) // 2
        + hi
        - lo
    ).reshape(-1)
    dcounts = sizes * (sizes - 1)
    downer = owner_ids.repeat_interleave(dcounts, output_size=2 * total_pairs)
    dt = torch.arange(2 * total_pairs, device=device) - F.pad(dcounts.cumsum(0), (1, 0))[downer]
    dn = sizes[downer]
    da = dt // (dn - 1)
    db = dt % (dn - 1)
    db = db + (db >= da).long()
    return {
        "i": i,
        "j": j,
        "node_owner": node_owner,
        "dense_forward": forward,
        "dense_reverse": reverse,
        "dense_diag": diag,
        "dense_sizes": [9 * n * n for n in values],
        "block_order": order,
        "transpose_blocks": transpose_blocks,
        "upper_ids": upper,
        "block_row": a + offsets[owner],
        "block_col": b + offsets[owner],
        "block_owner": owner,
        "block_weight": 1.0 / (len(values) * n.float() * n),
        "dense_a": da + offsets[downer],
        "dense_b": db + offsets[downer],
    }


@lru_cache(maxsize=8)
def compact_indices(values, device):
    """Index independent pair blocks and diagonal blocks without dense matrices."""
    sizes = torch.tensor(values, device=device, dtype=torch.long)
    owners = torch.arange(len(values), device=device)
    total_nodes = sum(values)
    total_pairs = sum(n * (n - 1) // 2 for n in values)
    node_owner = owners.repeat_interleave(sizes, output_size=total_nodes)
    counts = sizes * (sizes - 1) // 2
    pair_owner = owners.repeat_interleave(counts, output_size=total_pairs)
    offsets = F.pad(sizes.cumsum(0), (1, 0))
    pair_offsets = F.pad(counts.cumsum(0), (1, 0))
    upper_offsets = F.pad(((3 * sizes) * (3 * sizes + 1) // 2).cumsum(0), (1, 0))
    t = torch.arange(total_pairs, device=device) - pair_offsets[pair_owner]
    n = sizes[pair_owner]
    a = n - 2 - torch.floor((torch.sqrt((4 * n * (n - 1) - 8 * t - 7).double()) - 1) / 2).long()
    b = t - a * (2 * n - a - 1) // 2 + a + 1
    xyz = torch.arange(3, device=device)
    row, col = 3 * a[:, None, None] + xyz[None, :, None], 3 * b[:, None, None] + xyz[None, None, :]
    d = 3 * n[:, None, None]
    pair_upper = upper_offsets[pair_owner, None, None] + row * d - row * (row - 1) // 2 + col - row
    node = torch.arange(total_nodes, device=device) - offsets[node_owner]
    row, col = 3 * node[:, None, None] + xyz[None, :, None], 3 * node[:, None, None] + xyz[None, None, :]
    lo, hi = torch.minimum(row, col), torch.maximum(row, col)
    d = 3 * sizes[node_owner, None, None]
    diag_upper = upper_offsets[node_owner, None, None] + lo * d - lo * (lo - 1) // 2 + hi - lo
    return {"i": a + offsets[pair_owner], "j": b + offsets[pair_owner], "node_owner": node_owner,
            "pair_upper": pair_upper, "diag_upper": diag_upper,
            "pair_weight": 2.0 / (len(values) * n.float().square()),
            "diag_weight": 1.0 / (len(values) * sizes[node_owner].float().square())}


def compact_loss(blocks, diagonal, upper, rotation, indices, loss_type="block_rmse"):
    """Weight symmetric off-diagonal blocks twice and diagonal blocks once."""
    def terms(prediction, ids, weights):
        target = rotation @ upper[ids] @ rotation.T
        error = prediction - target
        if loss_type == "block_rmse":
            values = torch.linalg.vector_norm(error, dim=(1, 2)) / 3
        elif loss_type == "mse":
            values = error.square().mean((1, 2))
        elif loss_type == "mae":
            values = error.abs().mean((1, 2))
        else:
            raise ValueError("Compact loss supports block_rmse, mse, and mae")
        return (values * weights).sum()
    return terms(blocks, indices["pair_upper"], indices["pair_weight"]) + terms(
        diagonal, indices["diag_upper"], indices["diag_weight"])
