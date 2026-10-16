"""Build equal-size molecular training graphs without per-molecule ASE copies.

The neighbor-list call, radius, maximum neighbor count, and final geometry are
the same as Orb's batched adapter. This path accepts only the packed molecular
training fields, without cells, constraints, or user-supplied extra features.
"""

import torch
from torch.nn import functional as F


def molecular_edges(positions, cells, pbc, sizes, owners, adapter, b, n):
    """Keep Orb's radius list and omit an unnecessary top-k sort for small molecules."""
    from orb_models.common.atoms import graph_featurization as graph
    maximum = int(adapter.max_num_neighbors)
    helpers = ("_compute_neighbor_list_with_fallback", "get_neighbor_list_from_neighbor_matrix",
               "select_k_neighbors_from_alchemi_neighbor_matrix")
    if maximum <= 0 or not all(hasattr(graph, name) for name in helpers):
        return graph.batch_compute_pbc_radius_graph(
            positions=positions.contiguous(), cells=cells, pbcs=pbc,
            radius=torch.tensor([adapter.radius], device=positions.device), n_node=sizes,
            node_batch_index=owners, max_number_neighbors=torch.full_like(sizes, maximum),
            device=positions.device)
    pointers = torch.arange(b + 1, dtype=torch.int32, device=positions.device) * n
    # Match the scalar conversion in Orb's public batched graph routine.
    radius = float(torch.tensor([adapter.radius], device="cpu").item())
    neighbors, counts, shifts = graph._compute_neighbor_list_with_fallback(
        positions=positions, cell=cells, pbc=pbc, cutoff=radius,
        batch_ptr=pointers, fill_value=-1)
    if n - 1 > maximum:
        neighbors, counts, shifts = graph.select_k_neighbors_from_alchemi_neighbor_matrix(
            positions, cells[owners], neighbors, counts, shifts,
            max_number_neighbors=maximum, fill_value=-1)
    edges, edge_pointers, unit_shifts = graph.get_neighbor_list_from_neighbor_matrix(
        neighbors, num_neighbors=counts, neighbor_shift_matrix=shifts, fill_value=-1)
    degrees = edge_pointers[1:] - edge_pointers[:-1]
    counts = degrees.reshape(b, n).sum(1, dtype=degrees.dtype)
    vectors = positions[edges[1].long()] - positions[edges[0].long()]
    return edges, vectors, unit_shifts, counts


def molecular_training_graph(batch, adapter, device):
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs

    if "rotated_coords" not in batch:
        raise ValueError("Tensor graphs require packed rotated coordinates")
    coords = batch["rotated_coords"]
    b, n, _ = coords.shape
    # Orb's single-system adapter uses a separate graph routine. Keep that path
    # until its edge membership has an independent agreement audit.
    if b == 1:
        from ase import Atoms
        atoms = Atoms(numbers=batch["atomic_numbers"][0].numpy(), positions=coords[0].numpy())
        atoms.info.update(charge=int(batch["charge"][0]), spin=int(batch["spin"][0]))
        graph = adapter.from_ase_atoms_list([atoms], device=device)
        graph._hip_known_molecular = True
        return graph
    positions = coords.reshape(-1, 3).to(device=device, dtype=torch.float32)
    cpu_sizes = torch.full((b,), n, dtype=torch.long)
    sizes = cpu_sizes.to(device)
    owners = torch.arange(b, device=device).repeat_interleave(n)
    cells = torch.zeros((b, 3, 3), device=device, dtype=torch.float32)
    pbc = torch.zeros((b, 3), device=device, dtype=torch.bool)
    neighbors = torch.full_like(sizes, adapter.max_num_neighbors)
    edges, vectors, shifts, counts = molecular_edges(positions, cells, pbc, sizes, owners, adapter, b, n)
    numbers = batch["atomic_numbers"].reshape(-1).to(device=device, dtype=torch.long)
    graph = AtomGraphs(
        senders=edges[0].long(), receivers=edges[1].long(), n_node=cpu_sizes, n_edge=counts,
        node_features={"positions": positions.float(), "atomic_numbers": numbers,
                       "atomic_numbers_embedding": F.one_hot(numbers, num_classes=118).float(),
                       "atom_identity": torch.arange(n, device=device).repeat(b)},
        edge_features={"vectors": vectors.float(), "unit_shifts": shifts.float()},
        system_features={"cell": cells.float(), "pbc": pbc,
                         "total_charge": batch["charge"].to(device=device, dtype=torch.float32),
                         "spin_multiplicity": batch["spin"].to(device=device, dtype=torch.float32)},
        node_targets={}, edge_targets={}, system_targets={}, system_id=None, fix_atoms=None,
        tags=torch.zeros(b * n, device=device), radius=adapter.radius, max_num_neighbors=neighbors)
    # Constructing known scalar metadata on CPU avoids two device-to-host waits
    # in AtomGraphs.__init__. All returned graph tensors live on the target device.
    graph.n_node = sizes
    graph.node_batch_index = owners
    graph._hip_known_molecular = True
    return graph
