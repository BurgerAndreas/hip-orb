"""Deterministic CPU augmentation, optionally inside spawned loader workers."""
import hashlib

from ase import Atoms
import torch

from hip_orb.data import collate_samples
from hip_orb.graphs import canonical_message_graph, hessian_pair_topology


def prepare_batch(batch, seed=0, adapter=None):
    # Local RNG makes augmentation independent of worker count and scheduling.
    identities = batch["omol_index"].numpy().tobytes() if "omol_index" in batch else batch["coords"].numpy().tobytes()
    identity = hashlib.sha256(identities + str(seed).encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(identity[:8], "little") % (2**63 - 1))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    rotation[:, 0] *= torch.linalg.det(rotation)
    positions = batch["coords"].numpy() @ rotation.numpy().T
    molecules = []
    for i, coords in enumerate(positions):
        atoms = Atoms(numbers=batch["atomic_numbers"][i].numpy(), positions=coords)
        atoms.info.update(charge=int(batch["charge"][i]), spin=int(batch["spin"][i]))
        molecules.append(atoms)
    result = dict(batch, rotation=rotation, molecules=molecules)
    if adapter is not None:
        graph = canonical_message_graph(adapter.from_ase_atoms_list(molecules, device="cpu", edge_method="knn_scipy"))
        result["prepared_pairs"] = hessian_pair_topology(graph.senders, graph.receivers, graph.n_node.tolist())
        result["prepared_graph"] = graph
    return result


class PreparedCollator:
    """Contains only a CPU adapter, never the model or CUDA tensors."""
    def __init__(self, seed=0, adapter=None):
        self.seed, self.adapter = seed, adapter

    def __call__(self, samples):
        return prepare_batch(collate_samples(samples), self.seed, self.adapter)
