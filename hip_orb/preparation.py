"""Deterministic CPU augmentation, optionally inside spawned loader workers."""
import hashlib
import math

from ase import Atoms
import torch

from hip_orb.data import collate_samples
from hip_orb.graphs import canonical_message_graph, hessian_pair_topology


def prepare_batch(batch, seed=0, adapter=None, translation_scale=0.0,
                  rotation_sampling="uniform", include_molecules=True):
    if rotation_sampling not in ("uniform", "legacy"):
        raise ValueError("Rotation sampling must be uniform or legacy")
    if not math.isfinite(translation_scale) or translation_scale < 0:
        raise ValueError("Translation scale must be finite and nonnegative")
    # Local RNG makes augmentation independent of worker count and scheduling.
    identities = batch["omol_index"].numpy().tobytes() if "omol_index" in batch else batch["coords"].numpy().tobytes()
    identity = hashlib.sha256(identities + str(seed).encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(identity[:8], "little") % (2**63 - 1))
    rotation, triangular = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    # QR chooses column signs. Correct them before restricting Haar O(3) to SO(3).
    if rotation_sampling == "uniform":
        signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
        rotation *= signs[None, :]
    rotation[:, 0] *= torch.linalg.det(rotation)
    positions = batch["coords"].numpy() @ rotation.numpy().T
    translation = torch.zeros(len(positions), 1, 3)
    if translation_scale:
        translation.uniform_(-translation_scale, translation_scale, generator=generator)
        positions = positions + translation.numpy()
    if not include_molecules:
        if adapter is not None:
            raise ValueError("CPU graph preparation requires ASE molecules")
        result = dict(batch, rotation=rotation, translation=translation,
                      rotated_coords=torch.from_numpy(positions))
        result.pop("molecules", None)
        return result
    molecules = []
    for i, coords in enumerate(positions):
        atoms = Atoms(numbers=batch["atomic_numbers"][i].numpy(), positions=coords)
        atoms.info.update(charge=int(batch["charge"][i]), spin=int(batch["spin"][i]))
        molecules.append(atoms)
    result = dict(batch, rotation=rotation, translation=translation, molecules=molecules)
    if adapter is not None:
        graph = canonical_message_graph(adapter.from_ase_atoms_list(molecules, device="cpu", edge_method="knn_scipy"))
        result["prepared_pairs"] = hessian_pair_topology(graph.senders, graph.receivers, graph.n_node.tolist())
        result["prepared_graph"] = graph
    return result


class PreparedCollator:
    """Contains only a CPU adapter, never the model or CUDA tensors."""
    def __init__(self, seed=0, adapter=None, translation_scale=0.0,
                 rotation_sampling="uniform"):
        self.seed, self.adapter = seed, adapter
        self.translation_scale = translation_scale
        self.rotation_sampling = rotation_sampling

    def __call__(self, samples):
        return prepare_batch(collate_samples(samples), self.seed, self.adapter,
                             translation_scale=self.translation_scale,
                             rotation_sampling=self.rotation_sampling)
