import h5py
import numpy as np
import torch

from hip_orb.metrics import hessian_metrics, vibrational_basis
from hip_orb.train import read_samples, transformed_sample


def test_collection_reader_and_rotation(tmp_path):
    path = tmp_path / "shard_00000.h5"
    h = np.arange(36, dtype=np.float32).reshape(6, 6)
    h = h + h.T
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, model="orbmol_v2")
        for key, values in {
            "natoms": [2],
            "atom_ptr": [0, 2],
            "hessian_ptr": [0, 36],
            "atomic_numbers": [1, 1],
            "coords": [[0, 0, 0], [1, 0, 0]],
            "charge": [0],
            "spin": [1],
            "omol_index": [17],
            "hessian_flat": h.reshape(-1),
        }.items():
            f.create_dataset(key, data=values)
    sample = read_samples(tmp_path, 1, 2, 2)[0]
    np.testing.assert_array_equal(sample[1], h)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _, transformed, _ = transformed_sample(sample, rotation)
    full = torch.kron(torch.eye(2), rotation).numpy()
    np.testing.assert_allclose(transformed, full @ h @ full.T)


def test_linear_and_nonlinear_vibrational_rank():
    masses = torch.ones(3)
    linear = torch.tensor([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    bent = linear.clone()
    bent[1, 1] = 1
    assert vibrational_basis(linear, masses).shape == (9, 4)
    assert vibrational_basis(bent, masses).shape == (9, 3)
    h = torch.eye(9)
    metrics = hessian_metrics(h, h, bent, masses)
    assert metrics["mae"] == 0
    assert abs(metrics["low_subspace_overlap"] - 1) < 1e-5
