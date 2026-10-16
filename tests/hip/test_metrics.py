import pytest
import torch

from hip_orb.metrics import hessian_metrics, vibrational_basis


def test_mode_cosine_uses_mass_weighted_projected_modes():
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    masses = torch.tensor([1., 12., 16.], dtype=torch.float64)
    basis = vibrational_basis(positions, masses)
    eigenvalues = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
    rotation = torch.tensor([[0.6, -0.8, 0.], [0.8, 0.6, 0.], [0., 0., 1.]], dtype=torch.float64)
    roots = masses.repeat_interleave(3).sqrt()

    def raw(reduced):
        weighted = basis @ reduced @ basis.T
        return weighted * roots[:, None] * roots[None, :]

    target = raw(eigenvalues)
    predicted = raw(rotation @ eigenvalues @ rotation.T)
    metrics = hessian_metrics(predicted, target, positions, masses)
    assert metrics["first_mode_abs_cosine"] == pytest.approx(0.6)
    assert metrics["first_mode_overlap"] == pytest.approx(0.36)
    assert metrics["eigenvalue_mae"] == pytest.approx(0., abs=1e-12)
