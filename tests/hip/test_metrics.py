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


@pytest.mark.parametrize('n', [1, 2, 4, 16])
@pytest.mark.parametrize('batch', [1, 4])
def test_block_rotations_match_dense_transform_and_gradients(n, batch):
    from hip_orb.metrics import rotate_hessians
    torch.manual_seed(45)
    h = torch.randn(batch, 3*n, 3*n, dtype=torch.float64, requires_grad=True)
    r = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    transform = torch.kron(torch.eye(n, dtype=h.dtype), r.contiguous())
    expected = transform @ h @ transform.T
    actual = rotate_hessians(h, r)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual.square().sum(), h)[0]
    b = torch.autograd.grad(expected.square().sum(), h)[0]
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('n', [1, 2, 3, 8])
@pytest.mark.parametrize('linear', [False, True])
def test_batched_metrics_match_individual_metrics(n, linear):
    from hip_orb.metrics import batched_hessian_metrics
    torch.manual_seed(97)
    positions = torch.randn(4, n, 3, dtype=torch.float64)
    if linear:
        positions[..., 1:] = 0
    masses = torch.rand(4, n, dtype=torch.float64) + 1
    h = torch.randn(4, 3*n, 3*n, dtype=torch.float64)
    h = h + h.transpose(-1, -2)
    target = torch.randn_like(h)
    target = target + target.transpose(-1, -2)
    actual = batched_hessian_metrics(h, target, positions, masses)
    for i, row in enumerate(actual):
        expected = hessian_metrics(h[i], target[i], positions[i], masses[i])
        assert row.keys() == expected.keys()
        for key in row:
            assert row[key] == pytest.approx(expected[key], abs=2e-11, rel=2e-11), key
