import pytest
import torch

from hip_orb.losses import hessian_error_loss


def test_block_rmse_matches_independent_blocks_and_gradient():
    error = torch.randn(2, 9, 9, dtype=torch.float64, requires_grad=True)
    expected = torch.stack([
        error[b, 3*i:3*i+3, 3*j:3*j+3].norm() / 3
        for b in range(2) for i in range(3) for j in range(3)
    ]).mean()
    actual = hessian_error_loss(error, "block_rmse")
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, error, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, error)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_block_rmse_value_and_gradient_rotate_correctly():
    torch.manual_seed(11)
    error = torch.randn(2, 12, 12, dtype=torch.float64, requires_grad=True)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    transform = torch.kron(torch.eye(4, dtype=torch.float64), rotation.contiguous())
    rotated = (transform @ error.detach() @ transform.T).requires_grad_()
    original_loss = hessian_error_loss(error, "block_rmse")
    rotated_loss = hessian_error_loss(rotated, "block_rmse")
    torch.testing.assert_close(original_loss, rotated_loss)
    original_loss.backward()
    rotated_loss.backward()
    torch.testing.assert_close(rotated.grad, transform @ error.grad @ transform.T)


def test_block_rmse_zero_blocks_have_finite_zero_gradients():
    error = torch.zeros(2, 6, 6, requires_grad=True)
    loss = hessian_error_loss(error, "block_rmse")
    loss.backward()
    assert float(loss.detach()) == 0
    assert torch.isfinite(error.grad).all()
    assert torch.count_nonzero(error.grad) == 0


def test_block_rmse_has_linear_growth_and_passes_gradcheck():
    error = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(hessian_error_loss(7 * error, "block_rmse"),
                               7 * hessian_error_loss(error, "block_rmse"))
    assert torch.autograd.gradcheck(lambda value: hessian_error_loss(value, "block_rmse"), (error,))


def test_block_rmse_averages_blocks_before_geometries():
    error = torch.zeros(2, 6, 6)
    error[0, :3, :3] = 4
    # One block has RMSE 4; the other seven blocks have RMSE zero.
    assert float(hessian_error_loss(error, "block_rmse")) == 0.5


@pytest.mark.parametrize("shape", [(6, 6), (2, 6, 9), (2, 5, 5)])
def test_invalid_block_shapes_fail(shape):
    with pytest.raises(ValueError, match="3N-by-3N"):
        hessian_error_loss(torch.zeros(shape), "block_rmse")


def test_mass_weighted_mse_matches_explicit_basis_transform_and_gradient():
    error = torch.randn(2, 9, 9, dtype=torch.float64, requires_grad=True)
    masses = torch.tensor([[1.008, 12.011, 15.999], [2.014, 14.007, 32.06]], dtype=torch.float64)
    transform = torch.diag_embed(masses.repeat_interleave(3, dim=-1).rsqrt())
    expected = (transform @ error @ transform).square().mean()
    actual = hessian_error_loss(error, "mass_weighted_mse", masses)
    torch.testing.assert_close(actual, expected)
    a, = torch.autograd.grad(actual, error, retain_graph=True)
    b, = torch.autograd.grad(expected, error)
    torch.testing.assert_close(a, b)


def test_mass_weighted_mse_is_rotation_invariant():
    error = torch.randn(2, 9, 9, dtype=torch.float64)
    masses = torch.tensor([[1., 12., 16.], [12., 1., 32.]], dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    transform = torch.kron(torch.eye(3, dtype=torch.float64), rotation.contiguous())
    torch.testing.assert_close(hessian_error_loss(transform @ error @ transform.T,
                                                 "mass_weighted_mse", masses),
                               hessian_error_loss(error, "mass_weighted_mse", masses))


def test_mass_weighted_mse_scaling_and_zero_gradient():
    error = torch.zeros(2, 6, 6, requires_grad=True)
    masses = torch.tensor([[1., 12.], [16., 32.]])
    loss = hessian_error_loss(error, "mass_weighted_mse", masses)
    loss.backward()
    assert float(loss.detach()) == 0
    assert torch.count_nonzero(error.grad) == 0
    error = torch.ones(2, 6, 6)
    torch.testing.assert_close(hessian_error_loss(error, "mass_weighted_mse", 2 * masses),
                               hessian_error_loss(error, "mass_weighted_mse", masses) / 4)


def test_mass_weighted_block_rmse_rotation_gradient_and_zero():
    from hip_orb.losses import rotate_hessian
    masses = torch.tensor([[1., 12., 16.], [14., 1., 32.]], dtype=torch.float64)
    error = torch.randn(2, 9, 9, dtype=torch.float64, requires_grad=True)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    weights = masses.repeat_interleave(3, dim=1).rsqrt()
    expected = hessian_error_loss(error * weights[:, :, None] * weights[:, None, :], "block_rmse")
    actual = hessian_error_loss(error, "mass_weighted_block_rmse", masses)
    torch.testing.assert_close(actual, expected)
    a, = torch.autograd.grad(actual, error, retain_graph=True)
    b, = torch.autograd.grad(expected, error)
    torch.testing.assert_close(a, b)
    rotated = rotate_hessian(error.detach(), rotation).requires_grad_()
    rotated_loss = hessian_error_loss(rotated, "mass_weighted_block_rmse", masses)
    torch.testing.assert_close(rotated_loss, actual)
    rotated_loss.backward()
    torch.testing.assert_close(rotated.grad, rotate_hessian(a, rotation))
    zero = torch.zeros_like(error, requires_grad=True)
    hessian_error_loss(zero, "mass_weighted_block_rmse", masses).backward()
    assert torch.count_nonzero(zero.grad) == 0
