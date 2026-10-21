"""Hessian losses with tensor reductions over equal-size geometry batches."""

import torch

LOSS_TYPES = ("mse", "mae", "block_rmse", "mass_weighted_mse", "mass_weighted_block_rmse")


def rotate_hessian(matrix, rotation):
    """Rotate all 3-by-3 blocks without building a dense Kronecker matrix."""
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] % 3:
        raise ValueError("Hessian rotation requires B-by-3N-by-3N matrices")
    n = matrix.shape[-1] // 3
    rotation = rotation.to(device=matrix.device, dtype=matrix.dtype)
    return torch.einsum("ab,kibjd,cd->kiajc", rotation,
                        matrix.reshape(-1, n, 3, n, 3), rotation).reshape_as(matrix)


def rotation_consistency_loss(first, second, first_rotation, second_rotation,
                              loss_type="block_rmse", masses=None):
    """Compare two predictions in the second frame, retaining both gradients."""
    relative = second_rotation @ first_rotation.T
    return hessian_error_loss(second - rotate_hessian(first, relative), loss_type, masses)


def hessian_error_loss(error, loss_type="mse", masses=None):
    """Average matrix errors, or rotation-invariant RMSE of each atom-pair block.

    Block RMSE includes all diagonal and off-diagonal blocks. Each geometry has
    equal weight. vector_norm gives a finite zero gradient at exact agreement.
    """
    if loss_type == "mse":
        return error.square().mean()
    if loss_type == "mae":
        return error.abs().mean()
    if loss_type in ("mass_weighted_mse", "mass_weighted_block_rmse"):
        if (error.ndim != 3 or error.shape[-1] != error.shape[-2]
                or error.shape[-1] % 3 or masses is None
                or masses.shape != (error.shape[0], error.shape[-1] // 3)):
            raise ValueError("Mass-weighted MSE requires 3N-by-3N matrices and B-by-N atomic masses")
        weights = masses.to(device=error.device, dtype=error.dtype).rsqrt().repeat_interleave(3, dim=-1)
        # Weight the difference, which equals the difference of weighted Hessians.
        weighted_error = error * weights[:, :, None] * weights[:, None, :]
        if loss_type == "mass_weighted_mse":
            return weighted_error.square().mean()
        return hessian_error_loss(weighted_error, "block_rmse")
    if loss_type != "block_rmse":
        raise ValueError(f"Hessian loss must be one of {LOSS_TYPES}")
    if error.ndim != 3 or error.shape[-1] != error.shape[-2] or error.shape[-1] % 3:
        raise ValueError("Block RMSE requires a batch of square 3N-by-3N matrices")
    n = error.shape[-1] // 3
    blocks = error.reshape(error.shape[0], n, 3, n, 3)
    # Reduce only Cartesian axes. The result contains one norm per atom pair.
    block_rmse = torch.linalg.vector_norm(blocks, dim=(2, 4)) / 3
    return block_rmse.mean(dim=(1, 2)).mean()
