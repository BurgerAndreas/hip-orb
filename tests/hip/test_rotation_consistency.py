from types import SimpleNamespace

import pytest
import torch

from hip_orb.losses import rotate_hessian, rotation_consistency_loss
from hip_orb.benchmark_training import training_step


@pytest.mark.parametrize("loss_type", ["mse", "mae", "block_rmse", "mass_weighted_mse"])
def test_two_frame_consistency_and_both_gradients(loss_type):
    generator = torch.Generator().manual_seed(42)
    first_rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    second_rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    matrix = torch.randn(2, 6, 6, generator=generator, dtype=torch.float64)
    first = rotate_hessian(matrix, first_rotation).requires_grad_()
    exact_second = rotate_hessian(matrix, second_rotation)
    masses = torch.tensor([[1., 12.], [12., 16.]], dtype=torch.float64)
    exact = rotation_consistency_loss(first, exact_second, first_rotation,
                                     second_rotation, loss_type, masses)
    assert float(exact.detach()) < 1e-12
    second = (exact_second + 0.5).detach().requires_grad_()
    loss = rotation_consistency_loss(first, second, first_rotation, second_rotation,
                                    loss_type, masses)
    loss.backward()
    for gradient in (first.grad, second.grad):
        assert torch.isfinite(gradient).all()
        assert float(gradient.norm()) > 0


def test_consistency_optimization_reduces_rotation_error():
    # An orientation-independent anisotropic matrix violates equivariance.
    matrix = torch.nn.Parameter(torch.diag(torch.tensor([1., 2., 4.])))
    rotation = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    optimizer = torch.optim.SGD([matrix], lr=0.5)
    def objective():
        prediction = matrix.unsqueeze(0)
        return rotation_consistency_loss(prediction, prediction, torch.eye(3), rotation, "mse")
    initial = float(objective().detach())
    for _ in range(30):
        optimizer.zero_grad()
        objective().backward()
        optimizer.step()
    assert float(objective().detach()) < initial * 0.001


@pytest.mark.parametrize("loss_type", ["mse", "mae", "block_rmse", "mass_weighted_mse"])
@pytest.mark.parametrize("consistency_weight", [0., 0.1])
@pytest.mark.parametrize("rotation_sampling", ["uniform", "legacy"])
def test_complete_two_view_update(loss_type, consistency_weight, rotation_sampling):
    calls = []
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.diag(torch.arange(1., 7.)))
            self.second = torch.nn.Parameter(torch.eye(6))
        def forward(self, features):
            calls.append("head")
            return [self.first if len(calls) == 1 else self.second]
    model = SimpleNamespace(head=Head(), adapter=SimpleNamespace(
        from_ase_atoms_list=lambda molecules, device: molecules), features=lambda graph: graph)
    batch = {"atomic_numbers": torch.tensor([[1, 6]]),
             "coords": torch.tensor([[[0., 0., 0.], [0., 0., 1.]]]),
             "charge": torch.zeros(1, dtype=torch.long), "spin": torch.ones(1, dtype=torch.long),
             "hessian_upper": torch.zeros(1, 21)}
    initial = {k: p.detach().clone() for k, p in model.head.named_parameters()}
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)
    result = training_step(model, optimizer, batch, device="cpu", loss_type=loss_type,
                           consistency_weight=consistency_weight, consistency_seed=71,
                           translation_scale=1.0, return_components=True,
                           rotation_sampling=rotation_sampling, reference_views=2)
    assert len(calls) == 2
    torch.testing.assert_close(result["total"], result["reference"] + consistency_weight * result["consistency"])
    assert result["consistency"] > 0
    for name, parameter in model.head.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter, initial[name])


def test_consistency_rejects_single_view():
    with pytest.raises(ValueError, match="two reference views"):
        training_step(None, None, {}, consistency_weight=0.1, reference_views=1)
