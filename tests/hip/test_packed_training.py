import copy

import numpy as np
import pytest
import torch

from hip_orb.data import unpack_hessian
from hip_orb.losses import hessian_error_loss
from hip_orb.model import Features, HessianHead
from hip_orb.packed_training import (
    collate_packed,
    packed_blocks,
    packed_indices,
    packed_loss,
    prepare_packed,
)


@pytest.mark.parametrize("sizes", [(1, 2, 5), (3, 3, 3)])
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("loss_type", ["block_rmse", "mse", "mae"])
def test_packed_predictions_loss_and_gradients(sizes, project, loss_type):
    torch.manual_seed(17)
    total = sum(sizes)
    feature = Features(
        torch.randn(total, 4, dtype=torch.float64),
        torch.randn(0, 4, dtype=torch.float64),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.randn(total, 3, dtype=torch.float64),
        torch.randn(total, dtype=torch.float64),
        torch.tensor(sizes),
    )
    head = HessianHead(4, 8, neighbor_pooling=True, translation_projection=project).double()
    reference = copy.deepcopy(head)
    model = type("Model", (), {"head": head})()
    indices = packed_indices(sizes)
    actual = packed_blocks(model, feature, indices, torch.tensor(sizes))
    feature2 = copy.deepcopy(feature)
    del feature2.packed_layout
    matrices = reference(feature2)
    expected = torch.cat(
        [
            m.reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1, 3, 3)
            for n, m in zip(sizes, matrices, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    targets = [torch.randn(3 * n, 3 * n, dtype=torch.float64) for n in sizes]
    target = torch.cat(
        [
            m.reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1, 3, 3)
            for n, m in zip(sizes, targets, strict=True)
        ]
    )
    loss = packed_loss(actual, target, indices, loss_type)
    ref_loss = torch.stack(
        [
            hessian_error_loss((p - t)[None], loss_type)
            for p, t in zip(matrices, targets, strict=True)
        ]
    ).mean()
    torch.testing.assert_close(loss, ref_loss, atol=1e-7, rtol=1e-7)
    loss.backward()
    ref_loss.backward()
    for p, q in zip(head.parameters(), reference.parameters(), strict=True):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-7)


def test_upper_indices_and_rotation_match_equal_size_path():
    samples = []
    for n in (1, 3, 7):
        samples.append(
            {
                "coords": np.random.default_rng(n).normal(size=(n, 3)).astype("float32"),
                "atomic_numbers": np.ones(n, dtype="int16"),
                "hessian_upper": np.arange(3 * n * (3 * n + 1) // 2, dtype="float32"),
                "charge": np.array(0),
                "spin": np.array(1),
                "omol_index": np.array(n),
            }
        )
    batch = prepare_packed(collate_packed(samples))
    blocks = batch["hessian_upper"][batch["indices"]["upper_ids"]].reshape(-1, 3, 3)
    ref = torch.cat(
        [
            unpack_hessian(torch.from_numpy(s["hessian_upper"]), len(s["coords"]))
            .reshape(len(s["coords"]), 3, len(s["coords"]), 3)
            .permute(0, 2, 1, 3)
            .reshape(-1, 3, 3)
            for s in samples
        ]
    )
    torch.testing.assert_close(blocks, ref, atol=0, rtol=0)
    r = batch["rotation"]
    torch.testing.assert_close(r @ r.T, torch.eye(3), atol=1e-6, rtol=1e-6)
    assert torch.linalg.det(r) > 0


@pytest.mark.parametrize(
    "sizes", [(1,), (1, 2, 5), (20, 20), (70, 3, 350, 20), (1, 40, 70, 150, 250, 350)]
)
def test_vectorized_device_indices_match_reference(sizes):
    from hip_orb.packed_training import device_indices

    actual = device_indices(torch.tensor(sizes), "cpu")
    expected = packed_indices(sizes)
    for key in expected:
        if key == "dense_sizes":
            assert expected[key] == actual[key]
        else:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-8, rtol=1e-7)


def test_precomputed_mixed_plan_covers_every_geometry_and_respects_budget():
    from hip_orb.benchmark_capacity import cohort_batches
    from hip_orb.benchmark_packed import coverage, mixed_batches

    groups = [(20, list(range(100))), (70, list(range(100, 160))), (350, list(range(160, 165)))]
    batches = mixed_batches(groups, batch_size=128, limit=4_000_000)
    assert coverage(batches) == coverage(cohort_batches(groups, 128, 4_000_000))
    sizes = {i: n for n, ids in groups for i in ids}
    assert len([i for b in batches for i in b]) == 165
    assert any(len({sizes[i] for i in b}) > 1 for b in batches)
    for b in batches:
        assert len(b) <= 128 and sum(9 * sizes[i] ** 2 for i in b) <= 4_000_000


def test_segmented_projection_is_self_adjoint_and_has_the_correct_backward():
    from hip_orb.packed_training import project_packed

    sizes = torch.tensor([5, 1, 3])
    indices = packed_indices(tuple(sizes.tolist()))
    torch.manual_seed(29)
    x = torch.randn(35, 3, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x)
    px = project_packed(x, indices, sizes)
    py = project_packed(y, indices, sizes)
    torch.testing.assert_close((px * y).sum(), (x * py).sum(), atol=1e-12, rtol=1e-12)
    (px * y).sum().backward()
    torch.testing.assert_close(x.grad, py, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(project_packed(px, indices, sizes), px, atol=1e-12, rtol=1e-12)
