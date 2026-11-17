"""Uncapped radius membership, Hessian values, and training derivatives."""

import copy
from types import SimpleNamespace

import pytest
import torch

from hip_orb.cutoff_inference import blocks_to_dense, cutoff_forward, cutoff_hessian
from hip_orb.model import Features, HessianHead
from hip_orb.neighbor_search import AsyncCPUPairs, RadiusPairs, neighbor_pairs


def keys(pairs, n):
    i, j = pairs
    assert i.dtype == j.dtype == torch.int32
    assert bool((i < j).all())
    result = (i.long() * n + j).sort().values
    assert len(result.unique()) == len(result)
    return result.cpu()


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sizes", [[], [0], [1], [37], [0, 17, 0, 23]])
@pytest.mark.parametrize("cutoff", [0.2, 2.0, 100.0])
def test_membership_and_chunks(device, sizes, cutoff):
    torch.manual_seed(71)
    p = torch.randn(sum(sizes), 3, device=device) * 3 - 4
    reference = keys(neighbor_pairs(p, sizes, cutoff, backend="brute", rows=7), len(p))
    for backend in ["cpu"] + (["gpu"] if device == "cuda" else []):
        plan = RadiusPairs(p, sizes, cutoff, backend, rows=5)
        torch.testing.assert_close(keys(plan.materialize(), len(p)), reference, atol=0, rtol=0)
        chunks = list(plan.chunks(97))
        assert all(len(i) <= 97 for i, _ in chunks)
        if chunks:
            pairs = tuple(torch.cat([c[k] for c in chunks]) for k in range(2))
            torch.testing.assert_close(keys(pairs, len(p)), reference, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_strict_boundary_coincident_points_and_molecule_separation(device):
    one = torch.tensor(1.0)
    below = torch.nextafter(one, torch.tensor(0.0))
    above = torch.nextafter(one, torch.tensor(2.0))
    p = torch.tensor(
        [[0, 0, 0], [below, 0, 0], [1, 0, 0], [above, 0, 0], [0, 0, 0], [0, 0, 0]], device=device
    )
    expected = keys(neighbor_pairs(p, [5, 1], 1.0, backend="brute"), len(p))
    assert 1 in expected and 2 not in expected and 3 not in expected and 4 in expected
    for backend in ["cpu"] + (["gpu"] if device == "cuda" else []):
        torch.testing.assert_close(
            keys(neighbor_pairs(p, [5, 1], 1, backend=backend), len(p)), expected, atol=0, rtol=0
        )


def test_invalid_inputs_and_async_snapshot():
    p = torch.randn(12, 3)
    for sizes, cutoff in [([11], 1), ([12], 0), ([12], float("inf"))]:
        with pytest.raises(ValueError):
            neighbor_pairs(p, sizes, cutoff)
    expected = keys(neighbor_pairs(p, [12], 2, backend="cpu"), len(p))
    with AsyncCPUPairs(p, [12], 2) as work:
        p.fill_(100)
        torch.testing.assert_close(keys(work.result("cpu"), len(p)), expected, atol=0, rtol=0)


def test_cpu_row_fallback(monkeypatch):
    import hip_orb.neighbor_search as search

    p = torch.randn(51, 3)
    expected = keys(neighbor_pairs(p, [20, 31], 2, backend="cpu"), len(p))
    monkeypatch.setattr(search, "CPU_PAIR_ARRAY_BUDGET", 0)
    actual = keys(neighbor_pairs(p, [20, 31], 2, backend="cpu", rows=7), len(p))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_cuda_three_coordinate_boundary_order(device):
    # Raw deltas from six real protein pairs, all mathematically inside 30 A.
    # Summing squared x,y before z incorrectly rounds their distances to 30.
    delta = torch.tensor(
        [
            [28.7249984741, -5.5279998779, -6.6569995880],
            [-22.9899902344, 0.1079978943, -19.2730026245],
            [3.8180007935, -23.0279998779, -18.8450012207],
            [9.5889892578, 25.4200019836, 12.7229995728],
            [-22.9639892578, 10.1200027466, 16.4389991760],
            [-22.9899902344, 0.1079978943, -19.2730026245],
        ],
        device=device,
    )
    p = torch.stack((torch.zeros_like(delta), delta), 1).reshape(-1, 3)
    expected = torch.arange(0, len(p), 2, dtype=torch.int64) * len(p) + torch.arange(1, len(p), 2)
    for backend in ["brute", "cpu"] + (["gpu"] if device == "cuda" else []):
        torch.testing.assert_close(
            keys(neighbor_pairs(p, [2] * 6, 30, backend=backend), len(p)), expected, atol=0, rtol=0
        )


def fixture_model(device):
    torch.manual_seed(91)
    n, latent = 17, 4
    i, j = torch.triu_indices(n, n, 1, device=device)
    senders, receivers = torch.cat((i, j)), torch.cat((j, i))
    features = Features(
        torch.randn(n, latent, device=device),
        torch.randn(len(senders), latent, device=device),
        senders,
        receivers,
        torch.randn(n, 3, device=device) * 2,
        torch.randn(n, device=device),
        torch.tensor([n], device=device),
    )
    head = HessianHead(latent, 8, cutoff=3, neighbor_pooling=True).to(device)
    return SimpleNamespace(head=head, features=lambda graph: features), features


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [7, 100000])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_hessian_and_parameter_gradients(device, chunk, checkpoint):
    reference, features = fixture_model(device)
    expected = cutoff_forward(reference, None, 3, pair_chunk=chunk, search="brute")
    loss = expected["blocks"].square().sum() + expected["diagonal"].square().sum()
    loss.backward()
    for backend in ["cpu"] + (["gpu"] if device == "cuda" else []):
        actual = SimpleNamespace(
            head=copy.deepcopy(reference.head), features=lambda graph: features
        )
        actual.head.zero_grad(set_to_none=True)
        result = cutoff_forward(
            actual, None, 3, pair_chunk=chunk, search=backend, checkpoint_chunks=checkpoint
        )
        torch.testing.assert_close(
            blocks_to_dense(result, len(features.nodes)),
            blocks_to_dense(expected, len(features.nodes)),
            atol=2e-5,
            rtol=2e-5,
        )
        (result["blocks"].square().sum() + result["diagonal"].square().sum()).backward()
        for a, b in zip(actual.head.parameters(), reference.head.parameters(), strict=True):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=5e-4, rtol=5e-4)
    with AsyncCPUPairs(features.positions.cpu(), [len(features.nodes)], 3) as work:
        result = cutoff_hessian(reference, None, 3, pair_chunk=chunk, pair_future=work)
    torch.testing.assert_close(
        blocks_to_dense(result, len(features.nodes)),
        blocks_to_dense(expected, len(features.nodes)),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_block_sink_and_dense_head_reference(device, backend):
    if backend == "gpu" and device != "cuda":
        pytest.skip("GPU sink needs CUDA")
    model, features = fixture_model(device)
    expected = cutoff_hessian(model, None, 100, residual_only=True, search="brute")
    # With every pair kept, the existing dense head is an independent oracle.
    dense = model.head(features, residual_only=True)[0]
    torch.testing.assert_close(
        blocks_to_dense(expected, len(features.nodes)), dense, atol=2e-5, rtol=2e-5
    )
    parts = []
    with torch.no_grad():
        result = cutoff_forward(
            model,
            None,
            100,
            pair_chunk=11,
            search=backend,
            residual_only=True,
            block_sink=lambda i, j, b: parts.append((i.clone(), j.clone(), b.clone())),
        )
    assembled = {
        "i": torch.cat([p[0] for p in parts]),
        "j": torch.cat([p[1] for p in parts]),
        "blocks": torch.cat([p[2] for p in parts]),
        "diagonal": result["diagonal"],
    }
    torch.testing.assert_close(
        blocks_to_dense(assembled, len(features.nodes)), dense, atol=2e-5, rtol=2e-5
    )
