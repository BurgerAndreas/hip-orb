import pytest
import torch

from hip_orb.distributed_train import _microbatches
from hip_orb.packed_training import compact_indices


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (0, [[0], [2], [3], [4], [5]]),
        (32, [[0, 2], [3], [4], [5]]),
        (41, [[0, 2], [3], [4, 5]]),
    ],
)
def test_microbatches_preserve_rank_order_and_keep_oversize_molecules_alone(limit, expected):
    sizes = [4, 5, 4, 12, 4, 5]
    assert list(_microbatches([0, 2, 3, 4, 5], sizes, limit)) == expected
    assert list(_microbatches([], sizes, limit)) == []


def test_microbatch_weights_preserve_global_molecule_weights():
    sizes = [3, 5, 2, 6, 4, 2]
    selected = [0, 2, 4, 5]
    weights = {"pair_weight": [], "diag_weight": []}
    for group in _microbatches(selected, sizes, pair_limit=20):
        group_weights = compact_indices(tuple(sizes[i] for i in group), "cpu")
        for name in weights:
            weights[name].append(group_weights[name] * (len(group) / len(sizes)))
    for name in weights:
        expected = torch.cat([
            compact_indices((sizes[i],), "cpu")[name] / len(sizes)
            for i in selected
        ])
        torch.testing.assert_close(torch.cat(weights[name]), expected)
