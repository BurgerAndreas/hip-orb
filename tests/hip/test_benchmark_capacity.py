from types import SimpleNamespace

import numpy as np
import pytest

from hip_orb.benchmark_capacity import (
    capacity,
    cohort_batches,
    cohort_groups,
    fingerprint,
    plan_stats,
)


def test_capacity_and_exact_binding_count():
    assert [capacity(n, 128, 4_000_000) for n in (20, 50, 70, 100, 200, 350)] == [
        128,
        128,
        90,
        44,
        11,
        3,
    ]
    stats = plan_stats(np.array([20, 350], dtype=np.int16), [400, 400], 128, 4_000_000)
    assert stats["steps_per_epoch"] == 4 + 134
    assert stats["full_budget_steps_fraction"] == 133 / 138
    assert stats["budget_reduced_steps_fraction"] == 134 / 138
    assert stats["mean_batch"] == 800 / 138


@pytest.mark.parametrize(
    "batch,limit", [(64, 2_000_000), (128, 4_000_000), (256, 8_000_000), (512, 12_000_000)]
)
def test_rebatching_keeps_exact_same_cohort_order_and_coverage(batch, limit):
    dataset = SimpleNamespace(
        paths=["a", "b", "c"], ptr=np.array([0, 400, 420, 820]), natoms=np.array([20, 70, 350])
    )
    groups = cohort_groups(dataset, 3)
    baseline = cohort_batches(groups, 128, 4_000_000)
    actual = cohort_batches(groups, batch, limit)
    assert fingerprint(actual) == fingerprint(baseline)
    flat = [i for part in actual for i in part]
    assert sorted(flat) == list(range(820))
    for part in actual:
        owners = np.searchsorted(dataset.ptr, part, side="right") - 1
        n = dataset.natoms[owners]
        assert (n == n[0]).all()
        assert len(part) <= batch and len(part) * (3 * int(n[0])) ** 2 <= limit


def test_confirmation_settings_reject_ambiguous_or_unsafe_input():
    import argparse

    from hip_orb.benchmark_capacity import parse_setting

    assert parse_setting("256:8000000") == (256, 8_000_000)
    for invalid in ("256", "256:0", "0:4000000", "256:x", "1:2:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_setting(invalid)
