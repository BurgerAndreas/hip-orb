import numpy as np

from hip_orb.profile_batch_steps import RepeatedPool, read_pool


def test_workers_receive_identical_batches_when_pool_does_not_divide_batch_size():
    pool = [object(), object(), object()]
    dataset = RepeatedPool(pool, batch_size=5, batches=4)
    reference = [pool[0], pool[1], pool[2], pool[0], pool[1]]
    assert len(dataset) == 20
    for batch in range(4):
        assert [dataset[batch * 5 + i] for i in range(5)] == reference


def test_source_pool_uses_only_requested_atom_count_and_stops_at_limit():
    class Source:
        natoms = np.array([20, 50, 20])
        ptr = np.array([0, 3, 6, 9])

        def __getitem__(self, index):
            assert index not in (3, 4, 5)
            return index

    samples, n = read_pool(Source(), requested=21, limit=5)
    assert n == 20
    assert samples == [0, 1, 2, 6, 7]
