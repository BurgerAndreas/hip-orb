import h5py
import numpy as np
import torch
from pathlib import Path

from hip_orb.data import OrbMolDataset, AtomBatchSampler, build_index, make_loader
from hip_orb.benchmark_data import prepare_cache


def fixture(tmp_path):
    paths = []
    for shard, n in enumerate((2, 2, 3)):
        path = tmp_path / f"shard_{shard:05d}.h5"
        count, dim = 5, 3 * n
        width = dim * (dim + 1) // 2
        with h5py.File(path, "w") as f:
            f.attrs.update(complete=True, model="orbmol_v2", source_dataset="test")
            f.create_dataset("natoms", data=np.full(count, n))
            f.create_dataset("atom_ptr", data=np.arange(count + 1) * n)
            f.create_dataset("hessian_ptr", data=np.arange(count + 1) * width)
            f.create_dataset("hessian_upper", data=np.arange(count * width, dtype=np.float32),
                             compression="gzip", shuffle=True)
            f.create_dataset("atomic_numbers", data=np.ones(count * n, dtype=np.int16))
            f.create_dataset("coords", data=np.zeros((count * n, 3), dtype=np.float32))
            f.create_dataset("forces", data=np.ones((count * n, 3), dtype=np.float32))
            for key in ("energy", "charge", "spin"):
                f.create_dataset(key, data=np.arange(count))
            f.create_dataset("omol_index", data=np.arange(count) + shard * count)
        paths.append(path)
    index = tmp_path / "index.npz"
    build_index(paths, index)
    return index


def test_sampler_coverage_budget_and_epoch(tmp_path):
    dataset = OrbMolDataset(fixture(tmp_path))
    for order in ("random", "shard"):
        sampler = AtomBatchSampler(dataset, order=order, batch_size=4, max_hessian_elements=200)
        batches = list(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(15))
        for batch in batches:
            sizes = [dataset[i]["coords"].shape[0] for i in batch]
            assert len(set(sizes)) == 1
            assert len(batch) * (3 * sizes[0])**2 <= 200
        sampler.set_epoch(1)
        assert list(sampler) != batches
    ranks = [list(AtomBatchSampler(dataset, rank=r, world_size=2)) for r in (0, 1)]
    owned = [{i for batch in batches for i in batch} for batches in ranks]
    assert not owned[0] & owned[1]
    assert owned[0] | owned[1] == set(range(15))
    dataset.close()


def test_cached_loading_and_spawned_worker(tmp_path):
    index = fixture(tmp_path)
    datasets = [OrbMolDataset(index, cache_mode=mode, cache_bytes=1000, max_open=1)
                for mode in ("none", "chunks", "arrays")]
    for i in (0, 7, 14, 3):
        for dataset in datasets[1:]:
            for key in datasets[0][i]:
                np.testing.assert_array_equal(dataset[i][key], datasets[0][i][key])
            assert dataset._bytes <= 1000
    loader = make_loader(datasets[-1], workers=1, pin_memory=False)
    assert sorted(i for batch in loader for i in batch["omol_index"].tolist()) == list(range(15))
    loader._iterator._shutdown_workers()
    for dataset in datasets:
        dataset.close()


def test_zstd_and_memory_mapped_cache_match_source(tmp_path):
    import pytest
    pytest.importorskip("hdf5plugin")
    index = fixture(tmp_path)
    with np.load(index) as original:
        content = {key: original[key].copy() for key in original.files}
    source = OrbMolDataset(index)
    for storage in ("zstd", "mmap"):
        backend = "mmap" if storage == "mmap" else "hdf5"
        paths = prepare_cache([Path(p) for p in source.paths],
                              tmp_path / storage, backend)
        new_index = tmp_path / f"{storage}.npz"
        np.savez(new_index, **(content | {"paths": np.asarray([str(p) for p in paths])}))
        restored = OrbMolDataset(new_index, backend=backend)
        for row in range(len(source)):
            for key, value in source[row].items():
                np.testing.assert_array_equal(restored[row][key], value)
        restored.close()
    source.close()


def test_bounded_pilot_reads_packed_nested_shards(tmp_path):
    from hip_orb.train import read_samples
    from pathlib import Path
    index = fixture(tmp_path)
    nested = tmp_path / "000"
    nested.mkdir()
    with np.load(index) as original:
        for path in original["paths"]:
            Path(path).rename(nested / Path(path).name)
    samples = read_samples(tmp_path, 2, 2, 2)
    assert len(samples) == 2
    assert samples[0][1].shape == (6, 6)
    np.testing.assert_array_equal(samples[0][1], samples[0][1].T)


def test_training_payload_and_bounded_worker_exhaustion(tmp_path):
    dataset = OrbMolDataset(fixture(tmp_path), training_only=True)
    sample = dataset[0]
    assert "energy" not in sample and "forces" not in sample
    assert "hessian_upper" in sample and "coords" in sample
    loader = make_loader(dataset, workers=1, max_batches=2, batch_size=2, pin_memory=False)
    first = list(loader)
    assert len(first) == 2
    loader.batch_sampler.set_epoch(1)
    assert len(list(loader)) == 2
    loader._iterator._shutdown_workers()
    dataset.close()


def test_preparation_runs_in_spawned_workers(tmp_path):
    from hip_orb.preparation import PreparedCollator
    dataset = OrbMolDataset(fixture(tmp_path), training_only=True)
    reference = make_loader(dataset, workers=0, pin_memory=False, max_batches=2,
                            collate_fn=PreparedCollator())
    expected = list(reference)
    spawned = make_loader(dataset, workers=1, pin_memory=False, max_batches=2,
                          collate_fn=PreparedCollator())
    try:
        actual = list(spawned)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left["rotation"], right["rotation"], atol=0, rtol=0)
            torch.testing.assert_close(left["omol_index"], right["omol_index"], atol=0, rtol=0)
            assert len(left["molecules"]) == len(left["coords"])
    finally:
        spawned._iterator._shutdown_workers()
        dataset.close()


def test_benchmark_stratification_covers_all_sizes(tmp_path):
    dataset = OrbMolDataset(fixture(tmp_path), training_only=True)
    batches = list(make_loader(dataset, stratified=True, batch_size=2, pin_memory=False))
    expected_sizes = sorted(set(dataset.natoms.tolist()))
    assert [batch["coords"].shape[1] for batch in batches[:len(expected_sizes)]] == expected_sizes
    assert sorted(index for batch in batches for index in batch["omol_index"].tolist()) == list(range(15))
    dataset.close()
