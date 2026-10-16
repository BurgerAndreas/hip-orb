import pytest
import torch

from hip_orb.preparation import prepare_batch


def batch(b, n):
    torch.manual_seed(13)
    return {"coords": torch.randn(b, n, 3) * 3,
            "atomic_numbers": torch.randint(1, 10, (b, n), dtype=torch.int16),
            "charge": torch.arange(b) % 2, "spin": torch.ones(b, dtype=torch.long)}


def test_tensor_preparation_preserves_exact_rotations_and_coordinates():
    for scale in (0.0, 1.0):
        raw = batch(3, 7)
        expected = prepare_batch(raw, seed=3, translation_scale=scale)
        actual = prepare_batch(raw, seed=3, translation_scale=scale, include_molecules=False)
        assert "molecules" not in actual
        torch.testing.assert_close(actual["rotation"], expected["rotation"], atol=0, rtol=0)
        torch.testing.assert_close(actual["translation"], expected["translation"], atol=0, rtol=0)
        for a, b in zip(actual["rotated_coords"], expected["molecules"], strict=True):
            torch.testing.assert_close(a.double(), torch.from_numpy(b.positions), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA molecular graph check")
@pytest.mark.parametrize("b,n", [(1, 20), (128, 2), (128, 20), (90, 70), (19, 150), (3, 350)])
def test_tensor_graph_has_identical_orb_edges_and_features(b, n):
    from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
    from hip_orb.tensor_graph import molecular_training_graph
    from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
    adapter = ForcefieldAtomsAdapter(radius=6., max_num_neighbors=120)
    raw = batch(b, n)
    reference = prepare_batch(raw, seed=7)
    tensor_batch = prepare_batch(raw, seed=7, include_molecules=False)
    expected = refresh_message_vectors(canonical_message_graph(
        adapter.from_ase_atoms_list(reference["molecules"], device="cuda")))
    actual = refresh_message_vectors(canonical_message_graph(
        molecular_training_graph(tensor_batch, adapter, "cuda")))
    for name in ("senders", "receivers", "n_node", "n_edge", "node_batch_index", "tags", "max_num_neighbors"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
    assert actual.batch_total_n_node == expected.batch_total_n_node
    for name in ("node_features", "edge_features", "system_features"):
        a, b = getattr(actual, name), getattr(expected, name)
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)
