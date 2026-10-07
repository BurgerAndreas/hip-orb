from types import SimpleNamespace

import pytest
import torch

from hip_orb.graphs import canonical_message_graph, hessian_pair_topology, refresh_message_vectors, deterministic_graph_audit
from hip_orb.model import Features, HessianHead


def test_canonical_edges_keep_features_and_targets_aligned():
    graph = SimpleNamespace(batch_total_n_node=3, senders=torch.tensor([2, 0, 1]),
                            receivers=torch.tensor([0, 2, 0]),
                            edge_features={"vectors": torch.tensor([[20.], [2.], [10.]]), "unit_shifts": torch.zeros(3, 3)},
                            edge_targets={"marker": torch.tensor([20, 2, 10])},
                            node_features={"positions": torch.tensor([[0., 0, 0], [1, 0, 0], [3, 0, 0]])},
                            system_features={"pbc": torch.tensor([[False, False, False]])})
    canonical_message_graph(graph)
    assert graph.senders.tolist() == [0, 1, 2]
    assert graph.receivers.tolist() == [2, 0, 0]
    assert graph.edge_targets["marker"].tolist() == [2, 10, 20]
    assert graph.edge_features["vectors"].flatten().tolist() == [2., 10., 20.]
    refresh_message_vectors(graph)
    assert graph.edge_features["vectors"][:, 0].tolist() == [3., -1., -3.]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("cutoff", [None, 1.5, 1e-6])
def test_cpu_pairs_match_device_mapping_outputs_and_gradients(fused, cutoff):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    # Two molecules, missing message edges, and distinct directed features.
    features = Features(torch.randn(5, 4), torch.randn(3, 4), torch.tensor([0, 1, 4]),
                        torch.tensor([1, 0, 3]), torch.randn(5, 3), torch.randn(5), torch.tensor([3, 2]))
    head = HessianHead(4, 12, cutoff=cutoff)
    head.fuse_pairs = fused
    before = head(features)
    loss = sum(value.square().mean() for value in before)
    loss.backward()
    grads = [None if value.grad is None else value.grad.clone() for value in head.parameters()]
    head.zero_grad(set_to_none=True)
    features.pairs = hessian_pair_topology(features.senders, features.receivers, [3, 2])
    after = head(features)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    sum(value.square().mean() for value in after).backward()
    for value, expected in zip(head.parameters(), grads):
        if expected is None:
            assert value.grad is None
        else:
            torch.testing.assert_close(value.grad, expected, atol=0, rtol=0)
    assert features.pairs[0]["forward"].tolist() == [0, -1, -1]
    assert features.pairs[1]["reverse"].tolist() == [2]


def test_pair_topology_rejects_cross_molecule_edges_and_duplicates():
    with pytest.raises(ValueError, match="cross molecules"):
        hessian_pair_topology(torch.tensor([0]), torch.tensor([2]), [2, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        hessian_pair_topology(torch.tensor([0, 0]), torch.tensor([1, 1]), [2])


def test_one_atom_has_empty_pair_topology():
    pairs = hessian_pair_topology(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), [1])
    assert all(value.numel() == 0 for value in pairs[0].values())


def test_graph_audit_restores_training_settings_on_error():
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="test"):
        with deterministic_graph_audit():
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.are_deterministic_algorithms_enabled()
            raise RuntimeError("test")
    assert torch.get_float32_matmul_precision() == precision
    assert torch.are_deterministic_algorithms_enabled() == deterministic
