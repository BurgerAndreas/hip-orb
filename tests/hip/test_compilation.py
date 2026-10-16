import copy
from types import SimpleNamespace

import pytest
import torch

from hip_orb.compilation import compiler_candidate
from hip_orb.model import directed_pair_inputs, HessianHead, Features


@pytest.mark.parametrize('nedges', [0, 3])
def test_pair_fusion_matches_existing_direction_inputs(nedges):
    torch.manual_seed(4)
    nodes, edges = torch.randn(4, 8), torch.randn(nedges, 8)
    charges = torch.randn(4)
    i, j = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    forward = torch.tensor([-1, 0, 2]) if nedges else torch.full((3,), -1)
    reverse = torch.full((3,), -1)
    delta, distance = torch.randn(3, 3), torch.rand(3, 1) + 0.1
    def reference(a, b, ids, direction):
        padded = torch.cat((edges, nodes.new_zeros(1, 8)))
        present = ids >= 0
        return torch.cat((nodes[a], nodes[b], padded[torch.where(present, ids, nedges)],
                          direction, torch.log1p(distance), charges[a, None], charges[b, None],
                          present[:, None].to(nodes.dtype)), dim=-1)
    expected = torch.cat((reference(i, j, forward, delta/distance),
                          reference(j, i, reverse, -delta/distance)))
    actual = directed_pair_inputs(nodes, edges, charges, i, j, forward, reverse, delta, distance)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compiler_candidate_restores_after_exception(monkeypatch):
    backbone = torch.nn.Linear(2, 2)
    head = HessianHead(latent_dim=2)
    model = SimpleNamespace(base=SimpleNamespace(model=backbone), head=head)
    sentinel = object()
    monkeypatch.setattr(backbone, 'compile', lambda **kwargs: setattr(backbone, '_compiled_call_impl', sentinel))
    monkeypatch.setattr(torch, 'compile', lambda function, **kwargs: sentinel)
    with pytest.raises(RuntimeError):
        with compiler_candidate(model, backbone=True, pair_inputs=True):
            assert backbone._compiled_call_impl is sentinel
            assert head.pair_inputs is sentinel
            raise RuntimeError('test restoration')
    assert backbone._compiled_call_impl is None
    assert head.pair_inputs is directed_pair_inputs


def test_known_conditioning_size_preserves_values():
    from hip_orb.compilation import shape_known_conditioning
    from orb_models.common.models.nn_util import ChargeSpinConditioner
    conditioner = ChargeSpinConditioner(16, emits_node_embs=True, emits_edge_embs=True)
    batch = SimpleNamespace(system_features={'total_charge': torch.tensor([0., 1.]),
        'spin_multiplicity': torch.tensor([0., 2.])}, n_node=torch.tensor([2, 3]),
        n_edge=torch.tensor([1, 4]), node_features={'positions': torch.zeros(5, 3)},
        senders=torch.zeros(5, dtype=torch.long))
    expected = conditioner(batch)
    actual = shape_known_conditioning(conditioner, batch)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
