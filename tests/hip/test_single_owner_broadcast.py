"""Conditioning broadcasts retain parameters, losses, gradients and Adam state."""
import copy

import pytest
import torch

from hip_orb.model import Features, HessianHead, _conditioning_rows


@pytest.mark.parametrize("sizes", [(1,), (5,), (2, 3)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_complete_head_updates(sizes, checkpoint):
    torch.manual_seed(19)
    n = sum(sizes)
    f = Features(torch.randn(n, 4), torch.empty(0, 4),
                 torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                 torch.randn(n, 3), torch.randn(n), torch.tensor(sizes),
                 total_charge=torch.zeros(len(sizes)), spin_multiplicity=torch.ones(len(sizes)))
    reference = HessianHead(4, 8, readout_conditioning=True)
    torch.nn.init.normal_(reference.readout_conditioner[-1].weight, std=0.05)
    candidate = copy.deepcopy(reference)
    candidate.single_owner_broadcast = True
    for h in (reference, candidate):
        h.checkpoint_context = checkpoint
    opts = [torch.optim.AdamW(h.parameters(), lr=1e-3) for h in (reference, candidate)]
    assert reference.config == candidate.config
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for _ in range(3):
        losses = []
        for head, opt in zip((reference, candidate), opts):
            opt.zero_grad(set_to_none=True)
            loss = sum(x.square().mean() for x in head(f, residual_only=True))
            loss.backward()
            losses.append(loss.detach())
            opt.step()
        torch.testing.assert_close(*losses, atol=1e-6, rtol=1e-4)
        for a, b in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
            for key in opts[0].state[a]:
                torch.testing.assert_close(opts[0].state[a][key], opts[1].state[b][key], atol=1e-6, rtol=1e-4)


def test_multiple_owners_keep_original_indexing_and_empty_rows():
    head = type("Head", (), {"single_owner_broadcast": True})()
    value = torch.randn(2, 8)
    ids = torch.tensor([1, 0, 1])
    assert torch.equal(_conditioning_rows(head, value, ids, 2), value[ids])
    assert _conditioning_rows(head, value[:1], ids[:0], 1).shape == (0, 8)
