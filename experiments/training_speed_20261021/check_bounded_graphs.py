from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_bounded_graphs_replays_changed_data_and_stops_capture():
    from experiments.training_speed_20261021.bounded_graphs import frozen_backbone_graphs

    class Base(torch.nn.Module):
        def forward(self, batch):
            return {"out": batch.node_features["positions"].square() + batch.shift}

    base = Base().cuda().eval()
    model = SimpleNamespace(base=SimpleNamespace(model=base), head=torch.nn.Identity())
    original = base.forward
    with (
        torch.no_grad(),
        frozen_backbone_graphs(model, max_graphs=1, max_nodes=16, capture_after=2) as stats,
    ):
        for i in range(6):
            batch = SimpleNamespace(
                node_features={"positions": torch.full((4, 3), float(i), device="cuda")},
                shift=torch.tensor(float(i + 1), device="cuda"),
            )
            torch.testing.assert_close(base(batch)["out"], original(batch)["out"])
        assert stats["captures"] == 1
        assert stats["replays"] == 4
        batch = SimpleNamespace(
            node_features={"positions": torch.randn(8, 3, device="cuda")},
            shift=torch.tensor(2.0, device="cuda"),
        )
        torch.testing.assert_close(base(batch)["out"], original(batch)["out"])
        assert stats["captures"] == 1
    assert base.forward == original
