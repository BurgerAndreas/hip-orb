"""Opt-in GPU checks against the exact frozen collection checkpoint."""

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("HIP_ORB_REAL_TESTS") != "1",
    reason="Enable real-checkpoint GPU tests explicitly",
)


def test_real_batch_energy_force_and_training_invariance():
    from ase.build import molecule

    from hip_orb.model import HIPOrb

    model = HIPOrb.pretrained()
    atoms = [molecule("H2O"), molecule("CH4")]
    for item in atoms:
        item.info.update(charge=0, spin=1)

    def graph(items):
        return model.adapter.from_ase_atoms_list(items, device="cuda")

    before = model.base.predict(graph(atoms), split=False)
    parameters = {key: value.detach().clone() for key, value in model.base.state_dict().items()}
    model.train()
    batched = model(graph(atoms))
    for item, predicted in zip(atoms, batched, strict=True):
        torch.testing.assert_close(predicted, predicted.T)
        torch.testing.assert_close(model(graph([item]))[0], predicted, atol=2e-4, rtol=2e-4)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
    optimizer.zero_grad()
    sum(h.square().mean() for h in batched).backward()
    optimizer.step()
    after = model.predict(graph(atoms))
    for key in ("energy", "forces"):
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    for key, value in model.base.state_dict().items():
        torch.testing.assert_close(value, parameters[key], atol=0, rtol=0)

    # Measure full backbone + head latency, not just cached readout latency.
    for _ in range(3):
        with torch.no_grad():
            model(graph(atoms))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            model(graph(atoms))
    torch.cuda.synchronize()
    print(
        {
            "hip_batch_seconds": (time.perf_counter() - started) / 5,
            "hip_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    )
