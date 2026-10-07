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
    assert all(parameter.grad is None for parameter in model.base.parameters())
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


def test_real_coulomb_matches_upstream_derivative_without_labels():
    from ase.build import molecule

    from hip_orb.coulomb import fixed_charge_hessian
    from hip_orb.model import HIPOrb

    model = HIPOrb.pretrained(precision="float32-highest")
    atoms = molecule("H2O")
    atoms.info.update(charge=1, spin=2)
    graph = model.adapter.from_ase_atoms_list([atoms], device="cuda")
    feature = model.features(graph)
    # Hold the model's latent charges fixed while differentiating the original kernel.
    positions = feature.positions.double()
    charges = feature.charges.double()
    module = model.base.coulomb_module

    def energy(x):
        return module._direct_coulomb(charges, x, graph.node_batch_index, graph.n_node, 1).sum()

    original = torch.autograd.functional.hessian(energy, positions).reshape(9, 9)
    analytical = fixed_charge_hessian(positions, charges, feature.constant, feature.sigma)
    torch.testing.assert_close(analytical, original, atol=1e-10, rtol=1e-10)
    feature.nodes.requires_grad_(False)
    assert not feature.nodes.requires_grad and not feature.charges.requires_grad
