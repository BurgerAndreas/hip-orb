"""Check diagnostic statistics without a GPU or external labels."""
import torch

from hip_orb.audit_randomness import difference, envelope, summary


def test_repeat_envelope_uses_every_pair():
    values = [torch.tensor([0., 1.]), torch.tensor([2., 1.]), torch.tensor([-3., 1.])]
    result = envelope(values)
    assert result["max_abs"] == 5
    assert result["failed_1e6_components"] == 1
    assert abs(result["rms"] - 5 / 2**0.5) < 1e-12


def test_equal_outputs_have_no_variation():
    value = torch.tensor([1., 2.])
    assert difference(value, value)["max_abs"] == 0
    assert summary([{"forces": value}, {"forces": value.clone()}])["forces"]["failed_1e6_components"] == 0
