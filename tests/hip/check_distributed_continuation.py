"""Compare a real packed update with two ranks before continuing from the checkpoint.

Launch with torchrun and the normal runner arguments. The probe restores model,
optimizer and RNG states before consuming the first continuation batch.
"""
import copy
import json
import os

import torch

from hip_orb import distributed_train as dt
from hip_orb.mixed_precision import invalidate_trainable_precision_caches
from hip_orb.packed_consistency import packed_consistency_update

_original_update = dt._distributed_consistency_update
_verified = False


def verify_and_update(model, optimizer, batch, *args, **kwargs):
    global _verified
    if not _verified:
        _verified = True
        sample = dt._slice_batch(batch, list(range(min(2, len(batch["sizes"])))))
        head = copy.deepcopy(model.head.state_dict())
        state = copy.deepcopy(optimizer.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        model.head.checkpoint_pair_activations = True

        def restore():
            model.head.load_state_dict(head)
            # load_state_dict may retain tensor storage; each probe needs a fresh copy.
            optimizer.load_state_dict(copy.deepcopy(state))
            invalidate_trainable_precision_caches(model)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)

        expected_loss, _ = packed_consistency_update(model, optimizer, sample, *args, **kwargs)
        expected_grads = [None if p.grad is None else p.grad.clone() for p in model.head.parameters()]
        expected_head = copy.deepcopy(model.head.state_dict())
        expected_opt = copy.deepcopy(optimizer.state_dict())
        restore()
        actual_loss, _ = _original_update(model, optimizer, sample, *args, **kwargs)
        differences = []
        for (name, parameter), expected in zip(model.head.named_parameters(), expected_grads, strict=True):
            assert (parameter.grad is None) == (expected is None), name
            if expected is not None:
                delta = (parameter.grad - expected).float()
                differences.append({"name": name, "max_abs": float(delta.abs().max()),
                                    "relative_norm": float(delta.norm() / expected.float().norm().clamp_min(1e-20))})
        print(json.dumps({"event": "equivalence_diagnostics", "rank": int(os.environ["RANK"]),
                          "reference_loss": expected_loss, "distributed_loss": actual_loss,
                          "gradients": sorted(differences, key=lambda row: row["max_abs"], reverse=True)[:12]}), flush=True)
        for parameter, expected in zip(model.head.parameters(), expected_grads, strict=True):
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected, atol=3e-6, rtol=3e-4)
        assert abs(actual_loss - expected_loss) < max(1e-6, abs(expected_loss) * 2e-4)
        for name, expected in expected_head.items():
            torch.testing.assert_close(model.head.state_dict()[name], expected, atol=3e-6, rtol=3e-4)
        for key, values in expected_opt["state"].items():
            for name, expected in values.items():
                if torch.is_tensor(expected):
                    torch.testing.assert_close(optimizer.state_dict()["state"][key][name], expected, atol=3e-6, rtol=3e-4)
        print(json.dumps({"event": "real_two_rank_equivalence_pass", "rank": int(os.environ["RANK"]),
                          "loss": actual_loss, "peak_memory_bytes": torch.cuda.max_memory_allocated()}), flush=True)
        restore()
        dt._completed_updates = 0
    return _original_update(model, optimizer, batch, *args, **kwargs)


dt._distributed_consistency_update = verify_and_update
if __name__ == "__main__":
    dt.main()
