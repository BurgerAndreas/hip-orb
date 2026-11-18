"""Exercise original runner checkpoints with real Adam, LR and batch cursor."""
import json
import sys

import numpy as np
import pytest
import torch

from hip_orb import runner


def test_requested_stop_resumes_identical_updates(monkeypatch, tmp_path):
    for name in ("train", "validation"):
        np.savez(tmp_path / f"{name}.npz", paths=np.asarray([str(tmp_path / f"{name}.h5")]),
                 ptr=np.array([0, 8]), natoms=np.array([2]), sources=np.asarray([name]))
    request = tmp_path / "STOP"
    seen, seeds, loaders = [], [], []

    class Model:
        def __init__(self):
            with torch.random.fork_rng():
                torch.manual_seed(31)
                self.base = torch.nn.Linear(1, 1)
                self.head = torch.nn.Linear(1, 1)
                self.head.config = {}
        def train(self):
            pass

    class Loader:
        batch_sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
        def __init__(self, start):
            self.start = start
            self.closed = False
            self._iterator = self
        def __iter__(self):
            return iter({"coords": torch.ones(1, 2, 3), "id": i} for i in range(self.start, 8))
        def _shutdown_workers(self):
            self.closed = True

    def loader(dataset, **kwargs):
        result = Loader(kwargs.get("start_batch", 0))
        loaders.append(result)
        return result

    def update(model, optimizer, batch, device, seed, **kwargs):
        i = batch["id"]
        seen.append(i)
        seeds.append((seed, kwargs["consistency_seed"]))
        optimizer.zero_grad(set_to_none=True)
        x = torch.tensor([[float(i + 1)]])
        loss = (model.head(x) - torch.rand(())).square().mean()
        loss.backward()
        optimizer.step()
        if len(seen) == 2 and should_request[0]:
            request.touch()  # Request arrives during an update, after Adam advances.
        return float(loss.detach()), 1

    monkeypatch.setattr(runner.HIPOrb, "pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(runner, "configure_execution", lambda *a, **k: {})
    monkeypatch.setattr(runner, "make_loader", loader)
    monkeypatch.setattr(runner, "train_update", update)
    monkeypatch.setattr(runner, "validate", lambda *a: {"validation": {}, "validation_samples": 0})
    should_request = [False]

    def run(name, checkpoint=None):
        argv = ["train", "--train-index", str(tmp_path / "train.npz"),
                "--validation-index", str(tmp_path / "validation.npz"), "--device", "cpu",
                "--output", str(tmp_path / name), "--max-steps", "6", "--lr-schedule", "cosine",
                "--stop-request-file", str(request)]
        if checkpoint:
            argv += ["--checkpoint", str(checkpoint)]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        return torch.load(tmp_path / name / "head.pt", weights_only=True)

    torch.manual_seed(18)
    continuous = run("continuous")
    expected_seeds = list(seeds)
    seen.clear(); seeds.clear()
    torch.manual_seed(18)
    should_request[0] = True
    stopped = run("paused")
    report = json.loads((tmp_path / "paused/report.json").read_text())
    assert stopped["global_step"] == stopped["epoch_batches"] == 2
    assert report["stop_reason"] == "requested" and not report["auto_continue"]
    assert not report["training_complete"] and not report["final_validation_completed"]
    assert request.exists() and all(loader.closed for loader in loaders)
    request.unlink()
    torch.manual_seed(999)  # Restore must use checkpoint RNG, not process startup RNG.
    resumed = run("resumed", tmp_path / "paused/head.pt")
    assert seen == list(range(6)) and seeds == expected_seeds
    assert resumed["global_step"] == resumed["epoch_batches"] == 6
    for key in ("head", "optimizer", "lr_schedule_config", "batching_config", "augmentation_config", "seed", "torch_rng_state"):
        def compare(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a: compare(a[k], b[k])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b): compare(x, y)
            else:
                assert a == b
        compare(continuous[key], resumed[key])


def test_existing_request_is_latched(tmp_path):
    from hip_orb.session_stop import UpdateBoundary
    path = tmp_path / "STOP"
    path.touch()
    boundary = UpdateBoundary(path)
    assert boundary.reached(False)
    path.unlink()
    assert boundary.reached(False)


@pytest.mark.parametrize("reason,expect_successor", [("requested", False), ("session_limit", True)])
def test_continuation_launcher_honors_manual_pause(tmp_path, reason, expect_successor):
    import os
    from pathlib import Path
    import subprocess

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    original = Path(__file__).parents[2] / "scripts/hip_train_steps.sbatch"
    (scripts / original.name).write_text(original.read_text())
    (scripts / "load_env.sh").write_text(":\n")
    (scripts / "hip_train.sbatch").write_text("exit 0\n")
    (scripts / "submit_job.sh").write_text('touch "$HIP_TRAIN_OUTPUT/submitted"; echo 12345\n')
    (tmp_path / "report.json").write_text(json.dumps({"training_complete": False, "stop_reason": reason}))
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), HIP_TRAIN_OUTPUT=str(tmp_path),
               HIP_LOADER_PYTHON=sys.executable, SLURM_JOB_NAME="test", SLURM_JOB_ID="7")
    subprocess.run(["bash", str(scripts / original.name)], env=env, check=True, capture_output=True)
    assert (tmp_path / "submitted").exists() == expect_successor


def test_loader_creation_does_not_advance_training_rng():
    from hip_orb.data import make_loader
    # Worker base-seed draws must not perturb the restored trainer RNG when a
    # resumed process creates an extra iterator. Payloads remain deterministic.
    class Dataset:
        paths = ["unused"]
        ptr = np.array([0, 1])
        natoms = np.array([2])
        def __len__(self): return 1
        def __getitem__(self, i): return i
    loader = make_loader(Dataset(), workers=0, pin_memory=False, collate_fn=lambda x: x)
    state = torch.get_rng_state().clone()
    assert next(iter(loader)) == [0]
    assert torch.equal(state, torch.get_rng_state())
