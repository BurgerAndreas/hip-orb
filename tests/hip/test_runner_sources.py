"""The real training runner requires separate training and validation sources."""
import sys

import numpy as np
import pytest

from hip_orb import runner, train


def index(path, shard, source):
    np.savez(path, paths=np.asarray([str(shard)]), ptr=np.asarray([0, 1]),
             natoms=np.asarray([2]), sources=np.asarray([source]))
    return path


@pytest.mark.parametrize("same_shard,same_source", [(True, False), (False, True)])
def test_separate_sources_required_before_model_loading(monkeypatch, tmp_path, same_shard, same_source):
    training = index(tmp_path / "train.npz", tmp_path / "train.h5", "training")
    validation = index(tmp_path / "validation.npz",
                       tmp_path / ("train.h5" if same_shard else "validation.h5"),
                       "training" if same_source else "validation")
    monkeypatch.setattr(runner.HIPOrb, "pretrained", lambda *a, **k: pytest.fail("Model must not load"))
    monkeypatch.setattr(sys, "argv", ["train", "--train-index", str(training),
                                      "--validation-index", str(validation), "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2


def test_training_entry_point_has_no_alternate_runner():
    assert train.main is runner.main


def test_validation_index_is_required(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["train", "--train-index", str(tmp_path / "train.npz"),
                                      "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2


@pytest.mark.parametrize("session_duration", [None, 6, 10])
@pytest.mark.parametrize("loss_type", ["mse", "mae", "block_rmse", "mass_weighted_mse"])
def test_timed_run_stops_updates_and_always_validates(monkeypatch, tmp_path, session_duration, loss_type):
    import json
    import torch

    training = index(tmp_path / "train.npz", tmp_path / "train.h5", "training")
    validation = index(tmp_path / "validation.npz", tmp_path / "val.h5", "validation")
    clock = [0.0]
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])

    class Model:
        base = torch.nn.Linear(1, 1)
        head = torch.nn.Linear(1, 1)
        head.config = {}

        def train(self):
            pass

    monkeypatch.setattr(runner.HIPOrb, "pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(runner, "configure_execution", lambda model, mode: {"mode": mode})
    loader_settings = []

    class Loader:
        batch_sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()

        def __iter__(self):
            return iter([{"coords": [0, 0]}] * 20)

    def make_loader(dataset, **settings):
        loader_settings.append(settings)
        return Loader()

    monkeypatch.setattr(runner, "make_loader", make_loader)
    monkeypatch.setattr(runner, "prepare_batch", lambda batch, **kwargs: batch)

    def step(*args, **kwargs):
        assert kwargs["loss_type"] == loss_type
        clock[0] += 4
        return 1.0

    monkeypatch.setattr(runner, "training_step", step)
    validation_starts = []

    def validate(*args):
        validation_starts.append(clock[0])
        clock[0] += 1
        return {"validation": {"rmse": 1.0}, "validation_samples": 10}

    monkeypatch.setattr(runner, "validate", validate)
    argv = ["train", "--train-index", str(training),
        "--validation-index", str(validation), "--output", str(tmp_path / "out"),
        "--device", "cpu", "--max-duration-seconds", "20" if session_duration == 10 else "10",
        "--validation-every-seconds", "5", "--validation-batch-size", "256",
        "--validation-samples", "10000", "--loss-type", loss_type]
    if session_duration:
        argv += ["--session-duration-seconds", str(session_duration)]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    if session_duration:
        paused = json.loads((tmp_path / "out/report.json").read_text())
        assert paused["global_step"] == (3 if session_duration == 10 else 2)
        assert not paused["training_complete"]
        assert not paused["final_validation_completed"]
        assert validation_starts == ([8] if session_duration == 10 else [])
        monkeypatch.setattr(sys, "argv", argv + ["--checkpoint", str(tmp_path / "out/head.pt")])
        runner.main()
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["global_step"] == (5 if session_duration == 10 else 3)
    assert validation_starts == ([8, 17, 22] if session_duration == 10
                                 else [12] if session_duration else [8, 13])
    assert report["final_validation_completed"]
    assert report["checkpoint_readable"]
    assert report["loss_type"] == loss_type
    assert f"train_{loss_type}" in report
    saved = torch.load(tmp_path / "out/head.pt", weights_only=True)
    assert saved["loss_type"] == loss_type
    assert loader_settings[0]["batch_size"] == 256
    assert loader_settings[0]["max_samples"] == 10000
    assert loader_settings[1]["batch_size"] == 128
    if session_duration:
        assert loader_settings[3]["start_batch"] == (3 if session_duration == 10 else 2)
    other_loss = "mae" if loss_type == "mse" else "mse"
    monkeypatch.setattr(sys, "argv", argv + ["--checkpoint", str(tmp_path / "out/head.pt"),
                                            "--loss-type", other_loss])
    with pytest.raises(ValueError, match="Checkpoint Hessian loss differs"):
        runner.main()
