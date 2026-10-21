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


@pytest.mark.parametrize("batching,device", [("equal", "cuda"), ("balanced", "cpu")])
def test_mixed_precision_requires_cuda_packed_training(monkeypatch, tmp_path, batching, device):
    monkeypatch.setattr(runner.HIPOrb, "pretrained", lambda *a, **k: pytest.fail("Model must not load"))
    monkeypatch.setattr(sys, "argv", ["train", "--train-index", str(tmp_path / "train.npz"),
                                     "--validation-index", str(tmp_path / "val.npz"),
                                     "--output", str(tmp_path), "--batching", batching,
                                     "--device", device, "--loss-type", "block_rmse",
                                     "--precision-policy", "fp16-message-head"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_nonfinite_packed_loss_saves_resumable_state_and_batch(monkeypatch, tmp_path):
    import torch

    training = index(tmp_path / 'train.npz', tmp_path / 'train.h5', 'training')
    validation = index(tmp_path / 'val.npz', tmp_path / 'val.h5', 'validation')

    class Model:
        def __init__(self):
            self.base = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 1)
            self.head.config = {}

        def train(self):
            pass

    class Sampler:
        fingerprint = 'test-plan'

        def __init__(self, *args, **kwargs):
            self.settings = {'test': True}

        def __len__(self):
            return 1

        def set_epoch(self, epoch):
            pass

    batch = {'sizes': torch.tensor([2]), 'omol_index': torch.tensor([123])}

    class Loader:
        batch_sampler = Sampler()

        def __iter__(self):
            return iter([batch])

    def fail(*args, **kwargs):
        raise FloatingPointError('Packed training loss is not finite')

    monkeypatch.setattr(runner.HIPOrb, 'pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(runner, 'configure_execution', lambda *a: {})
    monkeypatch.setattr(runner, 'BalancedBatchSampler', Sampler)
    monkeypatch.setattr(runner, 'make_loader', lambda *a, **k: Loader())
    monkeypatch.setattr(runner, 'packed_update', fail)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['train', '--train-index', str(training),
        '--validation-index', str(validation), '--output', str(output),
        '--device', 'cpu', '--batching', 'balanced', '--loss-type', 'block_rmse', '--max-steps', '2'])
    with pytest.raises(FloatingPointError, match='not finite'):
        runner.main()
    checkpoint = torch.load(output / 'head.pt', weights_only=True)
    failed = torch.load(output / 'nonfinite_batch.pt', weights_only=True)
    assert checkpoint['global_step'] == checkpoint['epoch_batches'] == 0
    assert checkpoint['optimizer']['state'] == {}
    assert failed['global_step'] == 0 and failed['seed'] == 7
    torch.testing.assert_close(failed['batch']['omol_index'], batch['omol_index'])


def test_throughput_includes_work_outside_update(monkeypatch, tmp_path, capsys):
    import json

    import torch

    training = index(tmp_path / 'train.npz', tmp_path / 'train.h5', 'training')
    validation = index(tmp_path / 'val.npz', tmp_path / 'val.h5', 'validation')
    clock = [0.0]
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: clock[0])

    class Model:
        def __init__(self):
            self.base = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 1)
            self.head.config = {}

        def train(self):
            pass

    class Loader:
        batch_sampler = type('Sampler', (), {'set_epoch': lambda *a: None})()

        def __iter__(self):
            return iter([{'coords': [0, 0]}])

    def lr(*args):
        clock[0] += 2.0
        return 0.001

    def step(*args, **kwargs):
        clock[0] += 1.0
        return 1.0

    monkeypatch.setattr(runner.HIPOrb, 'pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(runner, 'configure_execution', lambda *a: {})
    monkeypatch.setattr(runner, 'make_loader', lambda *a, **k: Loader())
    monkeypatch.setattr(runner, 'prepare_batch', lambda batch, **k: batch)
    monkeypatch.setattr(runner, 'training_step', step)
    monkeypatch.setattr(runner, 'learning_rate', lr)
    monkeypatch.setattr(runner, 'validate', lambda *a: {'validation': {}, 'validation_samples': 0})
    monkeypatch.setattr(sys, 'argv', ['train', '--train-index', str(training),
        '--validation-index', str(validation), '--output', str(tmp_path / 'out'),
        '--device', 'cpu', '--max-steps', '1'])
    runner.main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    row = next(r for r in rows if 'updates_per_second' in r)
    assert row['updates_per_second'] == pytest.approx(1 / 3)
    assert row['geometries_per_second'] == pytest.approx(2 / 3)


def test_zero_regularization_overrides_loaded_optimizer_settings_and_dropout():
    import torch
    from types import SimpleNamespace
    base = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.3))
    head = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.4))
    optimizer = torch.optim.AdamW(head.parameters(), weight_decay=0.2)
    loaded = optimizer.state_dict()
    loaded['param_groups'][0]['weight_decay'] = 0.7
    optimizer.load_state_dict(loaded)
    settings = runner.enforce_zero_regularization(SimpleNamespace(base=base, head=head), optimizer)
    assert settings == {'weight_decay': 0.0, 'dropout': 0.0}
    assert all(group['weight_decay'] == 0 for group in optimizer.param_groups)
    assert base[1].p == head[1].p == 0
    head.train()
    x = torch.randn(8, 2)
    torch.testing.assert_close(head(x), head(x), atol=0, rtol=0)
