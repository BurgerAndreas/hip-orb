import pytest
from hip_orb.schedules import learning_rate


def test_linear_warmup_and_cosine_endpoints():
    assert learning_rate(1, 0.001, "cosine", 1000000) == pytest.approx(0.001 / 150000)
    assert learning_rate(150000, 0.001, "cosine", 1000000) == 0.001
    assert learning_rate(575000, 0.001, "cosine", 1000000) == pytest.approx(0.0005)
    assert learning_rate(1000000, 0.001, "cosine", 1000000) == 0


def test_resumed_schedule_uses_absolute_update():
    full = [learning_rate(s, 0.001, "cosine", 1000) for s in range(1, 1001)]
    resumed = [learning_rate(s, 0.001, "cosine", 1000) for s in range(601, 1001)]
    assert resumed == full[600:]
    assert full[0] < full[148] < full[149]
    assert all(a >= b for a, b in zip(full[149:], full[150:]))


def test_step_limited_sessions_resume_optimizer_schedule(monkeypatch, tmp_path):
    import json
    import sys
    import numpy as np
    import torch
    from hip_orb import runner
    for name in ('train', 'val'):
        np.savez(tmp_path / f'{name}.npz', paths=np.array([name]), ptr=np.array([0, 1]),
                 natoms=np.array([2]), sources=np.array([name]))
    class Model:
        def __init__(self):
            self.base = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 1)
            self.head.config = {}
        def train(self):
            pass
    class Loader:
        batch_sampler = type('Sampler', (), {'set_epoch': lambda self, epoch: None})()
        def __iter__(self):
            return iter([{'coords': [0]}] * 20)
    monkeypatch.setattr(runner.HIPOrb, 'pretrained', lambda *a, **k: Model())
    monkeypatch.setattr(runner, 'configure_execution', lambda model, mode: {'mode': mode})
    monkeypatch.setattr(runner, 'make_loader', lambda *a, **k: Loader())
    rates, validations = [], []
    def update(model, optimizer, *args, **kwargs):
        rates.append(optimizer.param_groups[0]['lr'])
        return 1., 1
    def validate(*args):
        validations.append(len(rates))
        return {'validation': {'rmse': 1.}, 'validation_samples': 10}
    monkeypatch.setattr(runner, 'train_update', update)
    monkeypatch.setattr(runner, 'validate', validate)
    argv = ['train', '--train-index', str(tmp_path / 'train.npz'), '--validation-index',
            str(tmp_path / 'val.npz'), '--output', str(tmp_path / 'out'), '--device', 'cpu',
            '--max-steps', '5', '--session-steps', '2', '--validation-every', '2',
            '--lr-schedule', 'cosine', '--warmup-fraction', '0.2']
    for session, step in enumerate((2, 4, 5)):
        monkeypatch.setattr(sys, 'argv', argv + (['--checkpoint', str(tmp_path / 'out/head.pt')] if session else []))
        runner.main()
        report = json.loads((tmp_path / 'out/report.json').read_text())
        assert report['global_step'] == step
        assert report['training_complete'] == (step == 5)
    assert validations == [2, 4, 5]
    assert rates == [learning_rate(s, .001, 'cosine', 5, .2) for s in range(1, 6)]
