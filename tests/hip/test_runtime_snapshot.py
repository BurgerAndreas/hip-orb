import json

import pytest

from hip_orb.runtime_snapshot import create_snapshot, verify_snapshot


def test_snapshot_detects_a_missing_lazy_graph_import(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "runner.py").write_text('def step():\n    from hip_orb.tensor_graph import build\n')
    with pytest.raises(ValueError, match="hip_orb.tensor_graph"):
        create_snapshot(source, tmp_path / "broken")
    assert not json.loads((tmp_path / "broken/manifest.json").read_text())["complete"]


def test_snapshot_verifies_contents_and_refuses_overwrites(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "runner.py").write_text('def step():\n    from hip_orb.tensor_graph import build\n')
    (source / "tensor_graph.py").write_text('def build():\n    return 1\n')
    target = tmp_path / "runtime"
    manifest = create_snapshot(source, target)
    assert manifest["complete"]
    assert verify_snapshot(target) == manifest
    with pytest.raises(FileExistsError):
        create_snapshot(source, target)
    (target / "hip_orb/tensor_graph.py").write_text('def build():\n    return 2\n')
    with pytest.raises(ValueError, match="tensor_graph.py"):
        verify_snapshot(target)
