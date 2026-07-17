import importlib.util
from pathlib import Path

import pytest


def _load_weight_collector():
    path = Path("scripts/release/collect_weight_paths.py")
    spec = importlib.util.spec_from_file_location("collect_weight_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_weight_paths_reads_config_literals_and_expands_directories(tmp_path):
    module = _load_weight_collector()
    root = tmp_path / "weights"
    (root / "scene/model_dir").mkdir(parents=True)
    (root / "scene/model_dir/inference.yml").write_text("model", encoding="utf-8")
    (root / "scene/det_v1.onnx").write_bytes(b"onnx")
    config = tmp_path / "config.py"
    config.write_text(
        'det = "./weights/scene/det_v1.onnx"\n'
        'rec = "./weights/scene/model_dir"\n',
        encoding="utf-8",
    )

    paths = module.collect_weight_paths([config], root)

    assert paths == [
        Path("scene/det_v1.onnx"),
        Path("scene/model_dir/inference.yml"),
    ]


def test_collect_weight_paths_rejects_missing_assets(tmp_path):
    module = _load_weight_collector()
    root = tmp_path / "weights"
    root.mkdir()
    config = tmp_path / "config.py"
    config.write_text('model = "./weights/scene/missing.onnx"', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.onnx"):
        module.collect_weight_paths([config], root)


@pytest.mark.parametrize(
    "script_name",
    ("sync-plugin.sh", "sync-plugin-scenes.sh"),
)
def test_sync_scripts_use_atomic_versioned_releases(script_name):
    script = Path("scripts/release", script_name).read_text(encoding="utf-8")
    script += Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")
    script += Path("scripts/release/remote_activate.sh").read_text(encoding="utf-8")

    assert "RELEASE_ID" in script
    assert ".staging" in script
    assert "releases/" in script
    assert "current" in script
    assert "previous" in script
    assert "io.vie.requirements-sha256" in script
    assert "io.vie.python-abi" in script
    assert "io.vie.runtime-contract-sha256" in script
    assert "--no-weights" in script
    assert "--force-recreate" in script
    assert "/health/ready" in script
    assert "rollback" in script.lower()
    assert "trap 'rollback' ERR" in script
    assert "192.168." not in script
    assert "REMOTE_DIR:-/" not in script
