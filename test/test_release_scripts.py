import importlib.util
import subprocess
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


def test_panel_weight_collection_includes_ocr_metadata():
    module = _load_weight_collector()
    config = Path(
        "plugins/vie-plugin-panel-label/vie_plugin_panel_label/config.py"
    )

    paths = set(module.collect_weight_paths([config], Path("weights")))

    assert {
        Path("panel_label/v2/textline_ori_lcnet_v2/inference.yml"),
        Path(
            "panel_label/v2/"
            "PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml"
        ),
    } <= paths


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


def test_rollback_script_swaps_previous_and_validates_readiness():
    script = Path("scripts/release/rollback-plugin.sh").read_text(encoding="utf-8")

    assert "--remote" in script
    assert "--remote-dir" in script
    assert "current" in script
    assert "previous" in script
    assert "--force-recreate" in script
    assert "/health/ready" in script


def test_offline_release_script_exports_scene_images_in_one_archive():
    script = Path("scripts/release/build_docker_release.sh").read_text(
        encoding="utf-8"
    )

    assert "RELEASE_VERSION" in script
    assert "Dockerfile.runtime" in script
    assert "mobile_vision:panel-label-" in script
    assert "mobile_vision:scenes-" in script
    assert "Dockerfile.panel-label" not in script
    assert "Dockerfile.scenes" not in script
    assert "sha256sum" in script
    assert "docker save" in script
    assert "collect_weight_paths.py" in script
    assert "CUDAExecutionProvider" in script
    assert "CUDA_SMOKE_MODEL" in script
    assert "ort.InferenceSession" in script
    assert "--gpus all" in script
    assert "a5b4e1641db48752118dda353b8614c6d6570344062b58faea70b5350c41cf68" in script
    assert "from services.scenario_registry import scenario_registry" in script
    assert "EXPECTED_VIE_PLUGINS" in script
    assert "entry_point.load()" in script
    assert "requirements.scenes.txt" in script
    assert "--service panel|scenes|all" in script
    assert 'OUTPUT_SUFFIX="-panel-label"' in script
    assert 'OUTPUT_SUFFIX="-scenes"' in script
    assert "INCLUDE_FRAMEWORK=0 INCLUDE_PLUGINS=0" in script
    assert 'docker save "${IMAGES[@]}"' in script
    assert script.count("docker save") == 1
    assert '> "$OUT/image.tar.gz"' in script
    assert '"$SERVICE/image.tar.gz"' not in script


def test_scenario_registry_type_alias_is_cython_compatible():
    source = Path("services/scenario_registry.py").read_text(encoding="utf-8")

    assert "ScenarioType = Type[BusinessLogicBase]" in source
    assert "ScenarioType = type[BusinessLogicBase]" not in source


def test_baseline_overlay_can_exclude_framework():
    script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")

    assert 'INCLUDE_FRAMEWORK="${INCLUDE_FRAMEWORK:-1}"' in script
    assert '[ "$pattern" = "vie_framework-*.whl" ]' in script
    assert "BUILD_WHEEL_ARGS+=(--plugins-only)" in script


def test_wheel_builder_supports_plugins_only_mode():
    script = Path("scripts/release/build_wheels.py").read_text(encoding="utf-8")

    assert '"--plugins-only"' in script
    assert "if not args.plugins_only:" in script


def test_offline_release_script_help_lists_service_split():
    result = subprocess.run(
        ["bash", "scripts/release/build_docker_release.sh", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--service panel|scenes|all" in result.stdout
    assert "只构建 panel-label 服务" in result.stdout
    assert "只构建 scenes 服务" in result.stdout


def test_release_scripts_use_configurable_mobile_vision_environment():
    release_script = Path("scripts/release/build_docker_release.sh").read_text(
        encoding="utf-8"
    )
    sync_script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")

    for script in (release_script, sync_script):
        assert 'CONDA_ENV="${CONDA_ENV:-mobile_vision}"' in script
        assert "conda run -n ppocr" not in script

    assert "setuptools.config.pyprojecttoml" in release_script
    assert 'WHEEL_BUILDER_IMAGE="${WHEEL_BUILDER_IMAGE:-mobile_vision:base}"' in sync_script
    assert 'docker image inspect "$WHEEL_BUILDER_IMAGE"' in sync_script
    assert "使用隔离构建" in sync_script


def test_services_share_base_contract_and_offline_archive():
    panel_sync = Path("scripts/release/sync-plugin.sh").read_text(encoding="utf-8")
    scenes_sync = Path("scripts/release/sync-plugin-scenes.sh").read_text(
        encoding="utf-8"
    )
    deploy = Path("scripts/release/deploy_offline.sh").read_text(encoding="utf-8")

    for script in (panel_sync, scenes_sync):
        assert 'RUNTIME_DOCKERFILE="Dockerfile.runtime"' in script
        assert "RUNTIME_REQUIREMENTS=(requirements.txt requirements.scenes.txt)" in script

    release = Path("scripts/release/build_docker_release.sh").read_text(
        encoding="utf-8"
    )
    assert "compute_base_contract.sh" in release
    assert "io.vie.base-contract-sha256" in release
    assert 'docker save "${IMAGES[@]}"' in release
    assert "gunzip -c image.tar.gz | docker load" in deploy
    assert 'gunzip -c "$SERVICE/image.tar.gz"' not in deploy
