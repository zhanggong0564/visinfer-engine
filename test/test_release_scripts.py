import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


def _load_weight_collector():
    path = Path("scripts/release/collect_weight_paths.py")
    spec = importlib.util.spec_from_file_location("collect_weight_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_contract(root: Path) -> str:
    env = os.environ.copy()
    env["VIE_CONTRACT_ROOT"] = str(root)
    result = subprocess.run(
        ["bash", "scripts/release/compute_environment_contract.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _write_environment_contract_fixture(root: Path) -> None:
    (root / "whl").mkdir(parents=True)
    (root / "Dockerfile.base").write_text("base-v1\n", encoding="utf-8")
    (root / "Dockerfile.runtime").write_text("runtime-v1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi==1\n", encoding="utf-8")
    (root / "requirements.scenes.txt").write_text("numpy==1\n", encoding="utf-8")
    ort_wheel = root / (
        "whl/onnxruntime_gpu-1.20.1-cp310-cp310-"
        "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    ort_wheel.write_bytes(b"ort-wheel")


def test_environment_contract_ignores_framework_source_changes(tmp_path):
    _write_environment_contract_fixture(tmp_path)
    before = _environment_contract(tmp_path)

    source = tmp_path / "services/example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    assert _environment_contract(tmp_path) == before


@pytest.mark.parametrize("changed_file", ["requirements.txt", "Dockerfile.runtime"])
def test_environment_contract_tracks_runtime_inputs(tmp_path, changed_file):
    _write_environment_contract_fixture(tmp_path)
    before = _environment_contract(tmp_path)

    path = tmp_path / changed_file
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    assert _environment_contract(tmp_path) != before


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


def test_panel_weight_collection_includes_ocr_metadata(tmp_path):
    module = _load_weight_collector()
    config = Path(
        "plugins/vie-plugin-panel-label/vie_plugin_panel_label/config.py"
    )
    weights_root = tmp_path / "weights"
    expected_files = (
        "panel_label/v2/rfdetr-seg-nano-v1.1.onnx",
        "panel_label/v2/textline_ori_lcnet_v2.onnx",
        "panel_label/v2/textline_ori_lcnet_v2/inference.yml",
        "panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx",
        "panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml",
    )
    for relative_path in expected_files:
        path = weights_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = set(module.collect_weight_paths([config], weights_root))

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
    assert "io.vie.environment-contract-sha256" in script
    assert "--no-weights" in script
    assert "--allow-legacy-image" in script
    assert "--force-recreate" in script
    assert "/health/ready" in script
    assert "rollback" in script.lower()
    assert "trap 'rollback' ERR" in script
    assert "192.168." not in script
    assert "REMOTE_DIR:-/" not in script


def test_no_weights_skips_local_weight_collection():
    script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")

    weight_guard = 'if [ "$DO_WEIGHTS" -eq 1 ]; then'
    collector = 'scripts/release/collect_weight_paths.py'
    empty_manifest = ': > "$LOCAL_STAGE/weight-paths.txt"'

    assert script.index(weight_guard) < script.index(collector)
    assert empty_manifest in script


def test_legacy_image_requires_explicit_compatibility_flag():
    script = Path("scripts/release/remote_activate.sh").read_text(encoding="utf-8")

    assert 'ALLOW_LEGACY_IMAGE="${12}"' in script
    assert 'if [ "$ALLOW_LEGACY_IMAGE" -ne 1 ]' in script
    assert "旧镜像缺少环境契约标签" in script
    assert "IMAGE_REQUIREMENTS_SHA" in script
    assert "IMAGE_PYTHON_ABI" in script


@pytest.mark.parametrize(
    ("image_requirements", "image_abi", "allow_legacy", "expected_error"),
    [
        ("requirements-sha", "cp310", "0", "旧镜像缺少环境契约标签"),
        ("different", "cp310", "1", "依赖指纹不一致"),
        ("requirements-sha", "cp311", "1", "Python ABI 不一致"),
    ],
)
def test_legacy_image_requires_opt_in_and_matching_runtime(
    tmp_path,
    image_requirements,
    image_abi,
    allow_legacy,
    expected_error,
):
    root = tmp_path / "deploy"
    stage = root / "releases/release-1.staging"
    (stage / "pkg").mkdir(parents=True)
    (stage / "app.py").touch()
    (stage / "docker-compose.panel-label.yml").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1 $2\" = \"compose version\" ]; then exit 0; fi\n"
        "case \"$*\" in\n"
        "  *Config.Image*) echo legacy-image ;;\n"
        "  *requirements-sha256*) echo \"$TEST_IMAGE_REQUIREMENTS\" ;;\n"
        "  *python-abi*) echo \"$TEST_IMAGE_ABI\" ;;\n"
        "  *environment-contract-sha256*) echo '<no value>' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TEST_IMAGE_REQUIREMENTS"] = image_requirements
    env["TEST_IMAGE_ABI"] = image_abi
    result = subprocess.run(
        [
            "bash",
            "scripts/release/remote_activate.sh",
            str(root),
            "release-1",
            "docker-compose.panel-label.yml",
            "mobile-vision-panel-label",
            "http://127.0.0.1:3001/health/ready",
            "requirements-sha",
            "cp310",
            "runtime-sha",
            "0",
            "panel_label",
            "environment-sha",
            allow_legacy,
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_rollback_script_swaps_previous_and_validates_readiness():
    script = Path("scripts/release/rollback-plugin.sh").read_text(encoding="utf-8")

    assert "--remote" in script
    assert "--remote-dir" in script
    assert "current" in script
    assert "previous" in script
    assert "--force-recreate" in script
    assert "/health/ready" in script


def test_remote_activation_ignores_blank_weight_paths():
    script = Path("scripts/release/remote_activate.sh").read_text(encoding="utf-8")

    assert '[ -n "$weight" ] || continue' in script


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


def test_baseline_overlay_can_exclude_image_baked_code():
    script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")

    assert 'INCLUDE_FRAMEWORK="${INCLUDE_FRAMEWORK:-1}"' in script
    assert 'INCLUDE_PLUGINS="${INCLUDE_PLUGINS:-1}"' in script
    assert '[ "$pattern" = "vie_framework-*.whl" ]' in script
    assert '[ "$pattern" != "vie_framework-*.whl" ]' in script
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
    assert 'WHEEL_BUILDER_IMAGE="${WHEEL_BUILDER_IMAGE:-mobile_vision:base-builder}"' in sync_script
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
