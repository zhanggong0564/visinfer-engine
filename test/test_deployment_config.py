import json
import re
import runpy
from pathlib import Path

import pytest
import setuptools
from Cython import Build
from packaging.requirements import Requirement


LEGACY_DC_FUSE_FILES = (
    Path("services/dc_fuse/business_logic.py"),
    Path("services/dc_fuse/dc_fuse_detect.py"),
    Path("routers/dc_fuse_routers.py"),
    Path("schemas/dc_fuse_schemas.py"),
    Path("config/dc_fuse_config.py"),
)


def test_legacy_dc_fuse_example_files_remain_available():
    assert all(path.is_file() for path in LEGACY_DC_FUSE_FILES)


def test_framework_build_excludes_legacy_scene_examples(monkeypatch):
    setup_kwargs = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs))
    monkeypatch.setattr(Build, "cythonize", lambda sources, **kwargs: sources)

    setup_globals = runpy.run_path("setup.py", run_name="__build_contract__")
    py_sources = {Path(source) for source in setup_globals["py_sources"]}

    assert not py_sources.intersection(LEGACY_DC_FUSE_FILES)
    assert not any(Path("services/dc_fuse") in source.parents for source in py_sources)
    assert "services.dc_fuse" not in setup_kwargs["packages"]
    assert not any(
        package.startswith("services.dc_fuse.") for package in setup_kwargs["packages"]
    )


def test_compose_persists_data_directory():
    compose = Path("docker-compose.scenes.yml").read_text(encoding="utf-8")
    assert "./data:/app/workspace/data" in compose


def test_onnx_runtime_gpu_image_matches_ort_120_cuda_requirements():
    expected_base = (
        "swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/"
        "cuda:12.4.1-cudnn-runtime-ubuntu22.04"
    )
    for dockerfile_name in (
        "Dockerfile.base",
        "Dockerfile.panel-label",
        "Dockerfile.scenes",
    ):
        dockerfile = Path(dockerfile_name).read_text(encoding="utf-8")
        assert f"ARG BASE_IMAGE={expected_base}" in dockerfile

    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "onnxruntime-gpu==1.20.1" in requirements


def test_container_healthchecks_use_readiness_endpoint():
    for path in (
        Path("Dockerfile.panel-label"),
        Path("Dockerfile.scenes"),
        Path("docker-compose.panel-label.yml"),
        Path("docker-compose.scenes.yml"),
    ):
        content = path.read_text(encoding="utf-8")
        assert "http://127.0.0.1:3001/health/ready" in content


def test_panel_label_compose_uses_service_image_and_versioned_overlay():
    compose = Path("docker-compose.panel-label.yml").read_text(encoding="utf-8")

    assert "image: ${PANEL_LABEL_IMAGE:-mobile_vision:panel-label}" in compose
    assert "build:" not in compose
    assert 'ENABLED_SCENES=["panel_label"]' in compose
    assert "STRICT_STARTUP=true" in compose
    assert "http://127.0.0.1:3001/health/ready" in compose
    for mount in ("pkg", "weights", "app.py", "static"):
        assert f"./current/{mount}:/app/workspace/{mount}:ro" in compose


def test_scenes_compose_uses_service_image_and_versioned_overlay():
    compose = Path("docker-compose.scenes.yml").read_text(encoding="utf-8")

    assert "image: ${SCENES_IMAGE:-mobile_vision:scenes}" in compose
    assert "build:" not in compose
    for mount in ("pkg", "weights", "app.py", "static"):
        assert f"./current/{mount}:/app/workspace/{mount}:ro" in compose


def test_service_dockerfiles_bake_only_their_plugins():
    panel = Path("Dockerfile.panel-label").read_text(encoding="utf-8")
    scenes = Path("Dockerfile.scenes").read_text(encoding="utf-8")

    assert "COPY plugins/vie-plugin-panel-label/" in panel
    assert "--plugins panel-label" in panel
    assert "vie-plugin-line-squeeze" not in panel

    for plugin in (
        "dc-fuse",
        "indicator-light",
        "lap-surf",
        "line-squeeze",
        "plate-screw",
    ):
        assert f"COPY plugins/vie-plugin-{plugin}/" in scenes
    assert "vie-plugin-panel-label" not in scenes


def test_service_images_publish_sync_compatibility_labels():
    for path in (Path("Dockerfile.panel-label"), Path("Dockerfile.scenes")):
        dockerfile = path.read_text(encoding="utf-8")
        assert "io.vie.python-abi" in dockerfile
        assert "io.vie.requirements-sha256" in dockerfile
        assert "io.vie.runtime-contract-sha256" in dockerfile
        assert "io.vie.framework-version" in dockerfile
        assert "io.vie.plugins" in dockerfile
        assert "io.vie.plugin-versions" in dockerfile


def test_deploy_panel_label_compose_uses_service_image_and_versioned_overlay():
    compose_path = Path("deploy/docker-compose.panel-label.yml")
    if not compose_path.exists():
        pytest.skip("local deploy bundle is not present")

    compose = compose_path.read_text(encoding="utf-8")

    assert "image: ${PANEL_LABEL_IMAGE:-mobile_vision:panel-label}" in compose
    assert "build:" not in compose
    assert 'ENABLED_SCENES=["panel_label"]' in compose
    assert "STRICT_STARTUP=true" in compose
    assert "http://127.0.0.1:3001/health/ready" in compose
    for mount in ("pkg", "weights", "app.py", "static"):
        assert f"./current/{mount}:/app/workspace/{mount}:ro" in compose


@pytest.mark.parametrize(
    "compose_path",
    (
        Path("docker-compose.scenes.yml"),
        Path("docker-compose.panel-label.yml"),
        Path("deploy/docker-compose.panel-label.yml"),
    ),
)
def test_gpu_composes_enforce_onnx_runtime_safety(compose_path):
    if not compose_path.exists():
        pytest.skip(f"optional deploy bundle missing: {compose_path}")
    compose = compose_path.read_text(encoding="utf-8")

    assert "ONNX_REQUIRE_CUDA=true" in compose
    assert "STRICT_STARTUP=true" in compose
    assert "INFERENCE_MAX_CONCURRENCY=1" in compose
    assert "INFERENCE_MAX_QUEUE" not in compose
    assert "INFERENCE_QUEUE_TIMEOUT" not in compose


def test_swagger_ui_uses_local_static_assets():
    app = Path("app.py").read_text(encoding="utf-8")

    assert 'docs_url=None' in app
    assert 'app.mount("/static"' in app
    assert 'swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js"' in app
    assert 'swagger_css_url="/static/swagger-ui/swagger-ui.css"' in app
    assert 'swagger_favicon_url="/static/swagger-ui/favicon-32x32.png"' in app
    assert 'swagger_ui_parameters={"validatorUrl": None}' in app
    assert "cdn.jsdelivr.net" not in app
    assert "fastapi.tiangolo.com" not in app


def test_detection_routes_publish_common_response_schema_to_openapi():
    router = Path("routers/base_router.py").read_text(encoding="utf-8")

    assert "response_model=CommonResponse" in router


def test_openapi_documents_actual_detection_response_contract():
    from app import app

    schema = app.openapi()
    detection_item = schema["components"]["schemas"]["DetectionItemResponse"]
    status_schema = detection_item["properties"]["status"]

    assert status_schema["enum"] == ["true", "false"]
    assert "boolean" not in status_schema

    dc_fuse_op = schema["paths"]["/api/v1/dcfuse_detect"]["post"]
    assert "422" not in dc_fuse_op["responses"]
    assert dc_fuse_op["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommonResponse"
    }


def test_openapi_documents_dc_fuse_json_data_example():
    from app import app

    schema = app.openapi()
    dc_fuse_body = schema["components"]["schemas"]["Body__handle_request_api_v1_dcfuse_detect_post"]
    json_data_schema = dc_fuse_body["properties"]["json_data"]

    assert "DCFUSE_JSON_DATA_EXAMPLE" not in json_data_schema.get("example", "")
    assert '"product"' in json_data_schema["example"]
    assert '"product_model"' in json_data_schema["example"]
    assert json_data_schema["description"].startswith("JSON 字符串")


def test_openapi_documents_panel_label_json_data_example_from_real_request_log():
    from app import app

    schema = app.openapi()
    panel_body = schema["components"]["schemas"]["Body__handle_request_api_v1_panel_label_detect_post"]
    json_data_schema = panel_body["properties"]["json_data"]
    example = json.loads(json_data_schema["example"])

    assert example["product"] == "逆变器组件_SG1100UD-V3039_S"
    assert example["type"] == "A0ST6329"
    assert example["sn"] == "A2670608545"
    assert example["modelParams"]["product_type"] == "S1S2"
    assert example["modelParams"]["rule"] == "front"
    assert example["modelParams"]["line_order"] == "S2-14,S2-13,S1-13,S1-14"
    assert example["modelParams"]["guideline_coordinates"] == "0.154,0.114666666666667,0.771,0.76"
    assert example["modelParams"]["guide_line"][0]["FileName"] == "5、直流侧开关S1S2.png"
    assert example["modelParams"]["example_images"][0]["FileName"] == "屏幕截图 2026-04-22 145231.png"
    assert example["AICameraModel"][0]["Version"] == 4
    assert example["AICameraModel"][0]["AIParameterValue"] == "五路有熔丝盒无磁环"


def test_docker_images_include_offline_swagger_assets():
    for dockerfile_name in ("Dockerfile.panel-label", "Dockerfile.scenes"):
        dockerfile = Path(dockerfile_name).read_text(encoding="utf-8")
        assert "static /app/workspace/static" in dockerfile


def test_deploy_bundle_swagger_ui_uses_local_static_assets():
    app_path = Path("deploy/app.py")
    if not app_path.exists():
        pytest.skip("local deploy bundle is not present")

    app = app_path.read_text(encoding="utf-8")

    assert 'docs_url=None' in app
    assert 'app.mount("/static"' in app
    assert 'swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js"' in app
    assert 'swagger_css_url="/static/swagger-ui/swagger-ui.css"' in app
    assert 'swagger_favicon_url="/static/swagger-ui/favicon-32x32.png"' in app
    assert 'swagger_ui_parameters={"validatorUrl": None}' in app
    assert "cdn.jsdelivr.net" not in app
    assert "fastapi.tiangolo.com" not in app


def test_deploy_compose_mounts_offline_swagger_assets():
    compose_path = Path("deploy/docker-compose.panel-label.yml")
    if not compose_path.exists():
        pytest.skip("local deploy bundle is not present")

    compose = compose_path.read_text(encoding="utf-8")
    assert "./current/static:/app/workspace/static:ro" in compose


def test_deploy_bundle_includes_offline_swagger_assets():
    if not Path("deploy").exists():
        pytest.skip("local deploy bundle is not present")

    for asset in (
        "deploy/static/swagger-ui/swagger-ui-bundle.js",
        "deploy/static/swagger-ui/swagger-ui.css",
        "deploy/static/swagger-ui/favicon-32x32.png",
    ):
        assert Path(asset).is_file()


def test_sync_script_pushes_offline_swagger_assets_and_applies_compose():
    script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")
    activate = Path("scripts/release/remote_activate.sh").read_text(encoding="utf-8")

    assert "static/swagger-ui" in script
    assert 'cp "$COMPOSE_FILE"' in script
    assert 'docker compose -f "$COMPOSE_FILE" up -d --force-recreate' in activate


def test_panel_label_weight_sync_deletes_excluded_stale_model_files():
    script = Path("scripts/release/sync-common.sh").read_text(encoding="utf-8")

    assert "collect_weight_paths.py" in script
    assert 'rsync -a --files-from="$LOCAL_STAGE/weight-paths.txt"' in script


def test_framework_wheel_keeps_third_party_dependencies_external():
    framework = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in framework


def test_framework_package_version_includes_yolo_pipeline():
    framework = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'version = "2.1.2"' in framework


def test_runtime_requirements_use_onnx_without_paddle():
    requirements = [
        Requirement(line)
        for raw_line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]
    by_name = {}
    for requirement in requirements:
        by_name.setdefault(requirement.name.lower(), []).append(requirement)

    assert len(by_name["onnxruntime-gpu"]) == 1
    assert str(by_name["onnxruntime-gpu"][0].specifier) == "==1.20.1"
    assert "paddleocr" not in by_name
    assert "paddlex" not in by_name


def test_base_image_installs_local_onnx_wheel_before_requirements():
    dockerfile = Path("Dockerfile.base").read_text(encoding="utf-8")
    wheel = (
        "onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64."
        "manylinux_2_28_x86_64.whl"
    )
    local_install = f"pip install /tmp/{wheel} --no-deps"
    requirements_install = "pip install -r /tmp/requirements.txt"

    assert f"COPY whl/{wheel} /tmp/" in dockerfile
    assert local_install in dockerfile
    assert dockerfile.index(local_install) < dockerfile.index(requirements_install)
    assert "--force-reinstall" not in dockerfile
    assert "paddlepaddle_gpu" not in dockerfile.lower()


def test_runtime_image_installs_opencv_system_libraries():
    dockerfile = Path("Dockerfile.runtime").read_text(encoding="utf-8").lower()
    apt_install = re.search(
        r"apt-get install -y --no-install-recommends(?P<packages>.*?)&&",
        dockerfile,
        flags=re.DOTALL,
    )

    assert apt_install is not None
    packages = set(re.findall(r"^\s*([a-z0-9.+-]+)\s*\\?$", apt_install["packages"], re.MULTILINE))
    assert {"libgl1", "libgomp1"}.issubset(packages)
