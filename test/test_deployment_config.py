import json
from pathlib import Path

import pytest


def test_compose_persists_data_directory():
    compose = Path("docker-compose.scenes.yml").read_text(encoding="utf-8")
    assert "./data:/app/workspace/data" in compose


def test_container_healthchecks_use_readiness_endpoint():
    compose = Path("docker-compose.scenes.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile.runtime").read_text(encoding="utf-8")
    assert "http://127.0.0.1:3001/health/ready" in compose
    assert "http://127.0.0.1:3001/health/ready" in dockerfile


def test_panel_label_compose_uses_runtime_image_without_removed_dockerfile():
    compose = Path("docker-compose.panel-label.yml").read_text(encoding="utf-8")

    assert "image: mobile_vision:runtime" in compose
    assert "dockerfile: Dockerfile.panel-label" not in compose
    assert "build:" not in compose
    assert 'ENABLED_SCENES=["panel_label"]' in compose
    assert "STRICT_STARTUP=True" in compose
    assert "http://127.0.0.1:3001/health/ready" in compose


def test_deploy_panel_label_compose_uses_runtime_image_without_removed_dockerfile():
    compose_path = Path("deploy/docker-compose.panel-label.yml")
    if not compose_path.exists():
        pytest.skip("local deploy bundle is not present")

    compose = compose_path.read_text(encoding="utf-8")

    assert "image: mobile_vision:runtime" in compose
    assert "dockerfile: Dockerfile.panel-label" not in compose
    assert "build:" not in compose
    assert 'ENABLED_SCENES=["panel_label"]' in compose
    assert "STRICT_STARTUP=True" in compose
    assert "http://127.0.0.1:3001/health/ready" in compose


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
    # CI 重构后收敛为统一 runtime 镜像（删除场景专用 Dockerfile.panel-label）
    for dockerfile_name in ("Dockerfile.runtime",):
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
    assert "./static:/app/workspace/static:ro" in compose


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
    script = Path("scripts/release/sync-plugin.sh").read_text(encoding="utf-8")

    assert "deploy/static/" in script
    assert "docker-compose.panel-label.yml" in script
    assert "docker compose -f '${COMPOSE_FILE}' up -d" in script


def test_panel_label_weight_sync_deletes_excluded_stale_model_files():
    script = Path("scripts/release/sync-plugin.sh").read_text(encoding="utf-8")

    assert "--delete-excluded" in script


def test_framework_wheel_keeps_third_party_dependencies_external():
    framework = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in framework


def test_requirements_is_the_ppocr_runtime_dependency_entrypoint():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "python-multipart", "paddleocr", "paddlex"):
        assert package in requirements
