from pathlib import Path

import pytest


def test_compose_persists_data_directory():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "./data:/app/workspace/data" in compose


def test_container_healthchecks_use_readiness_endpoint():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "http://127.0.0.1:3001/health/ready" in compose
    assert "http://127.0.0.1:3001/health/ready" in dockerfile


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


def test_docker_images_include_offline_swagger_assets():
    for dockerfile_name in ("Dockerfile", "Dockerfile.panel-label"):
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


def test_framework_wheel_keeps_third_party_dependencies_external():
    framework = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in framework


def test_requirements_is_the_ppocr_runtime_dependency_entrypoint():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "python-multipart", "paddleocr", "paddlex"):
        assert package in requirements
