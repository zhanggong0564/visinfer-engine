from pathlib import Path


def test_compose_persists_data_directory():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "./data:/app/workspace/data" in compose


def test_container_healthchecks_use_readiness_endpoint():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "http://127.0.0.1:3007/health/ready" in compose
    assert "http://127.0.0.1:3007/health/ready" in dockerfile


def test_framework_wheel_keeps_third_party_dependencies_external():
    framework = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in framework


def test_requirements_is_the_ppocr_runtime_dependency_entrypoint():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("fastapi", "python-multipart", "paddleocr", "paddlex"):
        assert package in requirements
