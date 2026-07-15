import asyncio
import json

from app import readiness_check, router_registry
from services.inference_admission import AdmissionSnapshot


def _response_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_readiness_returns_503_for_failed_scene(monkeypatch):
    monkeypatch.setattr(
        router_registry,
        "preload_status",
        {
            "ok_scene": {"ready": True, "error": ""},
            "bad_scene": {"ready": False, "error": "load failed"},
        },
    )

    response = asyncio.run(readiness_check())

    assert response.status_code == 503
    body = _response_body(response)
    assert body["result"]["status"] == "not_ready"
    assert body["result"]["failed_scenes"] == ["bad_scene"]


def test_readiness_returns_200_when_all_scenes_ready(monkeypatch):
    monkeypatch.setattr(
        router_registry,
        "preload_status",
        {"ok_scene": {"ready": True, "error": ""}},
    )
    monkeypatch.setattr("app.settings.ONNX_REQUIRE_CUDA", False)

    response = asyncio.run(readiness_check())

    assert response.status_code == 200
    body = _response_body(response)
    assert body["result"]["status"] == "ready"
    assert body["result"]["failed_scenes"] == []


def test_readiness_exposes_sanitized_runtime_state(monkeypatch):
    monkeypatch.setattr(
        router_registry,
        "preload_status",
        {"panel_label": {"ready": True, "error": ""}},
    )
    monkeypatch.setattr("app.settings.ONNX_REQUIRE_CUDA", True)
    monkeypatch.setattr(
        "app.runtime_status_registry.public_snapshot",
        lambda: [
            {
                "model": "best.onnx",
                "providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            }
        ],
    )
    monkeypatch.setattr(
        "app.inference_admission_controller.snapshot",
        lambda: AdmissionSnapshot(1, 1, 2),
    )

    response = asyncio.run(readiness_check())
    body = _response_body(response)

    assert response.status_code == 200
    assert body["result"]["runtime"] == {
        "require_cuda": True,
        "models": [
            {
                "model": "best.onnx",
                "providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            }
        ],
        "inference": {
            "max_concurrency": 1,
            "active": 1,
            "waiting": 2,
        },
    }
    assert "/private/" not in json.dumps(body)


def test_readiness_returns_503_when_cuda_is_required_without_models(monkeypatch):
    monkeypatch.setattr(
        router_registry,
        "preload_status",
        {"panel_label": {"ready": True, "error": ""}},
    )
    monkeypatch.setattr("app.settings.ONNX_REQUIRE_CUDA", True)
    monkeypatch.setattr("app.runtime_status_registry.public_snapshot", lambda: [])

    response = asyncio.run(readiness_check())
    body = _response_body(response)

    assert response.status_code == 503
    assert body["result"]["status"] == "not_ready"
    assert body["result"]["failed_scenes"] == []
    assert body["result"]["runtime"]["models"] == []


def test_readiness_returns_503_when_cuda_is_required_for_cpu_model(monkeypatch):
    monkeypatch.setattr(
        router_registry,
        "preload_status",
        {"panel_label": {"ready": True, "error": ""}},
    )
    monkeypatch.setattr("app.settings.ONNX_REQUIRE_CUDA", True)
    monkeypatch.setattr(
        "app.runtime_status_registry.public_snapshot",
        lambda: [
            {
                "model": "cpu.onnx",
                "providers": ["CPUExecutionProvider"],
            }
        ],
    )

    response = asyncio.run(readiness_check())
    body = _response_body(response)

    assert response.status_code == 503
    assert body["result"]["status"] == "not_ready"
    assert body["result"]["failed_scenes"] == []
    assert body["result"]["runtime"]["models"] == [
        {
            "model": "cpu.onnx",
            "providers": ["CPUExecutionProvider"],
        }
    ]
