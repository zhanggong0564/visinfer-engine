import asyncio
import json

from app import readiness_check, router_registry


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

    response = asyncio.run(readiness_check())

    assert response.status_code == 200
    body = _response_body(response)
    assert body["result"]["status"] == "ready"
    assert body["result"]["failed_scenes"] == []
