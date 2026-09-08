import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from schemas.exceptions import InvalidParamsError, VisionAPIError
from schemas.error_codes import ErrorCode
from utils import vision_logger
from utils.async_utils import run_sync
from utils.log_context import current_log_context, detection_log_context
from utils.request_logging import RequestLoggingMiddleware, log_detection_result, log_json


def payload(record):
    return json.loads(record["message"][record["message"].index("{"):])


def test_streaming_timing_and_concurrent_context_propagation(log_records):
    async def app(scope, receive, send):
        # Middleware must not pre-read or buffer the body.
        assert scope["read_calls"] == 0
        first = await receive()
        second = await receive()
        assert first["body"] + second["body"] == b"abcdef"
        with detection_log_context(scope["path"], scope["path"] + ".jpg"):
            current_log_context().product_type = "T1"
            await run_sync(vision_logger.info, "worker", event="test.worker")
            log_detection_result({"verdict": "REVIEW", "status": "false", "detailList": [{"verdict": "REVIEW"}]})
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"part1", "more_body": True})
        await send({"type": "http.response.body", "body": b"part2"})
        await run_sync(vision_logger.info, "background", event="test.background")

    async def call(path):
        scope = {"type": "http", "method": "POST", "path": path, "read_calls": 0}
        messages = iter([
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def"},
        ])
        sent = []

        async def receive():
            scope["read_calls"] += 1
            await asyncio.sleep(0.01)
            return next(messages)

        async def send(message):
            sent.append(message)

        await RequestLoggingMiddleware(app, set())(scope, receive, send)
        assert current_log_context() is None
        assert [m["body"] for m in sent[1:]] == [b"part1", b"part2"]
        return dict(sent[0]["headers"])[b"x-request-id"].decode()

    async def run():
        return await asyncio.gather(call("/one"), call("/two"))

    ids = asyncio.run(run())
    assert len(set(ids)) == 2
    for path, request_id in zip(("/one", "/two"), ids):
        records = [r for r in log_records if r["extra"]["request_id"] == request_id]
        assert {r["extra"]["scene"] for r in records} == {path}
        assert {r["extra"]["filename"] for r in records} == {path + ".jpg"}
        assert {r["extra"]["event"] for r in records} >= {"test.worker", "test.background", "detection.completed", "http.completed"}
        summary = payload(next(r for r in records if r["extra"]["event"] == "http.completed"))
        assert summary["request_bytes"] == 6
        assert summary["pre_handler_ms"] >= summary["receive_wait_ms"] >= 10
        assert summary["response_total_ms"] >= summary["pre_handler_ms"]
        assert summary["background_ms"] >= 0
        assert summary["code"] == 1
        assert summary["verdict"] == "REVIEW"
        assert summary["response_complete"] is True
        result = payload(next(r for r in records if r["extra"]["event"] == "detection.completed"))
        assert result["detail_verdict_counts"] == {"PASS": 0, "FAIL": 0, "REVIEW": 1}


@pytest.mark.parametrize("unexpected", [False, True])
def test_error_response_keeps_request_id_and_business_code(log_records, unexpected):
    from app import global_exception_handler, vision_api_exception_handler

    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, skip_paths=set())
    app.add_exception_handler(VisionAPIError, vision_api_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    @app.get("/error")
    async def error():
        with detection_log_context("panel_label", "image.jpg"):
            if unexpected:
                raise RuntimeError("backend exploded")
            raise InvalidParamsError("bad\nparams")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            return await client.get("/error")

    response = asyncio.run(call())
    assert response.status_code == 200
    code = int(ErrorCode.INTERNAL_ERROR if unexpected else ErrorCode.INVALID_PARAMS)
    assert response.json()["code"] == code
    records = [r for r in log_records if r["extra"]["request_id"] == response.headers["X-Request-ID"]]
    assert records
    assert all(r["extra"]["scene"] == "panel_label" for r in records)
    summary = payload(next(r for r in records if r["extra"]["event"] == "http.completed"))
    assert summary["code"] == code
    assert summary["verdict"] is None
    assert sum(r["exception"] is not None for r in records) == int(unexpected)
    assert all("\n" not in r["message"] for r in records)


def test_summary_truncation_is_explicit_and_json_remains_valid():
    summary = json.loads(log_json({"label": "x" * 1000, "order": list(range(70)), "rule": "front", "newline": "A\nB"}))
    assert summary["label"]["chars"] == 1000
    assert summary["label"]["truncated"] is True
    assert summary["order"]["omitted"] == 20
    assert summary["order"]["total"] == 70
    assert summary["rule"] == "front"
    assert summary["newline"] == "A\nB"


def test_background_failure_does_not_replace_sent_verdict(log_records):
    from app import global_exception_handler
    from fastapi.responses import JSONResponse
    from starlette.background import BackgroundTask

    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, skip_paths=set())
    app.add_exception_handler(Exception, global_exception_handler)

    async def fail():
        raise OSError("background unavailable")

    @app.get("/background")
    async def endpoint():
        with detection_log_context("panel_label", "sample.jpg"):
            log_detection_result({"verdict": "PASS"})
        return JSONResponse({"code": 1}, background=BackgroundTask(fail))

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            return await client.get("/background")

    response = asyncio.run(call())
    assert response.json() == {"code": 1}
    records = [r for r in log_records if r["extra"]["request_id"] == response.headers["X-Request-ID"]]
    summary = payload(next(r for r in records if r["extra"]["event"] == "http.completed"))
    assert summary["code"] == 1
    assert summary["verdict"] == "PASS"
    assert summary["response_complete"] is True
    assert any(r["extra"]["event"] == "background.failed" for r in records)
    assert sum(r["exception"] is not None for r in records) == 1


def test_successful_health_probe_is_quiet_but_failure_is_logged(log_records):
    async def check(status):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(message):
            pass

        await RequestLoggingMiddleware(app, {"/health"})({"type": "http", "path": "/health", "method": "GET"}, None, send)

    asyncio.run(check(200))
    assert not log_records
    asyncio.run(check(503))
    assert payload(log_records[-1])["http_status"] == 503


def test_real_panel_request_logs_one_parameter_summary_and_correlated_backflow(
    monkeypatch, tmp_path, log_records
):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from app import vision_api_exception_handler
    from schemas.data_base import MoMResult
    from vie_plugin_panel_label.plugin import panel_label_router

    monkeypatch.setattr(panel_label_router, "get_detector_singleton", lambda: SimpleNamespace(detect=lambda _: MoMResult(status=True, message="ok")))
    monkeypatch.setattr(panel_label_router.backflow_service, "data_dir", str(tmp_path))
    monkeypatch.setattr(panel_label_router.response_builder, "vis_enabled", False)
    monkeypatch.setattr("routers.base_router.record_call", lambda *args: None)
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, skip_paths=set())
    app.add_exception_handler(VisionAPIError, vision_api_exception_handler)
    app.include_router(panel_label_router.get_router())
    request = {
        "product": "test", "type": "test",
        "modelParams": {
            "product_type": "T1", "line_order": "A,null,C", "rule": "front",
            "guideline_coordinates": "0,0,1,1",
            "example_images": [{"FileName": "reference", "FilePath": "private-address" * 200}],
        },
        "unneeded": "large-unused-object" * 100,
    }
    ok, image = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/panel_label_detect", files={"file": ("sample.png", image.tobytes(), "image/png")}, data={"json_data": json.dumps(request)})

    response = asyncio.run(call())
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    records = [r for r in log_records if r["extra"]["request_id"] == request_id]
    assert records
    assert all(r["extra"]["scene"] == "panel_label" for r in records)
    assert all("private-address" not in r["message"] and "large-unused-object" not in r["message"] for r in records)
    params = [r for r in records if r["extra"]["event"] == "request.params"]
    assert len(params) == 1
    assert payload(params[0])["line_order"] == [["A", None, "C"]]
    completed = next(r for r in records if r["extra"]["event"] == "backflow.completed")
    saved = json.loads(Path(payload(completed)["record_path"]).read_text())
    assert saved["request_id"] == request_id
    assert saved["request_params"] == request
    assert payload(next(r for r in records if r["extra"]["event"] == "http.completed"))["verdict"] == "PASS"
