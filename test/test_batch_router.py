import asyncio

import numpy as np
import pytest
from fastapi import BackgroundTasks

from routers.base_batch_router import BaseBatchRouter
from routers.upload_processor import DecodedUpload
from schemas.exceptions import InvalidImageError, InvalidParamsError


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeUpload:
    def __init__(self, filename):
        self.filename = filename


class _BatchRouter(BaseBatchRouter):
    def __init__(self):
        super().__init__(
            router_name="batch",
            api_path="/batch",
            summary="batch",
            description="batch",
            detector_type="mvs",
            response_model=dict,
            min_files=2,
        )

    def request_schema(self, json_dict):
        return json_dict

    def build_batch_response(self, result_dict):
        return {"code": 1, "message": "成功", "result": result_dict}


def _prepare(monkeypatch, status="PASS"):
    router = _BatchRouter()
    events = []
    persisted = []
    stats = []

    async def process(file, original_filename, **kwargs):
        events.append(f"upload:{original_filename}")
        return DecodedUpload(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            raw_bytes=None,
            extension=".jpg",
        )

    class Detector:
        def inspect(self, images, params):
            events.append(("inspect", [name for name, _ in images], params))
            return {"status": status, "images": len(images)}

    monkeypatch.setattr(router.upload_processor, "process", process)
    monkeypatch.setattr(router, "get_detector_singleton", Detector)
    monkeypatch.setattr(
        router.backflow_service,
        "persist_record",
        lambda **kwargs: persisted.append(kwargs),
    )
    monkeypatch.setattr(
        "routers.base_batch_router.record_call",
        lambda scene, verdict: stats.append((scene, verdict)),
    )
    return router, events, persisted, stats


def test_batch_router_preserves_order_and_records_once(monkeypatch):
    router, events, persisted, stats = _prepare(monkeypatch)
    background = BackgroundTasks()
    response = _run(
        router._handle_batch_request(
            background_tasks=background,
            files=[_FakeUpload("box-1.jpg"), _FakeUpload("box-2.jpg")],
            json_data='{"selected_item_key":"plug"}',
        )
    )
    _run(background())

    assert response["result"] == {"status": "PASS", "images": 2}
    assert events[-1] == (
        "inspect",
        ["box-1.jpg", "box-2.jpg"],
        {"selected_item_key": "plug"},
    )
    assert len(persisted) == 2
    assert {item["batch_id"] for item in persisted} == {
        persisted[0]["batch_id"]
    }
    assert [item["batch_index"] for item in persisted] == [1, 2]
    assert all(item["batch_size"] == 2 for item in persisted)
    assert stats == [("mvs", "ok")]


@pytest.mark.parametrize(
    ("status", "verdict"),
    [("FAIL", "ng"), ("REVIEW", "ng")],
)
def test_batch_router_classifies_non_pass_results(monkeypatch, status, verdict):
    router, _, _, stats = _prepare(monkeypatch, status=status)
    background = BackgroundTasks()
    _run(
        router._handle_batch_request(
            background_tasks=background,
            files=[_FakeUpload("box-1.jpg"), _FakeUpload("box-2.jpg")],
            json_data="{}",
        )
    )
    _run(background())

    assert stats == [("mvs", verdict)]


def test_batch_router_rejects_too_few_files_and_records_error(monkeypatch):
    router, _, persisted, stats = _prepare(monkeypatch)

    with pytest.raises(InvalidParamsError, match="至少需要上传 2 张"):
        _run(
            router._handle_batch_request(
                background_tasks=BackgroundTasks(),
                files=[_FakeUpload("box-1.jpg")],
                json_data="{}",
            )
        )

    assert persisted == []
    assert stats == [("mvs", "error")]


def test_batch_router_persists_already_processed_items_on_upload_failure(
    monkeypatch,
):
    router, _, persisted, stats = _prepare(monkeypatch)
    calls = 0

    async def process(file, original_filename, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InvalidImageError("bad image")
        return DecodedUpload(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            raw_bytes=b"first",
            extension=".jpg",
        )

    monkeypatch.setattr(router.upload_processor, "process", process)

    with pytest.raises(InvalidImageError, match="bad image"):
        _run(
            router._handle_batch_request(
                background_tasks=BackgroundTasks(),
                files=[_FakeUpload("box-1.jpg"), _FakeUpload("box-2.jpg")],
                json_data="{}",
            )
        )

    assert len(persisted) == 1
    assert persisted[0]["original_filename"] == "box-1.jpg"
    assert persisted[0]["result_dict"]["error"] == "bad image"
    assert stats == [("mvs", "error")]


def test_batch_router_persists_all_items_once_on_response_failure(monkeypatch):
    router, _, persisted, stats = _prepare(monkeypatch)

    def fail_response(result):
        raise RuntimeError("response failed")

    monkeypatch.setattr(router, "build_batch_response", fail_response)

    with pytest.raises(RuntimeError, match="response failed"):
        _run(
            router._handle_batch_request(
                background_tasks=BackgroundTasks(),
                files=[_FakeUpload("box-1.jpg"), _FakeUpload("box-2.jpg")],
                json_data="{}",
            )
        )

    assert len(persisted) == 2
    assert all(
        item["result_dict"]["error"] == "response failed"
        for item in persisted
    )
    assert stats == [("mvs", "error")]
