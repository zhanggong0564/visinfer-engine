import asyncio
from types import SimpleNamespace

import numpy as np

from routers.response_builder import ResponseBuilder
from schemas import ERROR_CODE_MESSAGES, ErrorCode


def _result(detail_list=None):
    return {
        "status": "true",
        "detailList": detail_list or [],
        "error_msg": "",
        "message": "检测成功",
    }


def test_sanitize_detail_list_names_handles_nested_and_flat_results():
    flat = {"detailList": [{"name": None}, {"name": 12}, {}]}
    nested = {"result": {"detailList": [{"name": None}]}}

    ResponseBuilder.sanitize_detail_list_names(flat)
    ResponseBuilder.sanitize_detail_list_names(nested)

    assert flat["detailList"] == [
        {"name": ""},
        {"name": "12"},
        {"name": ""},
    ]
    assert nested["result"]["detailList"] == [{"name": ""}]


def test_build_injects_visualization_only_into_response(monkeypatch):
    calls = []

    def render(image, detail_list, **kwargs):
        calls.append((image, detail_list, kwargs))
        return "data:image/jpeg;base64,abc"

    async def controlled_run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("routers.response_builder.render_detection_overlay", render)
    monkeypatch.setattr("routers.response_builder.run_sync", controlled_run_sync)
    source = _result()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    response = asyncio.run(
        ResponseBuilder(True, 1280, 85).build(
            image,
            source,
            SimpleNamespace(extra={"guideline": [1, 2, 3, 4]}),
        )
    )

    assert "vis_image" not in source
    assert response.result.vis_image == "data:image/jpeg;base64,abc"
    assert calls[0][0] is image
    assert calls[0][1] is source["detailList"]
    assert calls[0][2] == {
        "guides": [(1, 2, 3, 4)],
        "max_side": 1280,
        "jpeg_quality": 85,
    }


def test_build_disabled_skips_renderer_and_records_zero(monkeypatch):
    def unexpected_render(*args, **kwargs):
        raise AssertionError("renderer must not be called")

    monkeypatch.setattr(
        "routers.response_builder.render_detection_overlay", unexpected_render
    )
    stages = []
    source = _result([{"name": None, "status": False}])

    response = asyncio.run(
        ResponseBuilder(False, 1280, 85).build(
            np.zeros((2, 2, 3), dtype=np.uint8),
            source,
            SimpleNamespace(extra={}),
            lambda name, duration: stages.append((name, duration)),
        )
    )

    assert stages == [("vis_render", 0.0)]
    assert response.code == int(ErrorCode.SUCCESS)
    assert response.message == ERROR_CODE_MESSAGES[ErrorCode.SUCCESS]
    assert response.result.detailList[0].name == ""
