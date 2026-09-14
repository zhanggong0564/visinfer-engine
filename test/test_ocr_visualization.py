"""Public OCR response, explicit coordinate spaces, and packaged Chinese font."""

import asyncio
import base64
from dataclasses import asdict
from importlib.resources import files
from types import SimpleNamespace

import cv2
import numpy as np

from routers.response_builder import ResponseBuilder
from routers.visualization import render_detection_overlay
from utils.visualization_text import legend_font
from schemas.common import CommonResponse, DetectionItemResponse
from services.base import CoordinateSpace, OCRToken, Region


def evidence(space=CoordinateSpace.PIXEL):
    return asdict(OCRToken("堵头 Plug", Region(((.2, .3), (.8, .3), (.8, .6), (.2, .6)), space), .97, .85))


def test_common_response_preserves_ocr_tokens_and_openapi():
    response = CommonResponse(code=1, message="ok", result={
        "detailList": [{"verdict": "REVIEW", "status": "false", "ocr_tokens": [evidence()]}],
        "status": "false", "verdict": "REVIEW", "error_msg": "", "message": "review",
    })
    data = response.model_dump(mode="json")
    entry = data["result"]["detailList"][0]["ocr_tokens"][0]
    assert entry == {
        "text": "堵头 Plug", "recognition_score": .97, "detection_score": .85,
        "region": {"space": "PIXEL", "polygon": [[.2, .3], [.8, .3], [.8, .6], [.2, .6]]},
    }
    assert "OCRToken" in CommonResponse.model_json_schema()["$defs"]
    assert DetectionItemResponse(name="legacy").ocr_tokens is None


def test_shared_renderer_uses_detail_coordinate_not_ocr(monkeypatch):
    captured = []
    monkeypatch.setattr(cv2, "polylines", lambda canvas, points, *args: captured.append(points[0].tolist()))
    result = render_detection_overlay(np.zeros((100, 100, 3), np.uint8), [{
        "name": "FQ001811", "coordinate": [10, 20, 80, 20, 80, 40, 10, 40],
        "ocr_tokens": [evidence()],
    }])
    assert result
    assert captured == [[[10, 20], [80, 20], [80, 40], [10, 40]]]


def test_chinese_legend_stays_in_original_top_left_canvas():
    image = np.full((200, 400, 3), 255, np.uint8)
    result = render_detection_overlay(image, [{
        "name": "堵头：FQ001811", "coordinate": [40, 90, 160, 90, 160, 110, 40, 110],
    }])
    output = cv2.imdecode(np.frombuffer(base64.b64decode(result.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    assert output.shape == image.shape
    assert np.any(output[6:50, 6:250] < 200)
    assert np.all(output[160:, :] > 245)
    font = legend_font(18)
    assert bytes(font.getmask("堵")) != bytes(font.getmask("头"))
    assert "SIL OPEN FONT LICENSE" in files("utils").joinpath("fonts/OFL.txt").read_text()


def test_quadrilateral_guides_follow_sloped_edges():
    result = render_detection_overlay(np.full((300, 300, 3), 255, np.uint8), [], guides=[
        (.1, .1, .9, .3, .8, .9, .2, .8),
    ])
    output = cv2.imdecode(np.frombuffer(base64.b64decode(result.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    blue = (output[:, :, 0] > 150) & (output[:, :, 1] < 100) & (output[:, :, 2] < 100)
    assert np.count_nonzero(blue[52:69, 140:161]) > 0
    assert np.count_nonzero(blue[25:35, 140:161]) == 0


def test_response_builder_passes_multiple_guides_and_keeps_evidence(monkeypatch):
    captured = []
    monkeypatch.setattr("routers.response_builder.render_detection_overlay",
                        lambda image, details, **kwargs: captured.append(kwargs["guides"]) or "image")
    guides = [(0, 0, .4, 0, .4, 1, 0, 1), (.6, 0, 1, 0, 1, 1, .6, 1)]
    result = asyncio.run(ResponseBuilder(True, 1280, 85).build(
        np.zeros((20, 20, 3), np.uint8),
        {"detailList": [{"ocr_tokens": [evidence()]}], "status": "false", "error_msg": "", "message": ""},
        SimpleNamespace(extra={"guidelines": guides}),
    ))
    assert captured == [guides]
    assert result.result.detailList[0].ocr_tokens[0].text == "堵头 Plug"


def test_scene_projection_changes_only_drawing_input(monkeypatch):
    captured = {}
    source = {"detailList": [{"name": "FQ001811"}], "status": "true", "error_msg": "", "message": ""}
    drawing = [{"name": "堵头：FQ001811"}]

    def render(image, detail_list, **kwargs):
        captured["details"] = detail_list
        captured.update(kwargs)
        return "image"

    monkeypatch.setattr("routers.response_builder.render_detection_overlay", render)
    response = asyncio.run(ResponseBuilder(True, 1280, 85).build(
        np.zeros((20, 20, 3), np.uint8), source,
        SimpleNamespace(extra={"guideline": (.1, .2, .5, .4), "visualization_details": drawing}),
    ))
    assert captured["guides"] == [(.1, .2, .5, .4)]
    assert captured["details"] is drawing
    assert "group_ocr" not in captured
    assert response.result.detailList[0].name == "FQ001811"
    assert source["detailList"][0]["name"] == "FQ001811"
