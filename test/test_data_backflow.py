"""数据回流落盘单元测试

回归点：检测失败（如型号未注册）时，图片仍须落盘以便收集新型号样本。
"""
import asyncio
import json
import sys
import types

import numpy as np
import pytest
from fastapi import BackgroundTasks

python_multipart = types.ModuleType("python_multipart")
python_multipart.__version__ = "0.0.20"
sys.modules.setdefault("python_multipart", python_multipart)

from routers.base_router import BaseRouter, DecodedUpload
from routers.upload_persistence import write_bytes_atomically
from schemas.exceptions import ProductNotRegisteredError


def _run(coro):
    """兼容 PPOCR 环境 Python 3.10.0 的 executor 关闭问题。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeUpload:
    """最小 UploadFile 替身：只需 filename + async read。"""

    def __init__(self, filename="线标检验FU211-1779526099406.jpg"):
        self.filename = filename

    async def read(self):
        return b"fake-image-bytes"


class _Router(BaseRouter):
    def __init__(self):
        super().__init__(
            router_name="t_router",
            api_path="/t_detect",
            summary="t",
            description="t",
            detector_type="panel_label",
        )

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return None


def _make_router(monkeypatch, detect_side_effect):
    router = _Router()
    events = []

    async def _fake_process_image(*args, **kwargs):
        events.append("pending_published")
        return DecodedUpload(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_bytes=None,
            extension=".jpg",
        )

    monkeypatch.setattr(router, "_process_image", _fake_process_image)

    class _Detector:
        def detect(self, inputs):
            events.append("detect")
            return detect_side_effect()

    monkeypatch.setattr(router, "get_detector_singleton", lambda: _Detector())
    monkeypatch.setattr("routers.base_router.record_call", lambda scene, verdict: None)

    # 回流方法记录调用而不真正落盘
    calls = []
    monkeypatch.setattr(router, "_persist_record", lambda **kw: calls.append(kw))
    return router, calls, events


def test_persist_called_when_detect_fails(monkeypatch):
    """检测抛异常（型号未注册）时，仍应调用数据回流落盘并传出型号兜底目录。"""

    def _raise():
        raise ProductNotRegisteredError(
            "产品型号 'FU211' 未注册", product_type="FU211", scenario="panel_label"
        )

    router, calls, events = _make_router(monkeypatch, _raise)
    bg = BackgroundTasks()

    with pytest.raises(ProductNotRegisteredError):
        _run(
            router._handle_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )

    # 即便检测失败，回流也必须落盘
    assert events[:2] == ["pending_published", "detect"]
    assert len(calls) == 1, "检测失败时错误记录未落盘"
    assert calls[0]["original_filename"] == "线标检验FU211-1779526099406.jpg"


def test_image_persisted_before_detect(monkeypatch):
    """图片解码后、模型检测前，应先同步保存原图。"""
    def _detect():
        return {"detailList": [], "status": "true", "error_msg": "", "message": "ok"}

    router, calls, events = _make_router(monkeypatch, _detect)
    bg = BackgroundTasks()
    _run(
        router._process_detect_request(
            background_tasks=bg,
            file=_FakeUpload(),
            json_data='{"modelParams": {"product_type": "FU211"}}',
        )
    )

    assert events[:2] == ["pending_published", "detect"]


def test_persist_called_when_response_validation_fails(monkeypatch):
    """检测已成功但响应封装失败时，仍要同步数据回流，否则异常样本会丢。"""

    bad_result = {
        "detailList": [
            {
                "status": "false",
                "scene": "line",
                "coordinate": [],
                "accuracy": 0.8,
                "name": None,
            }
        ],
        "status": "false",
        "error_msg": "",
        "message": "mismatch",
    }
    router, calls, events = _make_router(monkeypatch, lambda: dict(bad_result))
    monkeypatch.setattr(router, "_sanitize_detail_list_names", lambda result_dict: None)
    bg = BackgroundTasks()

    with pytest.raises(Exception):
        _run(
            router._process_detect_request(
                background_tasks=bg,
                file=_FakeUpload(),
                json_data='{"modelParams": {"product_type": "FU211"}}',
            )
        )

    assert events[:2] == ["pending_published", "detect"]
    assert len(calls) == 1, "响应封装失败时错误记录未落盘"
    assert calls[0]["result_dict"]["error"]
    assert calls[0]["result_dict"]["result"]["detailList"][0]["name"] is None


def test_default_backflow_target_is_scene_agnostic():
    """框架默认 resolve_backflow_target 不解析文件名：场景=detector_type，
    型号取 product_type 兜底，文件名沿用原名（线标式中文场景解析归插件）。"""
    router = _Router()  # detector_type="panel_label"，未重写钩子 → 用框架默认
    target = router.resolve_backflow_target(
        "AI-中压线标检验TK2-1-1764780181920.jpg", fallback_product_type="TK2"
    )
    assert target.scene_dir == "panel_label"
    assert target.model_dir == "TK2"
    assert target.save_stem == "AI-中压线标检验TK2-1-1764780181920"


def test_default_backflow_target_unknown_model():
    """无 product_type 兜底时型号目录回退 _unknown_model。"""
    from routers.base_router import UNKNOWN_MODEL_DIR

    router = _Router()
    target = router.resolve_backflow_target("anything.jpg", fallback_product_type=None)
    assert target.model_dir == UNKNOWN_MODEL_DIR


def test_default_backflow_target_strips_client_path():
    """客户端文件名中的目录部分不能进入回流保存名。"""
    router = _Router()
    target = router.resolve_backflow_target("../../escape.jpg", fallback_product_type="TK2")
    assert target.save_stem == "escape"


def test_persist_record_keeps_outputs_under_data_dir(monkeypatch, tmp_path):
    """恶意文件名和型号不能让回流文件逃出 DATA_DIR。"""
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()

    router._persist_record(
        raw_image_bytes=b"raw",
        image_extension=".jpg",
        original_filename="../../escape.php",
        raw_json="{}",
        result_dict={"status": "true"},
        latency_ms=1.0,
        received_at="2026-06-12T10:00:00.000",
        fallback_product_type="../model",
    )

    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    assert all(path.resolve().is_relative_to(tmp_path.resolve()) for path in files)
    assert any(path.suffix == ".jpg" for path in files)


def test_success_record_moves_pending_image_to_verdict_dir(monkeypatch, tmp_path):
    """检测得到 ok/ng 结论后，把 pending 图片移动到对应 verdict 目录。"""
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    pending = router._resolve_backflow_paths("sample.jpg", "2026-06-12T10:00:00.000", "TK2", "pending", ".jpg")
    write_bytes_atomically(b"raw", pending["image_path"])
    pending_image = tmp_path / "panel_label" / "2026-06-12" / "TK2" / "pending" / "images" / "sample.jpg"
    ok_image = tmp_path / "panel_label" / "2026-06-12" / "TK2" / "ok" / "images" / "sample.jpg"
    assert pending_image.exists()

    router._persist_record(
        original_filename="sample.jpg",
        raw_json='{"modelParams": {"product_type": "TK2"}}',
        result_dict={"detailList": [], "status": "true", "error_msg": "", "message": "ok"},
        latency_ms=1.0,
        received_at="2026-06-12T10:00:00.000",
        fallback_product_type="TK2",
    )

    assert not pending_image.exists()
    assert ok_image.exists()


def test_error_record_keeps_pending_image_and_writes_pending_json(monkeypatch, tmp_path):
    """程序异常失败时不移动图片，只把错误信息写到 pending/records。"""
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    pending = router._resolve_backflow_paths("sample.jpg", "2026-06-12T10:00:00.000", "TK2", "pending", ".jpg")
    write_bytes_atomically(b"raw", pending["image_path"])
    pending_image = tmp_path / "panel_label" / "2026-06-12" / "TK2" / "pending" / "images" / "sample.jpg"
    pending_record = tmp_path / "panel_label" / "2026-06-12" / "TK2" / "pending" / "records" / "sample.json"
    ng_image = tmp_path / "panel_label" / "2026-06-12" / "TK2" / "ng" / "images" / "sample.jpg"
    assert pending_image.exists()

    router._persist_record(
        original_filename="sample.jpg",
        raw_json='{"modelParams": {"product_type": "TK2"}}',
        result_dict={"error": "boom"},
        latency_ms=1.0,
        received_at="2026-06-12T10:00:00.000",
        fallback_product_type="TK2",
    )

    assert pending_image.exists()
    assert pending_record.exists()
    assert not ng_image.exists()
    assert json.loads(pending_record.read_text(encoding="utf-8"))["result"]["error"] == "boom"
