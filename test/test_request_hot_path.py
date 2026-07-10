import asyncio
import json
import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import BackgroundTasks

python_multipart = types.ModuleType("python_multipart")
python_multipart.__version__ = "0.0.20"
sys.modules.setdefault("python_multipart", python_multipart)

from routers.base_router import BaseRouter, DecodedUpload
from schemas.exceptions import InvalidImageError

from routers.upload_persistence import (
    StagedImageWrite,
    decode_image,
    detect_image_extension,
    write_bytes_atomically,
)


def _encoded(ext: str) -> bytes:
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


@pytest.mark.parametrize(
    ("encoded_extension", "expected_extension"),
    [(".jpg", ".jpg"), (".png", ".png"), (".bmp", ".bmp")],
)
def test_detect_image_extension_uses_file_signature(
    encoded_extension, expected_extension
):
    assert (
        detect_image_extension(_encoded(encoded_extension)) == expected_extension
    )


def test_detect_image_extension_rejects_unknown_payload():
    with pytest.raises(ValueError, match="不支持"):
        detect_image_extension(b"not-an-image")


def test_decode_image_rejects_invalid_payload():
    with pytest.raises(ValueError, match="解码失败"):
        decode_image(b"not-an-image")


@pytest.mark.parametrize("extension", [".jpg", ".png", ".bmp"])
def test_decode_image_returns_nonempty_three_channel_bgr(extension):
    image = decode_image(_encoded(extension))

    assert image.size > 0
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_staged_write_supports_bare_filename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = _encoded(".jpg")

    staged = StagedImageWrite.write(payload, "sample.jpg")
    staged.commit()

    assert Path("sample.jpg").read_bytes() == payload


def test_staged_write_is_invisible_until_commit(tmp_path):
    final_path = tmp_path / "images" / "sample.jpg"
    payload = _encoded(".jpg")

    staged = StagedImageWrite.write(payload, str(final_path))

    assert not final_path.exists()
    assert Path(staged.temporary_path).read_bytes() == payload
    assert Path(staged.temporary_path).parent == final_path.parent

    staged.commit()

    assert final_path.read_bytes() == payload
    assert not Path(staged.temporary_path).exists()


def test_discard_removes_temporary_file(tmp_path):
    final_path = tmp_path / "sample.png"
    staged = StagedImageWrite.write(_encoded(".png"), str(final_path))

    staged.discard()

    assert not Path(staged.temporary_path).exists()
    assert not final_path.exists()


def test_same_final_path_uses_unique_temporary_files(tmp_path):
    final_path = tmp_path / "sample.jpg"
    first = StagedImageWrite.write(_encoded(".jpg"), str(final_path))
    second = StagedImageWrite.write(_encoded(".jpg"), str(final_path))
    try:
        assert first.temporary_path != second.temporary_path
        assert Path(first.temporary_path).name.startswith(".vie-upload-")
        assert Path(second.temporary_path).name.startswith(".vie-upload-")
    finally:
        first.discard()
        second.discard()


def test_atomic_writer_cleans_up_when_replace_fails(monkeypatch, tmp_path):
    final_path = tmp_path / "sample.bmp"
    temporary_paths = []
    original_write = StagedImageWrite.write

    def capture_write(payload, path):
        staged = original_write(payload, path)
        temporary_paths.append(staged.temporary_path)
        return staged

    monkeypatch.setattr(StagedImageWrite, "write", staticmethod(capture_write))
    monkeypatch.setattr(
        os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("disk failure")),
    )

    with pytest.raises(OSError, match="disk failure"):
        write_bytes_atomically(_encoded(".bmp"), str(final_path))

    assert temporary_paths
    assert all(not Path(path).exists() for path in temporary_paths)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeUpload:
    def __init__(self, payload: bytes, filename: str):
        self.payload = payload
        self.filename = filename

    async def read(self):
        return self.payload


class _Router(BaseRouter):
    def __init__(self):
        super().__init__("test", "/test", "test", "test", "panel_label")

    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return {"image": image}


def test_process_image_publishes_original_bytes_before_return(monkeypatch, tmp_path):
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()
    payload = _encoded(".png")

    upload = _run(
        router._process_image(
            _FakeUpload(payload, "mismatched.jpg"),
            "mismatched.jpg",
            "2026-07-10T10:00:00.000",
            "TK2",
        )
    )

    pending = (
        tmp_path / "panel_label" / "2026-07-10" / "TK2"
        / "pending" / "images" / "mismatched.png"
    )
    assert isinstance(upload, DecodedUpload)
    assert upload.extension == ".png"
    assert upload.raw_bytes is None
    assert pending.read_bytes() == payload


def test_process_image_runs_decode_and_stage_concurrently(monkeypatch, tmp_path):
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()
    both_started = threading.Barrier(2, timeout=2)

    def decode(payload):
        both_started.wait()
        return np.zeros((8, 9, 3), dtype=np.uint8)

    original_write = StagedImageWrite.write

    def stage(payload, final_path):
        both_started.wait()
        return original_write(payload, final_path)

    monkeypatch.setattr("routers.base_router.decode_image", decode)
    monkeypatch.setattr(
        "routers.base_router.StagedImageWrite.write",
        staticmethod(stage),
    )

    upload = _run(
        router._process_image(
            _FakeUpload(_encoded(".jpg"), "sample.jpg"),
            "sample.jpg",
            "2026-07-10T10:00:00.000",
            "TK2",
        )
    )
    assert upload.raw_bytes is None


def test_decode_failure_discards_staged_file(monkeypatch, tmp_path):
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    router = _Router()
    truncated_jpeg = b"\xff\xd8\xffbroken"

    with pytest.raises(InvalidImageError, match="图片读取失败"):
        _run(
            router._process_image(
                _FakeUpload(truncated_jpeg, "broken.jpg"),
                "broken.jpg",
                "2026-07-10T10:00:00.000",
                "TK2",
            )
        )

    assert not list(tmp_path.rglob(".vie-upload-*.tmp"))
    assert not list(tmp_path.rglob("broken.jpg"))


def test_commit_failure_keeps_raw_bytes_and_cleans_temp(monkeypatch, tmp_path):
    monkeypatch.setattr("routers.base_router.DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "routers.base_router.StagedImageWrite.commit",
        lambda self: (_ for _ in ()).throw(OSError("replace failed")),
    )
    router = _Router()
    payload = _encoded(".jpg")

    upload = _run(
        router._process_image(
            _FakeUpload(payload, "sample.jpg"),
            "sample.jpg",
            "2026-07-10T10:00:00.000",
            "TK2",
        )
    )

    assert upload.raw_bytes == payload
    assert not list(tmp_path.rglob(".vie-upload-*.tmp"))
    assert not list(tmp_path.rglob("sample.jpg"))
