import asyncio
import time

import cv2
import numpy as np
import pytest

from routers.upload_persistence import StagedImageWrite
from routers.upload_processor import DecodedUpload, UploadProcessor
from schemas.exceptions import InvalidImageError


class FakeUpload:
    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return self.payload


def encoded_jpeg():
    ok, payload = cv2.imencode(".jpg", np.zeros((12, 12, 3), dtype=np.uint8))
    assert ok
    return payload.tobytes()


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.fixture(autouse=True)
def inline_run_sync(monkeypatch):
    async def run_inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("routers.upload_processor.run_sync", run_inline)


def test_process_publishes_original_bytes(tmp_path):
    final_path = tmp_path / "pending" / "sample.jpg"
    processor = UploadProcessor(max_upload_mb=1)
    result = run(processor.process(
        FakeUpload(encoded_jpeg()),
        original_filename="sample.jpg",
        pending_path_resolver=lambda extension: str(final_path),
    ))
    assert isinstance(result, DecodedUpload)
    assert result.raw_bytes is None
    assert result.extension == ".jpg"
    assert final_path.read_bytes() == encoded_jpeg()


def test_decode_and_stage_run_concurrently(monkeypatch, tmp_path):
    def slow_decode(payload):
        time.sleep(0.08)
        return np.zeros((1, 1, 3), dtype=np.uint8)

    original_write = __import__(
        "routers.upload_processor", fromlist=["StagedImageWrite"]
    ).StagedImageWrite.write

    def slow_write(payload, path):
        time.sleep(0.08)
        return original_write(payload, path)

    async def cooperative_run_sync(function, /, *args, **kwargs):
        if function is slow_decode:
            await asyncio.sleep(0.08)
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if function is slow_write:
            await asyncio.sleep(0.08)
            return original_write(*args, **kwargs)
        return function(*args, **kwargs)

    monkeypatch.setattr("routers.upload_processor.decode_image", slow_decode)
    monkeypatch.setattr("routers.upload_processor.StagedImageWrite.write", slow_write)
    monkeypatch.setattr("routers.upload_processor.run_sync", cooperative_run_sync)
    started = time.perf_counter()
    run(UploadProcessor(1).process(
        FakeUpload(encoded_jpeg()),
        "sample.jpg",
        lambda extension: str(tmp_path / f"sample{extension}"),
    ))
    assert time.perf_counter() - started < 0.14


def test_rejects_oversized_upload(tmp_path):
    payload = b"\xff\xd8\xff" + b"x" * (1024 * 1024)

    with pytest.raises(InvalidImageError, match="图片大小超过上限 1MB"):
        run(UploadProcessor(1).process(
            FakeUpload(payload),
            "large.jpg",
            lambda extension: str(tmp_path / f"large{extension}"),
        ))

    assert not list(tmp_path.iterdir())


def test_uses_signature_extension(tmp_path):
    resolved_extensions = []

    def resolve(extension):
        resolved_extensions.append(extension)
        return str(tmp_path / f"sample{extension}")

    result = run(UploadProcessor(1).process(
        FakeUpload(encoded_jpeg()),
        "misleading.png",
        resolve,
    ))

    assert result.extension == ".jpg"
    assert resolved_extensions == [".jpg"]
    assert (tmp_path / "sample.jpg").is_file()


def test_stage_write_failure_preserves_raw_bytes(monkeypatch, tmp_path):
    payload = encoded_jpeg()
    monkeypatch.setattr(
        "routers.upload_processor.StagedImageWrite.write",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = run(UploadProcessor(1).process(
        FakeUpload(payload),
        "sample.jpg",
        lambda extension: str(tmp_path / f"sample{extension}"),
    ))

    assert result.raw_bytes == payload
    assert result.extension == ".jpg"


def test_decode_failure_discards_stage(tmp_path):
    with pytest.raises(InvalidImageError, match="图片读取失败"):
        run(UploadProcessor(1).process(
            FakeUpload(b"\xff\xd8\xffbroken"),
            "broken.jpg",
            lambda extension: str(tmp_path / f"broken{extension}"),
        ))

    assert not list(tmp_path.rglob(".vie-upload-*.tmp"))
    assert not (tmp_path / "broken.jpg").exists()


def test_cancellation_waits_and_discards_stage(monkeypatch, tmp_path):
    async def exercise():
        stage_started = asyncio.Event()
        allow_stage_finish = asyncio.Event()
        stage_finished = asyncio.Event()

        stage_write = StagedImageWrite.write

        async def controlled_run_sync(function, /, *args, **kwargs):
            if getattr(function, "__func__", function) is getattr(
                stage_write, "__func__", stage_write
            ):
                async def finish_later():
                    stage_started.set()
                    await allow_stage_finish.wait()
                    try:
                        return function(*args, **kwargs)
                    finally:
                        stage_finished.set()

                return await asyncio.shield(asyncio.create_task(finish_later()))
            return function(*args, **kwargs)

        monkeypatch.setattr("routers.upload_processor.run_sync", controlled_run_sync)
        task = asyncio.create_task(
            UploadProcessor(1).process(
                FakeUpload(encoded_jpeg()),
                "cancelled.jpg",
                lambda extension: str(tmp_path / f"cancelled{extension}"),
            )
        )
        await stage_started.wait()
        task.cancel()
        allow_stage_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        await stage_finished.wait()

    run(exercise())

    assert not list(tmp_path.rglob(".vie-upload-*.tmp"))
    assert not (tmp_path / "cancelled.jpg").exists()
