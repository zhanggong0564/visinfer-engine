import asyncio
from dataclasses import dataclass
import time
from typing import Callable, Optional

import numpy as np
from fastapi import UploadFile

from routers.upload_persistence import (
    StagedImageWrite,
    decode_image,
    detect_image_extension,
)
from schemas.exceptions import InvalidImageError
from utils import vision_logger
from utils.async_utils import run_sync


StageRecorder = Callable[[str, float], None]
PendingPathResolver = Callable[[str], str]


@dataclass(frozen=True)
class DecodedUpload:
    image: np.ndarray
    raw_bytes: Optional[bytes]
    extension: str


class UploadProcessor:
    def __init__(self, max_upload_mb: int) -> None:
        self.max_upload_mb = max_upload_mb

    async def process(
        self,
        file: UploadFile,
        original_filename: str,
        pending_path_resolver: PendingPathResolver,
        stage_recorder: Optional[StageRecorder] = None,
    ) -> DecodedUpload:
        read_start = time.perf_counter()
        payload = await file.read()
        if stage_recorder:
            stage_recorder("image_read", (time.perf_counter() - read_start) * 1000)
        if len(payload) > self.max_upload_mb * 1024 * 1024:
            raise InvalidImageError(f"图片大小超过上限 {self.max_upload_mb}MB")

        format_start = time.perf_counter()
        try:
            extension = detect_image_extension(payload)
        except ValueError as exc:
            raise InvalidImageError(str(exc)) from exc
        if stage_recorder:
            stage_recorder(
                "image_format_detect",
                (time.perf_counter() - format_start) * 1000,
            )
        pending_path = pending_path_resolver(extension)

        async def timed(stage_name, function, *args):
            started = time.perf_counter()
            try:
                return await run_sync(function, *args)
            finally:
                if stage_recorder:
                    stage_recorder(
                        stage_name,
                        (time.perf_counter() - started) * 1000,
                    )

        async def discard_safely(staged):
            try:
                await run_sync(staged.discard)
            except Exception as exc:
                vision_logger.warning(
                    "数据回流原图暂存清理失败 filename={} error={}",
                    original_filename,
                    exc,
                )

        preparation = asyncio.gather(
            timed("image_decode", decode_image, payload),
            timed("image_stage_write", StagedImageWrite.write, payload, pending_path),
            return_exceptions=True,
        )
        try:
            image_result, stage_result = await asyncio.shield(preparation)
        except asyncio.CancelledError:
            image_result, stage_result = await preparation
            if isinstance(stage_result, StagedImageWrite):
                await discard_safely(stage_result)
            raise
        if isinstance(image_result, Exception):
            if isinstance(stage_result, StagedImageWrite):
                await discard_safely(stage_result)
            raise InvalidImageError("图片读取失败，请检查文件格式") from image_result
        if isinstance(stage_result, Exception):
            vision_logger.warning(
                "数据回流原图暂存失败 filename={} stage=write error={}",
                original_filename,
                stage_result,
            )
            return DecodedUpload(image_result, payload, extension)
        commit_start = time.perf_counter()
        try:
            await run_sync(stage_result.commit)
        except Exception as exc:
            await discard_safely(stage_result)
            vision_logger.warning(
                "数据回流原图发布失败 filename={} stage=commit error={}",
                original_filename,
                exc,
            )
            return DecodedUpload(image_result, payload, extension)
        finally:
            if stage_recorder:
                stage_recorder(
                    "image_commit",
                    (time.perf_counter() - commit_start) * 1000,
                )
        return DecodedUpload(image_result, None, extension)
