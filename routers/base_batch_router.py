"""Router template for multi-image scene requests."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import BackgroundTasks, File, Form, UploadFile

from routers.base_router import BaseRouter
from routers.upload_processor import DecodedUpload
from schemas.exceptions import InvalidParamsError
from services.call_stats import record_call
from utils import vision_logger
from utils.async_utils import run_sync
from utils.timing import StageTimer


@dataclass(frozen=True)
class BatchUpload:
    filename: str
    decoded: DecodedUpload

    @property
    def image(self) -> np.ndarray:
        return self.decoded.image


class BaseBatchRouter(BaseRouter):
    """Framework request lifecycle for ordered multi-image inference."""

    def __init__(
        self,
        router_name: str,
        api_path: str,
        summary: str,
        description: str,
        detector_type: str,
        response_model: Any,
        tag: str | None = None,
        min_files: int = 1,
    ) -> None:
        if min_files < 1:
            raise ValueError("min_files 必须大于等于 1")
        super().__init__(
            router_name=router_name,
            api_path=api_path,
            summary=summary,
            description=description,
            detector_type=detector_type,
            tag=tag,
            register_default_route=False,
        )
        self.min_files = min_files
        self.router.post(
            api_path,
            summary=summary,
            description=description,
            response_model=response_model,
        )(self._handle_batch_request)

    async def _handle_batch_request(
        self,
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(..., description="按顺序上传的图片列表"),
        json_data: str = Form(default="{}", description="可选检验参数 JSON"),
    ):
        try:
            return await self._process_batch_request(
                background_tasks,
                files,
                json_data,
            )
        except Exception:
            await run_sync(record_call, self.detector_type, "error")
            raise

    async def _process_batch_request(
        self,
        background_tasks: BackgroundTasks,
        files: list[UploadFile],
        json_data: str,
    ):
        timer = StageTimer()
        received_at = datetime.now().isoformat(timespec="milliseconds")
        batch_id = uuid4().hex[:12]
        uploads: list[BatchUpload] = []
        fallback_product_type = None
        error_persisted = False
        started = time.time()
        try:
            with timer.stage("validate_params"):
                request_params = await self._validate_and_parse_params(json_data)
                if len(files) < self.min_files:
                    raise InvalidParamsError(
                        f"至少需要上传 {self.min_files} 张图片"
                    )
            fallback_product_type = self._extract_product_type(request_params)

            for index, file in enumerate(files):
                filename = file.filename or f"unknown-{index + 1}.jpg"
                with timer.stage(f"process_image_{index + 1}"):
                    decoded = await self.upload_processor.process(
                        file,
                        filename,
                        pending_path_resolver=lambda extension, name=filename: (
                            self.backflow_service.resolve_paths(
                                name,
                                received_at,
                                fallback_product_type,
                                "pending",
                                extension,
                                batch_id=batch_id,
                                batch_index=index + 1,
                            )["image_path"]
                        ),
                        stage_recorder=lambda name, elapsed, item=index + 1: (
                            timer.record(f"{name}_{item}", elapsed)
                        ),
                    )
                uploads.append(BatchUpload(filename, decoded))

            images = [(upload.filename, upload.image) for upload in uploads]
            with timer.stage("get_detector"):
                detector = self.get_detector_singleton()
            try:
                with timer.stage("inspect"):
                    result_info = await self.inference_admission.run(
                        self.detector_type,
                        detector.inspect,
                        images,
                        request_params,
                    )
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            latency_ms = (time.time() - started) * 1000
            with timer.stage("result_to_dict"):
                result_dict = self._result_to_dict(result_info)
            try:
                with timer.stage("response_build"):
                    response = self.build_batch_response(result_dict)
            except Exception as exc:
                await self._persist_batch(
                    uploads,
                    json_data,
                    {"error": str(exc), "result": result_dict},
                    latency_ms,
                    received_at,
                    fallback_product_type,
                    batch_id,
                    len(files),
                )
                error_persisted = True
                raise

            with timer.stage("schedule_background_tasks"):
                for index, upload in enumerate(uploads, start=1):
                    background_tasks.add_task(
                        run_sync,
                        self.backflow_service.persist_record,
                        original_filename=upload.filename,
                        raw_json=json_data,
                        result_dict=result_dict,
                        latency_ms=latency_ms,
                        received_at=received_at,
                        fallback_product_type=fallback_product_type,
                        raw_image_bytes=upload.decoded.raw_bytes,
                        image_extension=upload.decoded.extension,
                        batch_id=batch_id,
                        batch_index=index,
                        batch_size=len(uploads),
                    )
                background_tasks.add_task(
                    run_sync,
                    record_call,
                    self.detector_type,
                    self.backflow_service.classify_result(result_dict),
                )
            return response
        except Exception as exc:
            if uploads and not error_persisted:
                await self._persist_batch(
                    uploads,
                    json_data,
                    {"error": str(exc)},
                    (time.time() - started) * 1000,
                    received_at,
                    fallback_product_type,
                    batch_id,
                    len(files),
                )
            raise
        finally:
            vision_logger.info(
                "批量请求阶段耗时 router={} detector_type={} batch_id={} files={} {}",
                self.router_name,
                self.detector_type,
                batch_id,
                len(files),
                timer.summary(),
            )

    async def _persist_batch(
        self,
        uploads: list[BatchUpload],
        raw_json: str,
        result_dict: dict,
        latency_ms: float,
        received_at: str,
        fallback_product_type: str | None,
        batch_id: str,
        batch_size: int,
    ) -> None:
        for index, upload in enumerate(uploads, start=1):
            await run_sync(
                self.backflow_service.persist_record,
                original_filename=upload.filename,
                raw_json=raw_json,
                result_dict=result_dict,
                latency_ms=latency_ms,
                received_at=received_at,
                fallback_product_type=fallback_product_type,
                raw_image_bytes=upload.decoded.raw_bytes,
                image_extension=upload.decoded.extension,
                batch_id=batch_id,
                batch_index=index,
                batch_size=batch_size,
            )

    def build_batch_response(self, result_dict: dict) -> Any:
        raise NotImplementedError("子类必须实现 build_batch_response 方法")
