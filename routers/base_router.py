'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:25:59
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-25 00:00:00
@FilePath     : base_router.py
@Description  :路由基类，封装所有路由共有的功能
'''

import inspect
import json
import os
import time
from abc import ABC
from datetime import datetime
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from config import settings
from schemas import CommonResponse, ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import InvalidParamsError, InternalError
from services import detection_factory
from services.call_stats import record_call
from utils import vision_logger
from utils.async_utils import run_sync
from utils.timing import StageTimer
from routers.upload_processor import UploadProcessor
from routers.backflow_service import BackflowService, BackflowTarget, UNKNOWN_MODEL_DIR
from routers.response_builder import ResponseBuilder

# 数据回流根目录：按 settings.DATA_DIR（相对路径锚定运行 cwd）解析为绝对路径。
# 不再基于 __file__ 推算——本模块会被编译成 .so 装进 venv，__file__ 会指向
# site-packages，导致落盘埋进 venv 且无法挂载持久化。
DATA_DIR = os.path.abspath(settings.DATA_DIR)



class BaseRouter(ABC):
    """路由基类，封装所有路由共有的功能"""

    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        self.router = APIRouter()
        self.router_name = router_name
        self.instance = None
        self.detector_type = detector_type
        self.backflow_service = BackflowService(self.detector_type, self.resolve_backflow_target, DATA_DIR)
        self.upload_processor = UploadProcessor(settings.MAX_UPLOAD_MB)
        self.response_builder = ResponseBuilder(settings.VIS_ENABLED, settings.VIS_MAX_SIDE, settings.VIS_JPEG_QUALITY)
        # 路由自描述的 Swagger 分组标签；为空时由 RouterRegistry 回退到模块名映射。
        # 让插件无需依赖框架 tag_map 即可声明中文分组名，保持框架对插件零知晓。
        self.tag = tag

        self.router.post(
            api_path,
            summary=summary,
            description=description,
            response_model=CommonResponse,
            response_description="统一检测响应",
        )(self._handle_request)

    def get_detector_singleton(self):
        if self.instance is None:
            detector = detection_factory.get_scenarios(self.detector_type)
            if not detector:
                raise InternalError(
                    f"未找到 {self.detector_type} 检测器",
                    detector_type=self.detector_type,
                )
            self.instance = detector
        return self.instance

    async def _handle_request(
        self,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="检测图片(jpg/png格式)"),
        json_data: str = Form(..., description="产品/物料号/模型参数的JSON字符串"),
    ):
        # 调用统计埋点：每一次调用都要计数——参数校验失败、图片非法等
        # 进不了检测的请求也算，统一记 error；成功路径在内层按检测结论记
        # ok/ng。record_call 自吞异常，统计失败绝不影响检测主流程。
        try:
            return await self._process_detect_request(background_tasks, file, json_data)
        except Exception:
            await run_sync(record_call, self.detector_type, "error")
            raise

    async def _process_detect_request(
        self,
        background_tasks: BackgroundTasks,
        file: UploadFile,
        json_data: str,
    ):
        timer = StageTimer()
        received_at = datetime.now().isoformat(timespec="milliseconds")
        original_filename = file.filename or "unknown.jpg"
        # json_data 可能很长（含 line_order 等），且防止以后夹带 base64 图撑爆日志，截断预览
        json_preview = json_data if len(json_data) <= 500 else f"{json_data[:500]}...(共{len(json_data)}字符)"
        try:
            vision_logger.info(f"接收{self.router_name}请求：图片={original_filename}, json_data={json_preview}")
            with timer.stage("validate_params"):
                request_params = await self._validate_and_parse_params(json_data)
            vision_logger.info(f"校验参数：{request_params}")
            fallback_product_type = self._extract_product_type(request_params)

            with timer.stage("process_image"):
                upload = await self.upload_processor.process(
                    file,
                    original_filename,
                    pending_path_resolver=lambda extension: self.backflow_service.resolve_paths(original_filename, received_at, fallback_product_type, "pending", extension)["image_path"],
                    stage_recorder=timer.record,
                )
            image = upload.image
            with timer.stage("build_inputs"):
                inputs = self.get_inputs(request_params, image)
                if inspect.isawaitable(inputs):
                    inputs = await inputs
            with timer.stage("get_detector"):
                detector = self.get_detector_singleton()

            start = time.time()
            # detect 是同步 CPU/GPU 密集操作，丢到线程池执行，避免阻塞事件循环
            try:
                with timer.stage("detect"):
                    result_info = await run_sync(detector.detect, inputs)
            except Exception as exc:
                # 检测失败（如型号未注册）也要落盘：数据回流的核心价值之一就是
                # 收集这些"没见过"的新型号样本去标注、补型号。异常会跳过下方
                # background_tasks 注册，故此处内联落盘后再重抛，交全局异常处理器响应。
                latency_ms = (time.time() - start) * 1000
                with timer.stage("persist_error_record"):
                    await run_sync(
                        self.backflow_service.persist_record,
                        raw_image_bytes=upload.raw_bytes,
                        image_extension=upload.extension,
                        original_filename=original_filename,
                        raw_json=json_data,
                        result_dict={"error": str(exc)},
                        latency_ms=latency_ms,
                        received_at=received_at,
                        fallback_product_type=fallback_product_type,
                    )
                raise
            end = time.time()
            latency_ms = (end - start) * 1000
            # 用 loguru 占位参数（惰性格式化）：仅当 sink 真要写该级别时才拼接字符串，
            # 避免在热路径上对结果对象做无谓的 str() 求值
            vision_logger.info("检测耗时：{:.4f}秒", end - start)
            vision_logger.debug("原始检测结果：{}", result_info)
            with timer.stage("result_to_dict"):
                result_dict = result_info if isinstance(result_info, dict) else result_info.to_dict()
            try:
                with timer.stage("response_build"):
                    result = await self.response_builder.build(image, result_dict, inputs, stage_recorder=timer.record)
            except Exception as exc:
                # 响应封装失败会跳过 FastAPI BackgroundTasks；此处必须同步回流，
                # 否则"检测成功但 CommonResponse 校验失败"的关键样本会丢失。
                vision_logger.exception("响应封装失败，执行同步数据回流 filename={}: {}", original_filename, exc)
                with timer.stage("persist_response_error_record"):
                    await run_sync(
                        self.backflow_service.persist_record,
                        raw_image_bytes=upload.raw_bytes,
                        image_extension=upload.extension,
                        original_filename=original_filename,
                        raw_json=json_data,
                        result_dict={"error": str(exc), "result": result_dict},
                        latency_ms=latency_ms,
                        received_at=received_at,
                        fallback_product_type=fallback_product_type,
                    )
                raise
            vision_logger.info("返回检测结果")

            with timer.stage("schedule_background_tasks"):
                background_tasks.add_task(
                    run_sync,
                    self.backflow_service.persist_record,
                    raw_image_bytes=upload.raw_bytes,
                    image_extension=upload.extension,
                    original_filename=original_filename,
                    raw_json=json_data,
                    result_dict=result_dict,
                    latency_ms=latency_ms,
                    received_at=received_at,
                    fallback_product_type=fallback_product_type,
                )
                # 调用统计：按检测结论记 ok/ng（与回流落盘共用分类规则；
                # 异常路径在外层 _handle_request 统一记 error，比落盘目录多一个分类）
                background_tasks.add_task(
                    run_sync,
                    record_call,
                    self.detector_type,
                    self.backflow_service.classify_result(result_dict),
                )
            return result
        finally:
            vision_logger.info(
                "请求阶段耗时 router={} detector_type={} file={} {}",
                self.router_name,
                self.detector_type,
                original_filename,
                timer.summary(),
            )

    @staticmethod
    def _extract_product_type(request_params: Any) -> Optional[str]:
        """从请求参数里取 product_type 作为型号兜底；schema 不同的子类可重写。"""
        if isinstance(request_params, dict):
            model_params = request_params.get("modelParams")
            if isinstance(model_params, dict):
                return model_params.get("product_type")
            return None
        model_params = getattr(request_params, "modelParams", None)
        if model_params is None:
            return None
        return getattr(model_params, "product_type", None)

    def get_inputs(self, request_params: Any, image: np.ndarray) -> dict:
        """获取模型输入"""
        raise NotImplementedError("子类必须实现get_inputs方法")

    async def _validate_and_parse_params(self, json_data: str) -> Any:
        """验证和解析参数"""
        try:
            json_dict = json.loads(json_data)
            request_params = self.request_schema(json_dict)
            return request_params
        except json.JSONDecodeError:
            raise InvalidParamsError("json_data 格式非法，需传入标准 JSON 字符串")
        except ValueError as e:
            raise InvalidParamsError(f"参数校验失败: {e}")

    def request_schema(self, json_dict) -> Any:
        """请求参数校验模式"""
        raise NotImplementedError("子类必须实现request_schema方法")

    def resolve_backflow_target(
        self,
        original_filename: str,
        fallback_product_type: Optional[str] = None,
    ) -> BackflowTarget:
        """决定数据回流落盘路径，框架对场景命名规则的唯一扩展点。

        框架默认实现【不解析文件名】，保持场景无关：
          - 场景目录 = ``detector_type``
          - 型号目录 = ``fallback_product_type``（来自请求 product_type），缺省回退 ``_unknown_model``
          - 落盘文件名 = 原始文件名去扩展名

        需要从文件名拆分场景/型号（如线标的 'AI-中压线标检验TK2-1-<ts>.jpg'）的场景，
        在各自插件 Router 里重写本方法即可，框架代码无需改动。
        """
        safe_filename = BackflowService.safe_client_filename(original_filename)
        stem = os.path.splitext(safe_filename)[0] or f"unknown_{int(time.time() * 1000)}"
        model_dir = (
            BackflowService.sanitize_dir_name(fallback_product_type)
            if fallback_product_type
            else UNKNOWN_MODEL_DIR
        )
        return BackflowTarget(scene_dir=self.detector_type, model_dir=model_dir, save_stem=stem)

    def get_router(self):
        return self.router
