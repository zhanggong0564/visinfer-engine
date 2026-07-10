'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:25:59
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-25 00:00:00
@FilePath     : base_router.py
@Description  :路由基类，封装所有路由共有的功能
'''

import asyncio
import inspect
import json
import os
import re
import time
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from config import settings
from schemas import CommonResponse, ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import InvalidParamsError, InvalidImageError, InternalError
from services import detection_factory
from services.call_stats import record_call
from services.utils.visualize import render_detection_overlay
from routers.upload_persistence import (
    StagedImageWrite,
    decode_image,
    detect_image_extension,
)
from utils import vision_logger
from utils.async_utils import run_sync
from utils.timing import StageTimer

# 数据回流根目录：按 settings.DATA_DIR（相对路径锚定运行 cwd）解析为绝对路径。
# 不再基于 __file__ 推算——本模块会被编译成 .so 装进 venv，__file__ 会指向
# site-packages，导致落盘埋进 venv 且无法挂载持久化。
DATA_DIR = os.path.abspath(settings.DATA_DIR)

UNKNOWN_MODEL_DIR = "_unknown_model"
SAFE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class BackflowTarget:
    """数据回流落盘的目标路径三要素：场景目录 / 型号目录 / 落盘文件名（不含扩展名）。

    由 ``BaseRouter.resolve_backflow_target`` 产出，是框架与各场景之间的边界：
    框架只认这三个字段去拼路径、写文件，不关心它们怎么从文件名/参数推导而来；
    场景专属的命名规则在各自插件 Router 里重写 ``resolve_backflow_target`` 实现。
    """

    scene_dir: str   # 顶层场景目录
    model_dir: str   # 型号目录
    save_stem: str   # 落盘文件名（不含扩展名）


@dataclass
class DecodedUpload:
    image: np.ndarray
    raw_bytes: Optional[bytes]
    extension: str


class BaseRouter(ABC):
    """路由基类，封装所有路由共有的功能"""

    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        self.router = APIRouter()
        self.router_name = router_name
        self.instance = None
        self.detector_type = detector_type
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
                upload = await self._process_image(
                    file,
                    original_filename,
                    received_at,
                    fallback_product_type,
                    timer.record,
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
                        self._persist_record,
                        image=image,
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
                with timer.stage("sanitize_result"):
                    self._sanitize_detail_list_names(result_dict)
                response_result = result_dict
                if settings.VIS_ENABLED:
                    # 引导框仅线标等场景经 inputs.extra 下发；其它场景无此键则不画，渲染器保持场景无关
                    vis_guides = None
                    extra = getattr(inputs, "extra", None)
                    guideline = extra.get("guideline") if isinstance(extra, dict) else None
                    if guideline:
                        vis_guides = [tuple(guideline)]
                    # 绘制+JPEG 编码是 CPU 操作，丢线程池避免阻塞事件循环
                    with timer.stage("vis_render"):
                        vis_b64 = await run_sync(
                            render_detection_overlay,
                            image,
                            result_dict.get("detailList", []),
                            guides=vis_guides,
                            max_side=settings.VIS_MAX_SIDE,
                            jpeg_quality=settings.VIS_JPEG_QUALITY,
                        )
                    # 注入副本：vis_image 只进响应，不进数据回流 record.json（落盘已有原图）
                    response_result = {**result_dict, "vis_image": vis_b64}
                else:
                    timer.record("vis_render", 0.0)
                with timer.stage("response_build"):
                    result = CommonResponse(
                        code=int(ErrorCode.SUCCESS),
                        message=ERROR_CODE_MESSAGES[ErrorCode.SUCCESS],
                        result=response_result,
                    )
            except Exception as exc:
                # 响应封装失败会跳过 FastAPI BackgroundTasks；此处必须同步回流，
                # 否则"检测成功但 CommonResponse 校验失败"的关键样本会丢失。
                vision_logger.exception("响应封装失败，执行同步数据回流 filename={}: {}", original_filename, exc)
                with timer.stage("persist_response_error_record"):
                    await run_sync(
                        self._persist_record,
                        image=image,
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
                    self._persist_record,
                    image=image,
                    original_filename=original_filename,
                    raw_json=json_data,
                    result_dict=result_dict,
                    latency_ms=latency_ms,
                    received_at=received_at,
                    fallback_product_type=fallback_product_type,
                )
                # 调用统计：按检测结论记 ok/ng（与回流落盘共用 _classify_result 二分；
                # 异常路径在外层 _handle_request 统一记 error，比落盘目录多一个分类）
                background_tasks.add_task(
                    run_sync,
                    record_call,
                    self.detector_type,
                    self._classify_result(result_dict),
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
    def _sanitize_detail_list_names(result_dict: Any) -> None:
        """CommonResponse validation 前兜底清洗 detailList.name，避免 None 触发 500。"""
        if not isinstance(result_dict, dict):
            return
        result_data = result_dict.get("result") if isinstance(result_dict.get("result"), dict) else result_dict
        detail_list = result_data.get("detailList", [])
        if not isinstance(detail_list, list):
            return
        name_list = []
        for idx, item in enumerate(detail_list):
            if not isinstance(item, dict):
                name_list.append(None)
                vision_logger.warning("detailList item is not dict, idx={}, item={}", idx, item)
                continue
            name = item.get("name")
            if name is None:
                vision_logger.warning("detailList item name is None, idx={}, item={}", idx, item)
                item["name"] = ""
                name = ""
            elif not isinstance(name, str):
                vision_logger.warning(
                    "detailList item name is not str, idx={}, name_type={}, item={}",
                    idx,
                    type(name).__name__,
                    item,
                )
                item["name"] = str(name)
                name = item["name"]
            name_list.append(name)
        vision_logger.info("detailList name list={}", name_list)

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

    async def _process_image(
        self,
        file: UploadFile,
        original_filename: str,
        received_at: str,
        fallback_product_type: Optional[str],
        stage_recorder: Optional[Callable[[str, float], None]] = None,
    ) -> DecodedUpload:
        read_start = time.perf_counter()
        payload = await file.read()
        if stage_recorder:
            stage_recorder("image_read", (time.perf_counter() - read_start) * 1000)

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(payload) > max_bytes:
            raise InvalidImageError(f"图片大小超过上限 {settings.MAX_UPLOAD_MB}MB")

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

        pending_paths = self._resolve_backflow_paths(
            original_filename,
            received_at,
            fallback_product_type,
            "pending",
            image_extension=extension,
        )

        async def timed(stage_name, func, *args):
            start = time.perf_counter()
            try:
                return await run_sync(func, *args)
            finally:
                if stage_recorder:
                    stage_recorder(
                        stage_name,
                        (time.perf_counter() - start) * 1000,
                    )

        image_result, stage_result = await asyncio.gather(
            timed("image_decode", decode_image, payload),
            timed(
                "image_stage_write",
                StagedImageWrite.write,
                payload,
                pending_paths["image_path"],
            ),
            return_exceptions=True,
        )

        if isinstance(image_result, Exception):
            if isinstance(stage_result, StagedImageWrite):
                await run_sync(stage_result.discard)
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
            await run_sync(stage_result.discard)
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
        safe_filename = self._safe_client_filename(original_filename)
        stem = os.path.splitext(safe_filename)[0] or f"unknown_{int(time.time() * 1000)}"
        model_dir = (
            self._sanitize_dir_name(fallback_product_type)
            if fallback_product_type
            else UNKNOWN_MODEL_DIR
        )
        return BackflowTarget(scene_dir=self.detector_type, model_dir=model_dir, save_stem=stem)

    @staticmethod
    def _sanitize_dir_name(name: str) -> str:
        """把 product_type 当目录名前清洗，去掉路径分隔符避免越界。"""
        sanitized = re.sub(r"[\\/]+", "_", str(name)).strip().strip(".")
        return sanitized or UNKNOWN_MODEL_DIR

    @staticmethod
    def _safe_client_filename(filename: str) -> str:
        """只保留客户端文件名本身，兼容 Windows 与 POSIX 路径分隔符。"""
        normalized = (filename or "unknown.jpg").replace("\\", "/")
        return normalized.rsplit("/", 1)[-1] or "unknown.jpg"

    @staticmethod
    def _safe_path(root: str, *parts: str) -> str:
        """构造并校验路径，确保规范化后的结果仍位于 root 下。"""
        root_path = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(root_path, *parts))
        if os.path.commonpath((root_path, candidate)) != root_path:
            raise ValueError("数据回流路径越界")
        return candidate

    @staticmethod
    def _classify_result(result_dict: dict) -> str:
        """按检测结论把样本分到 ok / ng 目录。

        判定依据为响应顶层 status（MoMResult.to_dict 输出字符串 'true'/'false'）。
        仅 status 明确为真（True / 'true'）时归 ok；其余一律归 ng——包括
        status 为假、未检出任何信息（无 status 字段）、以及检测异常落盘的
        {"error": ...} 记录。
        """
        status = result_dict.get("status")
        if status is True or (isinstance(status, str) and status.strip().lower() == "true"):
            return "ok"
        return "ng"

    def _resolve_backflow_paths(
        self,
        original_filename: str,
        received_at: str,
        fallback_product_type: Optional[str],
        verdict_dir: str,
        image_extension: Optional[str] = None,
    ) -> dict:
        target = self.resolve_backflow_target(original_filename, fallback_product_type)
        if image_extension is not None:
            if image_extension not in SAFE_IMAGE_EXTENSIONS:
                raise ValueError(f"不支持的回流图片扩展名: {image_extension}")
            ext = image_extension
        else:
            safe_filename = self._safe_client_filename(original_filename)
            ext = os.path.splitext(safe_filename)[1].lower()
            if ext not in SAFE_IMAGE_EXTENSIONS:
                ext = ".jpg"
        date_dir = received_at[:10]  # YYYY-MM-DD
        scene_dir = self._sanitize_dir_name(target.scene_dir)
        target_model_dir = self._sanitize_dir_name(target.model_dir)
        save_stem = self._sanitize_dir_name(target.save_stem)
        model_dir = self._safe_path(
            DATA_DIR, scene_dir, date_dir, target_model_dir, verdict_dir
        )
        image_dir = self._safe_path(model_dir, "images")
        record_dir = self._safe_path(model_dir, "records")
        return {
            "scene_dir": scene_dir,
            "model_dir": target_model_dir,
            "save_stem": save_stem,
            "verdict_dir": verdict_dir,
            "image_dir": image_dir,
            "record_dir": record_dir,
            "image_path": self._safe_path(image_dir, f"{save_stem}{ext}"),
            "record_path": self._safe_path(record_dir, f"{save_stem}.json"),
        }

    def _persist_image(
        self,
        image: np.ndarray,
        original_filename: str,
        received_at: str,
        fallback_product_type: Optional[str] = None,
        verdict_dir: str = "pending",
    ) -> None:
        """图片解码后立即落盘。最终 ok/ng 记录由 _persist_record 补齐。"""
        try:
            paths = self._resolve_backflow_paths(
                original_filename,
                received_at,
                fallback_product_type,
                verdict_dir,
            )
            os.makedirs(paths["image_dir"], exist_ok=True)
            if not cv2.imwrite(paths["image_path"], image):
                vision_logger.warning(f"数据回流图片写入失败 path={paths['image_path']}")
                return
            vision_logger.info(f"数据回流图片已即时落盘 path={paths['image_path']}")
        except Exception as e:
            vision_logger.warning(f"数据回流图片即时落盘失败 filename={original_filename}: {e}")

    def _persist_record(
        self,
        image: np.ndarray,
        original_filename: str,
        raw_json: str,
        result_dict: dict,
        latency_ms: float,
        received_at: str,
        fallback_product_type: Optional[str] = None,
    ) -> None:
        """后台数据回流：image + json 分文件夹落盘。失败只 warning，不影响接口。

        落盘路径（场景目录 / 型号目录 / 文件名）由 ``resolve_backflow_target`` 决定，
        场景专属命名规则在各插件 Router 重写该钩子；本方法只负责通用的目录与 IO。
        """
        try:
            # 在型号目录下再按检测结论分 ok / ng；程序异常失败保留 pending，不移动图片。
            # 例：data/中压线标检验/2026-06-05/TK2/ng/images/1764780181920.jpg
            is_error_record = isinstance(result_dict, dict) and "error" in result_dict
            verdict_dir = "pending" if is_error_record else self._classify_result(result_dict)
            paths = self._resolve_backflow_paths(
                original_filename,
                received_at,
                fallback_product_type,
                verdict_dir,
            )
            os.makedirs(paths["record_dir"], exist_ok=True)

            if not is_error_record:
                os.makedirs(paths["image_dir"], exist_ok=True)
                pending_paths = self._resolve_backflow_paths(
                    original_filename,
                    received_at,
                    fallback_product_type,
                    "pending",
                )
                if os.path.exists(pending_paths["image_path"]):
                    os.replace(pending_paths["image_path"], paths["image_path"])
                    vision_logger.info(
                        f"数据回流图片移动完成 src={pending_paths['image_path']} dst={paths['image_path']}"
                    )
                elif not cv2.imwrite(paths["image_path"], image):
                    vision_logger.warning(f"数据回流图片写入失败 path={paths['image_path']}")
                    return

            try:
                request_params = json.loads(raw_json)
            except json.JSONDecodeError:
                request_params = raw_json

            record = {
                "received_at": received_at,
                "detector_type": self.detector_type,
                "original_filename": original_filename,
                "saved_scene_dir": paths["scene_dir"],
                "saved_model_dir": paths["model_dir"],
                "verdict": "error" if is_error_record else verdict_dir,
                "request_params": request_params,
                "latency_ms": latency_ms,
                "result": result_dict,
            }
            with open(paths["record_path"], "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            vision_logger.warning(f"数据回流落盘失败 filename={original_filename}: {e}")

    def get_router(self):
        return self.router
