'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:25:59
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-25 00:00:00
@FilePath     : base_router.py
@Description  :路由基类，封装所有路由共有的功能
'''

import json
import os
import re
import time
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import settings
from schemas import CommonResponse, ErrorCode, ERROR_CODE_MESSAGES
from schemas.exceptions import InvalidParamsError, InvalidImageError, InternalError
from services import detection_factory, rotate_points
from utils import vision_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 文件名形如 "1+X线标检验PE1-A-1779526099406.jpg":
#   最后一段 -<digits> 是 timestamp，去掉扩展名后用贪婪匹配切出
_FILENAME_TS_RE = re.compile(r"^(.+)-(\d+)$")
UNKNOWN_MODEL_DIR = "_unknown_model"


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
        received_at = datetime.now().isoformat(timespec="milliseconds")
        original_filename = file.filename or "unknown.jpg"
        vision_logger.info(f"接收{self.router_name}请求：图片={original_filename}, json_data={json_data}")
        request_params = await self._validate_and_parse_params(json_data)
        vision_logger.info(f"校验参数：{request_params}")

        image, is_rotate = await self._process_image(file)
        inputs = self.get_inputs(request_params, image)
        detector = self.get_detector_singleton()
        start = time.time()
        # detect 是同步 CPU/GPU 密集操作，丢到线程池执行，避免阻塞事件循环
        result_info = await run_in_threadpool(detector.detect, inputs)
        end = time.time()
        latency_ms = (end - start) * 1000
        vision_logger.info(f"检测耗时：{end - start}秒")
        vision_logger.debug(f"原始检测结果：{result_info}")
        if is_rotate:
            w, h, _ = image.shape  ##注意这里是反向的
            result_info = rotate_points(result_info.to_dict(), w, h)
            vision_logger.info(f"旋转后的检测结果：{result_info}")
        result_dict = result_info if isinstance(result_info, dict) else result_info.to_dict()
        result = CommonResponse(
            code=int(ErrorCode.SUCCESS),
            message=ERROR_CODE_MESSAGES[ErrorCode.SUCCESS],
            result=result_dict,
        )
        vision_logger.info("参数校验通过，返回检测结果")

        background_tasks.add_task(
            self._persist_record,
            image=image,
            original_filename=original_filename,
            raw_json=json_data,
            result_dict=result_dict,
            latency_ms=latency_ms,
            received_at=received_at,
            fallback_product_type=self._extract_product_type(request_params),
        )
        return result

    @staticmethod
    def _extract_product_type(request_params: Any) -> Optional[str]:
        """从请求参数里取 product_type 作为型号兜底；schema 不同的子类可重写。"""
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

    async def _process_image(self, file: UploadFile) -> Tuple[np.ndarray, bool]:
        """只负责读取并解码图片，落盘逻辑见 _persist_record"""
        img_bytes = await file.read()
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(img_bytes) > max_bytes:
            vision_logger.error(f"上传图片过大: {len(img_bytes)} bytes > {max_bytes} bytes")
            raise InvalidImageError(f"图片大小超过上限 {settings.MAX_UPLOAD_MB}MB")
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        # imdecode 为同步 CPU 操作，丢到线程池避免阻塞事件循环
        image = await run_in_threadpool(cv2.imdecode, img_array, cv2.IMREAD_COLOR)
        if image is None:
            vision_logger.error("图片读取失败")
            raise InvalidImageError("图片读取失败，请检查文件格式")
        # h, w, _ = image.shape
        # is_rotate = w < h
        # if is_rotate:
        #     # 向左旋转90度
        #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image, False

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """从形如 '1+X线标检验PE1-A-1779526099406.jpg' 的文件名解析 (场景, 型号, timestamp)。

        规则：去扩展名 → 末尾 -<digits> 切出 timestamp → 前半段最后一个中文字符之前是场景名，之后是型号。
        不符合规则时返回 (None, None, None)，由调用方走 _unknown_model 兜底。
        """
        stem = Path(filename).stem
        m = _FILENAME_TS_RE.match(stem)
        if not m:
            return None, None, None
        body, timestamp = m.group(1), m.group(2)
        last_cjk_idx = -1
        for i, ch in enumerate(body):
            if "一" <= ch <= "鿿":
                last_cjk_idx = i
        if last_cjk_idx < 0:
            return None, None, None
        return body[: last_cjk_idx + 1], body[last_cjk_idx + 1 :], timestamp

    @staticmethod
    def _sanitize_dir_name(name: str) -> str:
        """把 product_type 当目录名前清洗，去掉路径分隔符避免越界。"""
        return re.sub(r"[\\/]+", "_", name).strip() or UNKNOWN_MODEL_DIR

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

        型号目录兜底链：文件名解析 → fallback_product_type → _unknown_model
        """
        try:
            scene, parsed_model, timestamp = self._parse_filename(original_filename)
            ext = os.path.splitext(original_filename)[1] or ".jpg"
            date_dir = received_at[:10]  # YYYY-MM-DD

            if parsed_model and timestamp:
                model_dir_name = parsed_model
                save_stem = timestamp
            elif fallback_product_type:
                model_dir_name = self._sanitize_dir_name(fallback_product_type)
                save_stem = os.path.splitext(original_filename)[0] or f"unknown_{int(time.time() * 1000)}"
            else:
                model_dir_name = UNKNOWN_MODEL_DIR
                save_stem = os.path.splitext(original_filename)[0] or f"unknown_{int(time.time() * 1000)}"

            model_dir = os.path.join(DATA_DIR, self.detector_type, date_dir, model_dir_name)
            image_dir = os.path.join(model_dir, "images")
            record_dir = os.path.join(model_dir, "records")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(record_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{save_stem}{ext}")
            record_path = os.path.join(record_dir, f"{save_stem}.json")

            if not cv2.imwrite(image_path, image):
                vision_logger.warning(f"数据回流图片写入失败 path={image_path}")
                return

            try:
                request_params = json.loads(raw_json)
            except json.JSONDecodeError:
                request_params = raw_json

            record = {
                "received_at": received_at,
                "detector_type": self.detector_type,
                "original_filename": original_filename,
                "scene_from_filename": scene,
                "model_from_filename": parsed_model,
                "saved_model_dir": model_dir_name,
                "request_params": request_params,
                "latency_ms": latency_ms,
                "result": result_dict,
            }
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            vision_logger.warning(f"数据回流落盘失败 filename={original_filename}: {e}")

    def get_router(self):
        return self.router
