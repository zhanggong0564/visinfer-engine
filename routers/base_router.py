'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:25:59
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-04 01:31:46
@FilePath     : base_router.py
@Description  :路由基类，封装所有路由共有的功能
'''

from abc import ABC, abstractmethod
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi import Depends
from schemas import CommonResponse
from utils import vision_logger
from services import detection_factory
from config import settings
import json
from typing import Any
import cv2
import numpy as np
import time
import os
from services import rotate_points

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


class BaseRouter(ABC):
    """路由基类，封装所有路由共有的功能"""

    def __init__(self, router_name, api_path, summary, description, detector_type):
        self.router = APIRouter()
        self.router_name = router_name
        self.instance = None
        self.detector_type = detector_type

        self.router.post(
            api_path,
            summary=summary,
            description=description,
        )(self._handle_request)

    def get_detector_singleton(self):
        if self.instance is None:
            detector = detection_factory.get_scenarios(self.detector_type)
            if not detector:
                raise HTTPException(status_code=500, detail=f"未找到{self.detector_type}检测器")
            self.instance = detector
        return self.instance

    async def _handle_request(
        self,
        file: UploadFile = File(..., description="检测图片(jpg/png格式)"),
        json_data: str = Form(..., description="产品/物料号/模型参数的JSON字符串"),
    ):
        vision_logger.info(f"接收{self.router_name}请求：图片={file.filename}, json_data={json_data}")
        request_params = await self._validate_and_parse_params(json_data)
        vision_logger.info(f"校验参数：{request_params}")

        image, is_rotate = await self._process_image(file)
        inputs = self.get_inputs(request_params, image)
        detector = self.get_detector_singleton()
        start = time.time()
        result_info = detector.detect(inputs)
        end = time.time()
        vision_logger.info(f"检测耗时：{end - start}秒")
        vision_logger.info(f"原始检测结果：{result_info}")
        if is_rotate:
            w, h, _ = image.shape  ##注意这里是反向的
            result_info = rotate_points(result_info.to_dict(), w, h)
            vision_logger.info(f"旋转后的检测结果：{result_info}")
        vision_logger.info(f"最终检测结果：{result_info.to_dict()}")
        result = CommonResponse(
            code=1, message="检测成功", result=result_info if isinstance(result_info, dict) else result_info.to_dict()
        )  # TODO
        vision_logger.info("参数校验通过，返回检测结果")
        return result

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
            raise HTTPException(status_code=400, detail="json_data格式非法，需传入标准JSON字符串")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"参数校验失败：{str(e)}")

    def request_schema(self, json_dict) -> Any:
        """请求参数校验模式"""
        raise NotImplementedError("子类必须实现request_schema方法")

    async def _process_image(self, file: UploadFile) -> np.ndarray:
        """处理图像"""
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")
        h, w, _ = image.shape
        is_rotate = w < h
        if is_rotate:
            # 向左旋转90度
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")
        try:
            save_path, filename = file.filename.split("-")
            os.makedirs(os.path.join(DATA_DIR, save_path), exist_ok=True)
            cv2.imwrite(os.path.join(DATA_DIR, save_path, filename), image)
        except Exception as e:
            vision_logger.error(f"图片保存失败-{file.filename}-{str(e)}")
        return image, is_rotate

    def get_router(self):
        return self.router
