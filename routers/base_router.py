'''
@Author       : gongzhang4
@Date         : 2026-01-19 08:25:59
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-19 09:13:44
@FilePath     : base_router.py
@Description  :路由基类，封装所有路由共有的功能
'''

from abc import ABC, abstractmethod
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi import Depends
from schemas import CommonResponse
from utils import vision_logger
from services import detector_factory
from config import settings
import json
from typing import Any
import cv2
import numpy as np


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
        detector=Depends(get_detector_singleton),
    ):
        vision_logger.info(f"接收{self.router_name}请求：图片={file.filename}, json_data={json_data}")
        request_params = await self._validate_and_parse_params(json_data)
        image = await self._process_image(file)

        result_info = await self._process_business_logic(image, request_params, detector)

        result = CommonResponse(code=1, message="检测成功", result=result_info)
        vision_logger.info("参数校验通过，返回检测结果")
        return result

    async def _validate_and_parse_params(self, json_data: str) -> Any:
        """验证和解析参数"""
        try:
            json_dict = json.loads(json_data)
            request_params = self.request_schema(**json_dict)
            return request_params
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="json_data格式非法，需传入标准JSON字符串")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"参数校验失败：{str(e)}")

    async def _process_image(self, file: UploadFile) -> np.ndarray:
        """处理图像"""
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")

        return image

    @abstractmethod
    async def _process_business_logic(self, image: np.ndarray, request_params: Any, detector: Any) -> dict:
        """处理业务逻辑，子类必须实现"""
        pass
