'''
@Author       : gongzhang4
@Date         : 2026-01-08 05:51:33
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-17 03:10:57
@FilePath     : lap_surf.py
@Description  :
'''

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from schemas import CommonResponse
import json
from utils import vision_logger
from services import LapSurfJudgeApi
from config import settings
import cv2
import numpy as np
from services import rotate_points

lap_surf_router = APIRouter()
_judge_instance = None


async def get_judge_singleton() -> LapSurfJudgeApi:
    """获取单例判断器实例"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LapSurfJudgeApi(
            model_path=settings.lap_surf.model_path, conf_threshold=settings.lap_surf.confThreshold
        )
        vision_logger.info(
            f"初始化LapSurfJudgeApi，模型路径={settings.lap_surf.model_path}，置信度阈值={settings.lap_surf.confThreshold}"
        )
    return _judge_instance


@lap_surf_router.post(
    "/lap_surf_detect",
    summary="搭接面检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    response_model=CommonResponse,
)
async def lap_surf_detect(
    file: UploadFile = File(..., description="搭接面检测图片(jpg/png格式)"),
    json_data: str = Form(..., description="产品/物料号/模型参数的JSON字符串"),
    judge: LapSurfJudgeApi = Depends(get_judge_singleton),
):
    vision_logger.info(f"接收lap_surf_detect请求：图片={file.filename},json_data={json_data}")
    if not file.content_type.startswith("image/"):
        vision_logger.warning(f"上传的文件不是图片: {file.content_type}")
        raise HTTPException(status_code=400, detail="文件类型必须为图片(jpg/png格式)")
    try:
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        is_rotate = w < h
        if is_rotate:
            # 向左旋转90度
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")
        result_info = judge.detect(image)
        result_info = rotate_points(result_info, w, h)
        vision_logger.info(f"检测结果: {json.dumps(result_info, ensure_ascii=False, indent=2)}")
        if result_info["status"] == "true":
            result = CommonResponse(code=1, message="检测成功", result=result_info)
        else:
            result = CommonResponse(code=0, message="检测失败", result=result_info)
        vision_logger.info("参数校验通过，返回检测结果")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片处理过程中发生错误: {str(e)}")
