'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:36:21
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 07:52:38
@FilePath     : dc_fuse.py
@Description  :
'''

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from schemas import DCFuseRequest, CommonResponse
import json
from utils import vision_logger
from services import DCFuseDetectorAPI
from config import settings
import cv2
import numpy as np

dc_router = APIRouter()
_detector_instance = None


async def get_detector_singleton() -> DCFuseDetectorAPI:
    """获取单例检测器实例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DCFuseDetectorAPI(
            model_path=settings.dc_fuse.model_path, conf_threshold=settings.dc_fuse.confThreshold
        )
        vision_logger.info(
            f"初始化DCFuseDetectorAPI，模型路径={settings.dc_fuse.model_path}，置信度阈值={settings.dc_fuse.confThreshold}"
        )
    return _detector_instance


@dc_router.post(
    "/dcfuse_detect",
    summary="直流熔丝检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    response_model=CommonResponse,
)
async def dcfuse_detect(
    file: UploadFile = File(..., description="直流熔丝检测图片(jpg/png格式)"),
    json_data: str = Form(..., description="产品/物料号/模型参数的JSON字符串"),
    detector: DCFuseDetectorAPI = Depends(get_detector_singleton),
):
    vision_logger.info(f"接收dc_fuse请求：图片={file.filename},json_data={json_data}")
    # 校验json_data是否为DCFuseRequest格式
    try:
        json_dict = json.loads(json_data)
        request_params = DCFuseRequest(**json_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="json_data格式非法，需传入标准JSON字符串")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数校验失败：{str(e)}")
        # 校验文件类型
    if not file.content_type.startswith("image/"):
        vision_logger.warning(f"上传的文件不是图片: {file.content_type}")
        raise HTTPException(status_code=400, detail="文件类型必须为图片(jpg/png格式)")
    product_model = request_params.modelParams.product_model
    vision_logger.info(f"产品型号={product_model}")
    all_products = list(detector.SUPPORTED_TYPES.keys())
    if product_model not in all_products:
        error_msg = f"产品信息 {product_model} 不支持！当前只支持: {all_products}"
        vision_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    try:
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")
        result_info = detector.detect(image, product_model)
        vision_logger.info(f"检测结果: {json.dumps(result_info, ensure_ascii=False, indent=2)}")

        vision_logger.info("检测成功")

        return CommonResponse(code=1, message="success", result=result_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片处理过程中发生错误: {str(e)}")
