'''
@Author       : gongzhang4
@Date         : 2026-01-16 05:33:06
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 06:28:13
@FilePath     : indicator.py
@Description  :
'''

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from schemas import IndicatorRequest, CommonResponse
import json
from utils import vision_logger
from services import IndicatorLightBusinessAPI
from config import settings
import cv2
import numpy as np

indicator_router = APIRouter()
_detector_instance = None


async def get_detector_singleton() -> IndicatorLightBusinessAPI:
    """获取单例检测器实例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IndicatorLightBusinessAPI(
            settings.indicator_light.ModelPath,
            settings.indicator_light.ConfThreshold,
            json_path=settings.indicator_light.JSON_PATH,
            sim_thr=settings.indicator_light.SIM_THR,
        )
        vision_logger.info(
            f"初始化IndicatorDetectorAPI，模型路径={settings.indicator_light.ModelPath}，置信度阈值={settings.indicator_light.ConfThreshold}"
        )
    return _detector_instance


@indicator_router.post(
    "/indicator_light_detect",
    summary="指示灯检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    response_model=CommonResponse,
)
async def indicator_light_detect(
    file: UploadFile = File(..., description="指示灯检测图片(jpg/png格式)"),
    json_data: str = Form(..., description="产品/物料号/模型参数的JSON字符串"),
    detector: IndicatorLightBusinessAPI = Depends(get_detector_singleton),
):
    vision_logger.info(f"接收indicator_light_detect请求：图片={file.filename},json_data={json_data}")
    # 校验json_data是否为DCFuseRequest格式
    try:
        json_dict = json.loads(json_data)
        request_params = IndicatorRequest(**json_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="json_data格式非法，需传入标准JSON字符串")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数校验失败：{str(e)}")
        # 校验文件类型
    if not file.content_type.startswith("image/"):
        vision_logger.warning(f"上传的文件不是图片: {file.content_type}")
        raise HTTPException(status_code=400, detail="文件类型必须为图片(jpg/png格式)")
    main_type = request_params.type
    type = request_params.modelParams.type
    product_type = f"{main_type}-{type}"

    vision_logger.info(f"产品型号={product_type}")
    register = request_params.modelParams.register
    try:
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        if w < h:
            # 向左旋转90度
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("rotated_image.jpg", image)  # Debug: 保存旋转后的图片

        if image is None:
            vision_logger.error("图片读取失败")
            raise HTTPException(status_code=400, detail="图片读取失败，请检查文件格式")
        result_info = detector.detect(image, type_s=product_type, is_register=register)
        vision_logger.info(f"检测结果: {json.dumps(result_info, ensure_ascii=False, indent=2)}")
        if result_info["status"] == "true":
            result = CommonResponse(code=1, message="检测成功", result=result_info)
        else:
            result = CommonResponse(code=0, message="检测失败", result=result_info)
        vision_logger.info("参数校验通过，返回检测结果")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片处理过程中发生错误: {str(e)}")
