'''
@Author       : gongzhang4
@Date         : 2026-01-08 01:41:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 10:08:26
@FilePath     : business_logic.py
@Description  :
'''

from .lap_surf_core import *
from .lap_surf_det import LapSurfDetONNX
from utils import vision_logger
from ..utils.utils import vis_box_mask
import cv2
from ..api import detection_factory
from ..base import BusinessLogicBase
from ..data_base import DetectResult, DetectionItem, MoMResult


@detection_factory.register("lap_surf")
class LapSurfJudgeApi(BusinessLogicBase):

    def _initialize_model(self, settings):
        try:
            self.detector = LapSurfDetONNX(
                settings.lap_surf.model_path, nc=4, confThreshold=settings.lap_surf.confThreshold
            )
        except Exception as e:
            vision_logger.error(f"加载模型失败: {e}")
            raise e

    def business_logic_post_process(self, outputs: DetectResult, product_type: str) -> MoMResult:
        ROIs = []
        # 螺丝
        screws = []
        # 螺母
        nuts = []
        # 搭接面
        lap_joints = []
        # try:
        # try:
        for i, cls in enumerate(outputs.class_ids):
            x1, y1, x2, y2 = map(int, outputs.boxes[i][:4])
            if cls == 0:
                roi = ROI(BoundingBox(x1, y1, x2, y2, cls, outputs.scores[i], self.detector.image_src_shape))
                ROIs.append(roi)
            elif cls == 2:
                nut = BoundingBox(x1, y1, x2, y2, cls, outputs.scores[i], self.detector.image_src_shape)
                nuts.append(nut)
            elif cls == 3:
                lap_joint = LapJoint(x1, y1, x2, y2, cls, outputs.scores[i], self.detector.image_src_shape)
                lap_joints.append(lap_joint)
            else:
                screw = BoundingBox(x1, y1, x2, y2, cls, outputs.scores[i], self.detector.image_src_shape)
                screws.append(screw)
        # 对ROIs进行排序
        ROIs.sort(key=lambda x: (x.bb.x1, x.bb.y1))

        rois_info = match_all_targets(ROIs, lap_joints, nuts, screws)

        is_valid = True
        for i, roi in enumerate(rois_info):
            if not roi.is_valid:
                is_valid = False
        result = MoMResult()
        for roi in rois_info:
            roi_list = roi.to_dict()
            for item in roi_list:
                result.detailList.append(DetectionItem().from_dict(item))
        if len(result.detailList) == 0:
            is_valid = False
            result.detailList.append(DetectionItem(status=is_valid))
        result.message = "success" if is_valid else "failed"
        result.status = is_valid
        # except Exception as e:
        #     result = MoMResult(error_msg=str(e))
        return result
