'''
@Author       : gongzhang4
@Date         : 2026-01-08 01:41:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 06:34:55
@FilePath     : business_logic.py
@Description  :
'''

from .lap_surf_core import *
from .yolo import yolo11ONNX
from utils import vision_logger
from ..utils import vis_box_mask
import cv2


class LapSurfJudgeApi:
    def __init__(self, model_path, conf_threshold=0.5):
        self.detector = self.load_model(model_path, conf_threshold)

    def load_model(self, model_path, conf_threshold):
        return yolo11ONNX(model_path, nc=4, confThreshold=conf_threshold)

    def detect(self, im):
        outputs = self.detector.infer(im)
        # vis = vis_box_mask(im, outputs)
        # cv2.imwrite("vis.jpg", vis)
        results = self.postprocess(outputs)
        return results

    def postprocess(self, outputs):
        ROIs = []
        # 螺丝
        screws = []
        # 螺母
        nuts = []
        # 搭接面
        lap_joints = []
        # try:
        for i, cls in enumerate(outputs["cls"]):
            x1, y1, x2, y2 = map(int, outputs["rect"][i][:4])
            if cls == 0:
                roi = ROI(BoundingBox(x1, y1, x2, y2, cls, outputs["score"][i], self.detector.image_src_shape))
                ROIs.append(roi)
            elif cls == 2:
                nut = BoundingBox(x1, y1, x2, y2, cls, outputs["score"][i], self.detector.image_src_shape)
                nuts.append(nut)
            elif cls == 3:
                lap_joint = LapJoint(x1, y1, x2, y2, cls, outputs["score"][i], self.detector.image_src_shape)
                lap_joints.append(lap_joint)
            else:
                screw = BoundingBox(x1, y1, x2, y2, cls, outputs["score"][i], self.detector.image_src_shape)
                screws.append(screw)
        # 对ROIs进行排序
        ROIs.sort(key=lambda x: (x.bb.x1, x.bb.y1))

        rois_info = match_all_targets(ROIs, lap_joints, nuts, screws)

        is_valid = True
        for i, roi in enumerate(rois_info):
            if not roi.is_valid:
                is_valid = False
        detailList = []
        for roi in rois_info:
            roi_list = roi.to_dict()
            detailList.extend(roi_list)
        if len(detailList) == 0:
            is_valid = False
            detailList.append({"status": "false", "scene": "", "coordinate": [], "accuracy": 0.0})
        dict_info = {
            "detailList": detailList,
            'status': "true" if is_valid else "false",
            "message": "success" if is_valid else "failed",
            "error_msg": "",
        }
        return dict_info
