'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:11:32
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 07:19:48
@FilePath     : business_logic.py
@Description  :
'''

from .yolo import DCFuseDetector
import numpy as np
from collections import defaultdict
from utils import vision_logger
from ..utils import vis_box_mask
import cv2


class ResultJudge:
    def __init__(
        self,
        ways=5,
        is_detect_metal_piece=True,
        is_detect_upper_screw=False,
        is_detect_nut=False,
        is_detectscrew=True,
        is_small_screw=False,
        metal_piece_num=2,
    ):
        self.ways = ways
        self.is_detect_metal_piece = is_detect_metal_piece
        self.is_detect_upper_screw = is_detect_upper_screw
        self.is_detect_nut = is_detect_nut
        self.is_detectscrew = is_detectscrew
        self.is_small_screw = is_small_screw
        self.metal_piece_num2 = metal_piece_num
        self.it2classes = {
            "screw_1": "screw",
            "nut_2": "nut",
            "brass_plate_6": "brass_plate",
            "metal_piece_4": "metal_piece",
            "no_screw_1": "screw",
            "upper_crossbeam_screw_9": "upper_screw",
            "lower_crossbeam_screw_10": "lower_screw",
            "no_upper_crossbeam_screw_9": "upper_screw",
            "no_lower_crossbeam_screw_10": "lower_screw",
            "small_screw_8": "small_screw",
        }

    def __call__(self, det_info):
        screw = det_info.get("screw_1", [])
        nut = det_info.get("nut_2", [])
        brass_plate = det_info.get("brass_plate_6", [])
        metal_piece = det_info.get("metal_piece_4", [])
        no_screw = det_info.get("no_screw_1", [])
        upper_screw = det_info.get("upper_crossbeam_screw_9", [])
        lower_screw = det_info.get("lower_crossbeam_screw_10", [])
        no_upper_screw = det_info.get("no_upper_crossbeam_screw_9", [])
        no_lower_screw = det_info.get("no_lower_crossbeam_screw_10", [])
        small_screw = det_info.get("small_screw_8", [])
        res = {
            "screw": True,
            "nut": True,
            "metal_piece": True,
            "upper_screw": True,
            "lower_screw": True,
            "brass_plate": True,
            "small_screw": True,
        }
        if self.is_detectscrew:
            if (len(screw) != self.ways * 2) and (len(no_screw) > 0):
                res["screw"] = False
        if self.is_small_screw:
            if len(small_screw) != self.ways:
                res["small_screw"] = False
        if self.is_detect_nut:
            if len(nut) != self.ways * 2:
                res["nut"] = False
        if self.is_detect_metal_piece:
            if self.metal_piece_num2:
                if len(metal_piece) != 2:
                    res["metal_piece"] = False
            else:
                if not (len(metal_piece) == 4 or len(metal_piece) == 6):
                    res["metal_piece"] = False
        if self.is_detect_upper_screw:
            if (len(upper_screw) != 2 or len(lower_screw) != 2) and (
                len(no_upper_screw) > 0 or len(no_lower_screw) > 0
            ):
                res["upper_screw"] = False
        if len(brass_plate) != self.ways:
            res["brass_plate"] = False
        return {k: v for k, v in res.items() if self._is_detection_enabled(k)}

    def _is_detection_enabled(self, key: str) -> bool:
        """检查指定检测项是否启用"""
        detection_map = {
            "screw": self.is_detectscrew,
            "nut": self.is_detect_nut,
            "metal_piece": self.is_detect_metal_piece,
            "upper_screw": self.is_detect_upper_screw,
            "lower_screw": self.is_detect_upper_screw,
            "brass_plate": True,
            "small_screw": self.is_small_screw,
        }
        return detection_map.get(key, False)


class DCFuseDetectorAPI:
    """直流熔丝盒检测对外接口类"""

    # 支持的检测类型配置
    SUPPORTED_TYPES = {
        "五路有熔丝盒有磁环": ResultJudge(
            ways=5, is_detectscrew=True, is_small_screw=True, is_detect_metal_piece=True, is_detect_upper_screw=True
        ),
        "五路有熔丝盒无磁环": ResultJudge(ways=5, is_detectscrew=True, is_detect_nut=True, is_detect_metal_piece=True),
        "六路无熔丝盒无磁环": ResultJudge(ways=6, is_detectscrew=False, is_detect_metal_piece=True, is_detect_nut=True),
        "六路有熔丝盒无磁环": ResultJudge(ways=6, is_detectscrew=True, is_detect_nut=True, is_detect_metal_piece=True),
        "七路无熔丝盒无磁环": ResultJudge(ways=7, is_detectscrew=False, is_detect_nut=True),
        "七路有熔丝盒无磁环": ResultJudge(
            ways=7, is_detect_metal_piece=True, is_detectscrew=True, is_detect_nut=True, metal_piece_num=2
        ),
    }

    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        self.detector = self.load_model(model_path, conf_threshold)
        self.label_mapping = {
            "screw": ["screw_1", "no_screw_1"],
            "nut": ["nut_2"],
            "small_screw": ["small_screw_8", "no_small_screw_8"],
            "brass_plate": ["brass_plate_6"],
            "metal_piece": ["metal_piece_4"],
            "upper_screw": ["upper_crossbeam_screw_9", "no_upper_crossbeam_screw_9"],
            "lower_screw": ["lower_crossbeam_screw_10", "no_lower_crossbeam_screw_10"],
        }

    def load_model(self, model_path: str, conf_threshold: float = 0.4):
        """加载模型"""
        return DCFuseDetector(model_path, conf_threshold)

    def detect(self, image: np.ndarray, detect_type: str) -> dict:
        """检测图像中的直流熔丝盒"""
        if detect_type not in self.SUPPORTED_TYPES:
            return {"details": [], "status": False, "error_msg": f"不支持的检测类型: {detect_type}"}
        try:
            infer_result = self.detector.infer(image)
            # image_vis = vis_box_mask(image.copy(), infer_result)
            # cv2.imwrite("vis.jpg", image_vis)

        except Exception as e:
            return {"details": [], "status": False, "error_msg": str(e), "message": "检测失败"}
        det_info = defaultdict(list)
        for label, bbox, score in zip(infer_result["cls_name"], infer_result["rect"], infer_result["score"]):
            # 归一化bbox
            ltx, lty, rbx, rby = bbox
            # 转成矩形框的四个顶点
            w, h = rbx - ltx, rby - lty
            x1, y1 = ltx, lty
            x2, y2 = ltx + w, lty
            x3, y3 = rbx, rby
            x4, y4 = ltx, lty + h
            points_normalized = [
                x1 / image.shape[1],
                y1 / image.shape[0],
                x2 / image.shape[1],
                y2 / image.shape[0],
                x3 / image.shape[1],
                y3 / image.shape[0],
                x4 / image.shape[1],
                y4 / image.shape[0],
            ]

            # bbox_normalized = [ltx / image.shape[1], lty / image.shape[0], rbx / image.shape[1], rby / image.shape[0]]
            det_info[label].append({"bbox": points_normalized, "score": score})

        # 执行结果判断
        result_judge = self.SUPPORTED_TYPES[detect_type]
        judge_result = result_judge(det_info)

        # 生成详细结果
        details = []
        status = True
        for cls, is_passed in judge_result.items():
            if not is_passed:
                status = False
            # 找到对应的原始标签和检测结果
            labels = self.label_mapping.get(cls, [])
            for label in labels:
                det_infos = det_info.get(label, [])
                for det in det_infos:
                    details.append(
                        {"status": is_passed, "scene": cls, "coordinate": det["bbox"], "accuracy": det["score"]}
                    )

        # 整体结果
        return {"detailList": details, "status": "true" if status else "false", "error_msg": "", "message": "检测成功"}
