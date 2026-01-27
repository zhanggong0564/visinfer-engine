'''
@Author       : gongzhang4
@Date         : 2026-01-23 06:32:10
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 05:25:40
@FilePath     : business_logic_v2.py
@Description  :
'''

'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:11:32
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-17 05:45:57
@FilePath     : business_logic.py
@Description  :
'''

from .dc_fuse_detect import DCFuseDetector
import numpy as np
from collections import defaultdict
from utils import vision_logger
from ..base.business_logic_base import BusinessLogicBase
from ..data_base import DetectResult, MoMResult, DetectionItem
from ..api import detection_factory


class ResultJudge:
    def __init__(
        self,
        ways=5,
        is_detect_metal_piece=True,
        is_detect_upper_screw=False,
        is_detect_nut=False,
        is_detectscrew=True,
        is_small_screw=False,
        metal_piece_num=4,
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
            if self.metal_piece_num2 == 2:
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


@detection_factory.register("dc_fuse")
class DCFuseDetectorAPI(BusinessLogicBase):
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

    def __init__(self, settings):
        super().__init__(settings)

        self.label_mapping = {
            "screw": ["screw_1", "no_screw_1"],
            "nut": ["nut_2"],
            "small_screw": ["small_screw_8", "no_small_screw_8"],
            "brass_plate": ["brass_plate_6"],
            "metal_piece": ["metal_piece_4"],
            "upper_screw": ["upper_crossbeam_screw_9", "no_upper_crossbeam_screw_9"],
            "lower_screw": ["lower_crossbeam_screw_10", "no_lower_crossbeam_screw_10"],
        }

    def _initialize_model(self, settings):
        try:
            """初始化模型"""
            self.detector = DCFuseDetector(settings.dc_fuse.model_path, settings.dc_fuse.confThreshold)
        except Exception as e:
            vision_logger.error(f"初始化模型失败: {e}")

    def business_logic_post_process(self, result: DetectResult, product_type: str) -> MoMResult:
        """业务逻辑后处理"""
        if product_type not in self.SUPPORTED_TYPES:
            return MoMResult(status=False, error_msg=f"不支持的检测类型: {product_type}")
        result_judge = self.SUPPORTED_TYPES[product_type]
        det_info = defaultdict(list)
        for label_id, bbox, score, label in zip(result.class_ids, result.boxes, result.scores, result.class_names):
            det_info[label].append({"bbox": bbox, "score": score})
        judge_result = result_judge(det_info)
        mom_result = MoMResult(status=True)
        for label, is_pass in judge_result.items():
            if not is_pass:
                mom_result.status = False
            labels = self.label_mapping.get(label, [])
            for label in labels:
                det_infos = det_info.get(label, [])
                for det in det_infos:
                    mom_result.detailList.append(
                        DetectionItem(status=is_pass, scene=label, coordinate=det["bbox"], accuracy=det["score"])
                    )
        mom_result.message = "检测成功"
        return mom_result

    def result_post_process(self, result: MoMResult, w, h) -> MoMResult:
        """结果后处理"""
        detailList = result.detailList
        for item in detailList:
            coordinate = item.coordinate
            ltx, lty, rbx, rby = coordinate
            x1, y1 = ltx, lty
            x2, y2 = rbx, lty
            x3, y3 = rbx, rby
            x4, y4 = ltx, rby
            item.coordinate = [x1 / w, y1 / h, x2 / w, y2 / h, x3 / w, y3 / h, x4 / w, y4 / h]
        return result
