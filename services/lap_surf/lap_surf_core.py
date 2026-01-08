'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:42:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 06:42:50
@FilePath     : lap_surf_core.py
@Description  :  lap surf 业务逻辑
'''

from typing import List
from utils import vision_logger


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, label, conf, image_src_shape):
        self.x1 = x1  # 左上角x
        self.y1 = y1  # 左上角y
        self.x2 = x2  # 右下角x
        self.y2 = y2  # 右下角y
        self.label = label  # "lap_joint"/"nut"/"screw"
        self.is_match = False  # 是否匹配
        self.conf = conf  # 置信度
        self.h, self.w, _ = image_src_shape

    @property
    def center(self):
        """计算目标中心坐标"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def edges(self):
        return self.x2


class LapJoint(BoundingBox):
    def __init__(self, x1, y1, x2, y2, label, conf, image_src_shape):
        super().__init__(x1, y1, x2, y2, label, conf, image_src_shape)
        self.__nuts = []  # 包含的螺母

    @property
    def nuts(self):
        """返回包含的螺母"""
        return self.__nuts

    def contains_center(self, target: BoundingBox):
        """判断目标中心是否在搭接面内"""
        cx, cy = target.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


class ROI:
    def __init__(self, bb: BoundingBox):
        self.bb = bb
        self.lap_joint = None
        self.nuts = []  # 包含的螺母
        self.screws = []  # 包含的螺丝
        self.label2name = {
            0: "roi",
            1: "螺丝",
            2: "螺母",
            3: "搭接面",
        }

    def contains_center(self, target: BoundingBox):
        """判断目标中心是否在ROI内"""
        cx, cy = target.center
        return self.bb.x1 <= cx <= self.bb.x2 and self.bb.y1 <= cy <= self.bb.y2

    @property
    def is_valid(self):
        """判断ROI是否有效. 有效条件: 包含 laparoint,laparoint的螺母数量为2,ROI内螺丝数量为2,螺母数量为4"""
        return (
            self.lap_joint is not None
            and len(self.lap_joint.nuts) == 2
            # and len(self.screws) == 2
            and len(self.nuts) == 4
        )

    def to_dict(self):
        result = []
        if self.bb:
            result.append(
                {
                    "coordinate": [
                        self.bb.x1 / self.bb.w,
                        self.bb.y1 / self.bb.h,
                        self.bb.x2 / self.bb.w,
                        self.bb.y2 / self.bb.h,
                    ],
                    "status": self.is_valid,
                    "scene": self.label2name[self.bb.label],
                    "accuracy": self.bb.conf,
                }
            )
        else:
            result.append(
                {
                    "coordinate": [],
                    "status": False,
                    "scene": "",
                    "accuracy": 0,
                }
            )
        if self.lap_joint:
            result.append(
                {
                    "coordinate": [
                        self.lap_joint.x1 / self.lap_joint.w,
                        self.lap_joint.y1 / self.lap_joint.h,
                        self.lap_joint.x2 / self.lap_joint.w,
                        self.lap_joint.y2 / self.lap_joint.h,
                    ],
                    "status": self.lap_joint.is_match,
                    "scene": self.label2name[self.lap_joint.label],
                    "accuracy": self.lap_joint.conf,
                }
            )
        else:
            result.append(
                {
                    "coordinate": [],
                    "status": False,
                    "scene": "",
                    "accuracy": 0,
                }
            )
        for i, nut in enumerate(self.lap_joint.nuts):
            result.append(
                {
                    "coordinate": [
                        nut.x1 / self.bb.w,
                        nut.y1 / self.bb.h,
                        nut.x2 / self.bb.w,
                        nut.y2 / self.bb.h,
                    ],
                    "status": nut.is_match,
                    "scene": self.label2name[nut.label],
                    "accuracy": nut.conf,
                }
            )
        for _, screw in enumerate(self.screws):
            result.append(
                {
                    "coordinate": [
                        screw.x1 / self.bb.w,
                        screw.y1 / self.bb.h,
                        screw.x2 / self.bb.w,
                        screw.y2 / self.bb.h,
                    ],
                    "status": True,
                    "scene": self.label2name[screw.label],
                    "accuracy": screw.conf,
                }
            )
        for _, nut in enumerate(self.nuts):
            result.append(
                {
                    "coordinate": [
                        nut.x1 / self.bb.w,
                        nut.y1 / self.bb.h,
                        nut.x2 / self.bb.w,
                        nut.y2 / self.bb.h,
                    ],
                    "status": True,
                    "scene": self.label2name[nut.label],
                    "accuracy": nut.conf,
                }
            )
        return result


def match_all_targets(
    rois: List[ROI],
    lap_joints: List[LapJoint],
    nuts: List[BoundingBox],
    screws: List[BoundingBox],
):
    """
    匹配所有目标到ROI：
    1. 搭接面：直接绑定到唯一包含其中心的ROI（无重叠）
    2. 螺母/螺丝：优先按中心归属，跨ROI时按与搭接面的距离判断
    """
    # 步骤1：匹配搭接面（无重叠，直接绑定）
    for lap in lap_joints:
        # 找到中心落在lap的的螺母
        lap_nuts = []
        for nut in nuts:
            if lap.contains_center(nut):
                nut.is_match = True
                lap_nuts.append(nut)
        # 绑定螺母到搭接面,
        for nut in lap_nuts:
            lap.nuts.append(nut)

        # 找到包含搭接面中心的ROI（唯一）
        for roi in rois:
            if roi.contains_center(lap):
                roi.lap_joint = lap
                lap.is_match = True
                break
    for nut in nuts:
        if not nut.is_match:
            # 找到包含螺母中心的所有ROI
            candidate_rois = [roi for roi in rois if roi.contains_center(nut)]

            if len(candidate_rois) == 1:
                # 唯一归属的ROI
                candidate_rois[0].nuts.append(nut)
            else:
                # 跨ROI时，选距离搭接面中心最近的ROI
                min_dist = float("inf")
                best_roi = None
                for roi in candidate_rois:
                    # 计算螺母到ROI内搭接面的距离（搭接面必存在）
                    nx, ny = nut.center
                    lx, ly = roi.lap_joint.center
                    dist = ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_roi = roi
                if best_roi is not None:
                    best_roi.nuts.append(nut)
                vision_logger.warning("螺母 可能超出 ROI 范围")
    for screw in screws:
        if not screw.is_match:
            # 找到包含螺丝中心的所有ROI
            candidate_rois = [roi for roi in rois if roi.contains_center(screw)]

            if len(candidate_rois) == 1:
                candidate_rois[0].screws.append(screw)
            else:
                # 跨ROI时，选距离搭接面中心最近的ROI
                min_dist = float("inf")
                best_roi = None
                for roi in candidate_rois:
                    # 计算螺丝到ROI内搭接面的距离（搭接面必存在）
                    sx, _ = screw.center
                    edges = roi.lap_joint.edges
                    dist = abs(sx - edges)
                    if dist < min_dist:
                        min_dist = dist
                        best_roi = roi
                best_roi.screws.append(screw)
                screw.is_match = True
    return rois
