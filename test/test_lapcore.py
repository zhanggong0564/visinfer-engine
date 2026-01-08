'''
@Author       : gongzhang4
@Date         : 2026-01-08 05:33:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 05:38:07
@FilePath     : test_lapcore.py
@Description  :
'''

'''
搭接面服务业务逻辑单元测试 - 使用pytest框架
@Author       : AI Assistant
@Date         : 2026-01-08
@Description  : 搭接面服务核心业务逻辑单元测试
'''

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.lap_surf.lap_surf_core import BoundingBox, LapJoint, ROI, match_all_targets
from services.lap_surf.business_logic import LapSurfJudgeApi


class TestBoundingBox:
    """BoundingBox类单元测试"""

    def test_bounding_box_initialization(self):
        """测试BoundingBox初始化"""
        image_shape = (480, 640, 3)
        bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)

        assert bb.x1 == 10
        assert bb.y1 == 20
        assert bb.x2 == 100
        assert bb.y2 == 200
        assert bb.label == 0
        assert bb.conf == 0.95
        assert bb.is_match is False
        assert bb.h == 480
        assert bb.w == 640

    def test_bounding_box_center_calculation(self):
        """测试中心点计算"""
        image_shape = (480, 640, 3)
        bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)

        center_x, center_y = bb.center
        assert center_x == 55.0  # (10 + 100) / 2
        assert center_y == 110.0  # (20 + 200) / 2

    def test_bounding_box_edges_property(self):
        """测试edges属性"""
        image_shape = (480, 640, 3)
        bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)

        assert bb.edges == 100  # x2坐标


class TestLapJoint:
    """LapJoint类单元测试"""
    

    def test_lap_joint_initialization(self):
        """测试LapJoint初始化"""
        image_shape = (480, 640, 3)
        lap_joint = LapJoint(50, 60, 150, 250, 3, 0.92, image_shape)

        assert lap_joint.x1 == 50
        assert lap_joint.y1 == 60
        assert lap_joint.x2 == 150
        assert lap_joint.y2 == 250
        assert lap_joint.label == 3
        assert lap_joint.conf == 0.92
        assert len(lap_joint.nuts) == 0

    def test_lap_joint_contains_center(self):
        """测试包含中心点判断"""
        image_shape = (480, 640, 3)
        lap_joint = LapJoint(50, 60, 150, 250, 3, 0.92, image_shape)

        # 测试在内部的点
        target_inside = BoundingBox(100, 150, 120, 180, 2, 0.85, image_shape)
        assert lap_joint.contains_center(target_inside) is True

        # 测试在外部的点
        target_outside = BoundingBox(10, 20, 30, 40, 2, 0.85, image_shape)
        assert lap_joint.contains_center(target_outside) is False


# class TestROI:
#     """ROI类单元测试"""

#     def test_roi_initialization(self):
#         """测试ROI初始化"""
#         image_shape = (480, 640, 3)
#         bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)
#         roi = ROI(bb)

#         assert roi.bb == bb
#         assert roi.lap_joint is None
#         assert len(roi.nuts) == 0
#         assert len(roi.screws) == 0
#         assert roi.label2name[0] == "roi"
#         assert roi.label2name[1] == "螺丝"
#         assert roi.label2name[2] == "螺母"
#         assert roi.label2name[3] == "搭接面"

#     def test_roi_contains_center(self):
#         """测试ROI包含中心点判断"""
#         image_shape = (480, 640, 3)
#         bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)
#         roi = ROI(bb)

#         # 测试在内部的点
#         target_inside = BoundingBox(50, 100, 70, 120, 2, 0.85, image_shape)
#         assert roi.contains_center(target_inside) is True

#         # 测试在外部的点
#         target_outside = BoundingBox(200, 300, 220, 320, 2, 0.85, image_shape)
#         assert roi.contains_center(target_outside) is False

#     def test_roi_is_valid_property(self):
#         """测试ROI有效性判断"""
#         image_shape = (480, 640, 3)
#         bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)
#         roi = ROI(bb)

#         # 初始状态应该无效
#         assert roi.is_valid is False

#         # 添加搭接面但螺母数量不足
#         lap_joint = LapJoint(50, 60, 80, 120, 3, 0.92, image_shape)
#         roi.lap_joint = lap_joint
#         assert roi.is_valid is False

#         # 添加2个螺母到搭接面
#         for i in range(2):
#             nut = BoundingBox(60+i*10, 70, 70+i*10, 80, 2, 0.85, image_shape)
#             lap_joint.nuts.append(nut)

#         # 添加4个螺母到ROI
#         for i in range(4):
#             nut = BoundingBox(20+i*15, 30, 30+i*15, 40, 2, 0.85, image_shape)
#             roi.nuts.append(nut)

#         # 现在应该有效
#         assert roi.is_valid is True

#     def test_roi_to_dict(self):
#         """测试ROI转换为字典"""
#         image_shape = (480, 640, 3)
#         bb = BoundingBox(10, 20, 100, 200, 0, 0.95, image_shape)
#         roi = ROI(bb)

#         # 测试空ROI转换
#         result = roi.to_dict()
#         assert len(result) == 1  # 只有ROI本身
#         assert result[0]["scene"] == "roi"
#         assert result[0]["status"] is False

#         # 添加完整信息后测试
#         lap_joint = LapJoint(50, 60, 80, 120, 3, 0.92, image_shape)
#         roi.lap_joint = lap_joint

#         # 添加2个螺母到搭接面
#         for i in range(2):
#             nut = BoundingBox(60+i*10, 70, 70+i*10, 80, 2, 0.85, image_shape)
#             lap_joint.nuts.append(nut)

#         # 添加4个螺母到ROI
#         for i in range(4):
#             nut = BoundingBox(20+i*15, 30, 30+i*15, 40, 2, 0.85, image_shape)
#             roi.nuts.append(nut)

#         # 添加2个螺丝到ROI
#         for i in range(2):
#             screw = BoundingBox(40+i*20, 50, 50+i*20, 60, 1, 0.88, image_shape)
#             roi.screws.append(screw)

#         result = roi.to_dict()
#         # ROI + LapJoint + 2个搭接面螺母 + 2个螺丝 + 4个ROI螺母 = 9个条目
#         assert len(result) == 9
#         assert result[0]["scene"] == "roi"
#         assert result[1]["scene"] == "搭接面"
#         assert result[2]["scene"] == "螺母"  # 搭接面的第一个螺母
#         assert result[3]["scene"] == "螺母"  # 搭接面的第二个螺母
#         assert result[4]["scene"] == "螺丝"  # 第一个螺丝
#         assert result[5]["scene"] == "螺丝"  # 第二个螺丝
#         assert result[6]["scene"] == "螺母"  # ROI的第一个螺母
#         assert result[7]["scene"] == "螺母"  # ROI的第二个螺母
#         assert result[8]["scene"] == "螺母"  # ROI的第三个螺母


# class TestMatchAllTargets:
#     """匹配算法单元测试"""

#     def test_match_all_targets_basic(self):
#         """测试基本匹配功能"""
#         image_shape = (480, 640, 3)

#         # 创建测试数据
#         rois = [
#             ROI(BoundingBox(0, 0, 200, 200, 0, 0.95, image_shape)),
#             ROI(BoundingBox(200, 0, 400, 200, 0, 0.95, image_shape))
#         ]

#         lap_joints = [
#             LapJoint(50, 50, 150, 150, 3, 0.92, image_shape),
#             LapJoint(250, 50, 350, 150, 3, 0.92, image_shape)
#         ]

#         nuts = [
#             BoundingBox(60, 60, 80, 80, 2, 0.85, image_shape),  # 在第一个搭接面内
#             BoundingBox(70, 70, 90, 90, 2, 0.85, image_shape),  # 在第一个搭接面内
#             BoundingBox(260, 60, 280, 80, 2, 0.85, image_shape),  # 在第二个搭接面内
#             BoundingBox(270, 70, 290, 90, 2, 0.85, image_shape),  # 在第二个搭接面内
#         ]

#         screws = [
#             BoundingBox(30, 30, 50, 50, 1, 0.88, image_shape),
#             BoundingBox(230, 30, 250, 50, 1, 0.88, image_shape)
#         ]

#         # 执行匹配
#         result = match_all_targets(rois, lap_joints, nuts, screws)

#         # 验证匹配结果
#         assert len(result) == 2

#         # 检查第一个ROI
#         roi1 = result[0]
#         assert roi1.lap_joint is not None
#         assert len(roi1.lap_joint.nuts) == 2  # 搭接面应该包含2个螺母
#         assert len(roi1.nuts) == 2  # ROI应该包含2个螺母（搭接面的螺母）
#         assert len(roi1.screws) == 1  # ROI应该包含1个螺丝

#         # 检查第二个ROI
#         roi2 = result[1]
#         assert roi2.lap_joint is not None
#         assert len(roi2.lap_joint.nuts) == 2
#         assert len(roi2.nuts) == 2
#         assert len(roi2.screws) == 1

#     def test_match_all_targets_cross_roi(self):
#         """测试跨ROI匹配场景"""
#         image_shape = (480, 640, 3)

#         # 创建重叠的ROI
#         rois = [
#             ROI(BoundingBox(0, 0, 300, 300, 0, 0.95, image_shape)),
#             ROI(BoundingBox(200, 0, 400, 300, 0, 0.95, image_shape))
#         ]

#         # 搭接面在重叠区域
#         lap_joints = [
#             LapJoint(250, 100, 350, 200, 3, 0.92, image_shape)
#         ]

#         # 螺母在重叠区域
#         nuts = [
#             BoundingBox(280, 120, 300, 140, 2, 0.85, image_shape)
#         ]

#         screws = [
#             BoundingBox(280, 80, 300, 100, 1, 0.88, image_shape)
#         ]

#         # 执行匹配
#         result = match_all_targets(rois, lap_joints, nuts, screws)

#         # 验证螺母被正确分配到最近的ROI
#         assert len(result) == 2

#         # 找到包含搭接面的ROI
#         roi_with_lap = None
#         for roi in result:
#             if roi.lap_joint is not None:
#                 roi_with_lap = roi
#                 break

#         assert roi_with_lap is not None
#         # 螺母应该被分配到包含搭接面的ROI
#         assert len(roi_with_lap.nuts) == 1


# class TestLapSurfJudgeApi:
#     """LapSurfJudgeApi类单元测试"""

#     @patch('services.lap_surf.business_logic.yolo11ONNX')
#     def test_api_initialization(self, mock_yolo):
#         """测试API初始化"""
#         mock_model = MagicMock()
#         mock_yolo.return_value = mock_model

#         api = LapSurfJudgeApi("dummy_model_path.onnx", conf_threshold=0.5)

#         mock_yolo.assert_called_once_with("dummy_model_path.onnx", nc=4, confThreshold=0.5)
#         assert api.detector == mock_model

#     @patch('services.lap_surf.business_logic.yolo11ONNX')
#     @patch('services.lap_surf.business_logic.vis_box_mask')
#     @patch('cv2.imwrite')
#     def test_api_detect(self, mock_imwrite, mock_vis_box_mask, mock_yolo):
#         """测试检测功能"""
#         # 模拟模型输出
#         mock_model = MagicMock()
#         mock_model.infer.return_value = {
#             "cls": [0, 2, 3, 1],
#             "rect": [[10, 20, 100, 200], [60, 70, 80, 90], [50, 60, 150, 250], [30, 40, 50, 60]],
#             "score": [0.95, 0.85, 0.92, 0.88]
#         }
#         mock_model.image_src_shape = (480, 640, 3)
#         mock_yolo.return_value = mock_model

#         mock_vis_box_mask.return_value = "dummy_vis_image"

#         api = LapSurfJudgeApi("dummy_model_path.onnx")
#         api.detector = mock_model

#         # 模拟输入图像
#         mock_image = MagicMock()
#         result = api.detect(mock_image)

#         # 验证调用
#         mock_model.infer.assert_called_once_with(mock_image)
#         mock_vis_box_mask.assert_called_once()
#         mock_imwrite.assert_called_once_with("vis.jpg", "dummy_vis_image")

#         # 验证结果格式
#         assert len(result) == 1
#         assert result[0]["code"] == 1
#         assert result[0]["message"] == "success"
#         assert "result" in result[0]
#         assert "detailList" in result[0]["result"]
#         assert "status" in result[0]["result"]


# def test_integration_scenario():
#     """集成测试场景"""
#     image_shape = (480, 640, 3)

#     # 创建完整的测试场景
#     rois = [ROI(BoundingBox(0, 0, 320, 240, 0, 0.95, image_shape))]
#     lap_joints = [LapJoint(100, 80, 220, 180, 3, 0.92, image_shape)]

#     # 创建4个螺母（2个在搭接面内，2个在ROI内）
#     nuts = [
#         BoundingBox(120, 100, 140, 120, 2, 0.85, image_shape),  # 在搭接面内
#         BoundingBox(180, 120, 200, 140, 2, 0.85, image_shape),  # 在搭接面内
#         BoundingBox(50, 50, 70, 70, 2, 0.85, image_shape),     # 在ROI内
#         BoundingBox(250, 50, 270, 70, 2, 0.85, image_shape),   # 在ROI内
#     ]

#     screws = [
#         BoundingBox(40, 40, 60, 60, 1, 0.88, image_shape),
#         BoundingBox(260, 40, 280, 60, 1, 0.88, image_shape)
#     ]

#     # 执行匹配
#     result = match_all_targets(rois, lap_joints, nuts, screws)

#     # 验证集成结果
#     assert len(result) == 1
#     roi = result[0]

#     # 验证有效性条件
#     assert roi.lap_joint is not None
#     assert len(roi.lap_joint.nuts) == 2  # 搭接面包含2个螺母
#     assert len(roi.nuts) == 4  # ROI总共包含4个螺母
#     assert roi.is_valid is True  # 应该满足有效性条件


if __name__ == "__main__":
    # 运行pytest测试
    pytest.main([__file__, "-v"])
