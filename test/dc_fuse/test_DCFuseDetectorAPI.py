'''
@Author       : gongzhang4
@Date         : 2026-01-23 08:00:22
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-23 08:46:32
@FilePath     : test_DCFuseDetectorAPI.py
@Description  :
'''

import unittest
import json
import sys
from pathlib import Path


sys.path.insert(0, Path(__file__).resolve().parent.parent.parent)
print(Path(__file__).resolve().parent.parent)

from services import DCFuseDetectorAPI
from services.data_base import DetectResult, MoMResult, DetectionItem


class TestDCFuseDetectorAPI(unittest.TestCase):
    def setUp(self):
        self.DCFuseDet = DCFuseDetectorAPI()

    def test_5L_Fuse_MagR(self):
        dect_reult = DetectResult()
        # screw,small_screw,metal_piece,brass_plate,upper,lower
        dect_reult.boxes = (
            [[1, 2, 3, 4] for _ in range(10)]  # 10个螺丝
            + [[5, 6, 7, 8] for _ in range(5)]  # 5个小螺丝
            + [[9, 10, 11, 12] for _ in range(4)]  # 4个金属片
            + [[13, 14, 15, 16] for _ in range(5)]  # 5个铜片
            + [[17, 18, 19, 20] for _ in range(2)]  # 2个上横梁螺丝
            + [[21, 22, 23, 24] for _ in range(2)]  # 2个下横梁螺丝
        )
        dect_reult.scores = [0.9] * 10 + [0.8] * 5 + [0.7] * 4 + [0.6] * 5 + [0.5] * 2 + [0.4] * 2
        dect_reult.class_ids = [0] * 10 + [1] * 5 + [2] * 4 + [3] * 5 + [4] * 2 + [5] * 2
        dect_reult.class_names = (
            ["screw_1" for _ in range(10)]
            + ["small_screw_8" for _ in range(5)]
            + ["metal_piece_4" for _ in range(4)]
            + ["brass_plate_6" for _ in range(5)]
            + ["upper_crossbeam_screw_9" for _ in range(2)]
            + ["lower_crossbeam_screw_10" for _ in range(2)]
        )
        mom_result = self.DCFuseDet.business_logic_post_process(dect_reult, "五路有熔丝盒有磁环")
        print(mom_result)
        self.assertEqual(mom_result.message, "检测成功")
        self.assertEqual(mom_result.error_msg, "")
        self.assertEqual(mom_result.status, True)
        for item in mom_result.detailList:
            self.assertEqual(item.status, True)
            self.assertEqual(len(item.coordinate), 4)
            # 浮点数
            for i in range(4):
                self.assertIsInstance(item.coordinate[i], int)

    def test_result_post_process(self):
        mom_result = MoMResult(status=True)
        mom_result.detailList = [
            DetectionItem(status=True, scene="screw_1", coordinate=[1, 2, 3, 4], accuracy=0.9),
            DetectionItem(status=True, scene="small_screw_8", coordinate=[5, 6, 7, 8], accuracy=0.8),
        ]
        result = self.DCFuseDet.result_post_process(mom_result, w=100, h=100)
        self.assertEqual(len(result.detailList), 2)
        for item in result.detailList:
            self.assertEqual(len(item.coordinate), 8)
            for i in range(8):
                self.assertIsInstance(item.coordinate[i], float)


if __name__ == '__main__':
    unittest.main()
