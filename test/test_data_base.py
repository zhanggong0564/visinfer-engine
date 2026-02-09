'''
@Author       : gongzhang4
@Date         : 2026-01-23 03:39:17
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 10:15:50
@FilePath     : test_data_base.py
@Description  : 测试数据基础类
'''

import unittest
import json
import sys
from pathlib import Path


sys.path.insert(0, Path(__file__).resolve().parent.parent)
print(Path(__file__).resolve().parent.parent)

from schemas.data_base import DetectResult, DetectionItem


class TestDetectResult(unittest.TestCase):
    def test_init_default(self):
        result = DetectResult()
        self.assertEqual(result.boxes, [])
        self.assertEqual(result.scores, [])
        self.assertEqual(result.class_ids, [])
        self.assertEqual(result.class_names, [])

    def test_init_with_params(self):
        result = DetectResult(boxes=[[1, 2, 3, 4]], scores=[0.5], class_ids=[0], class_names=['person'])
        self.assertEqual(result.boxes, [[1, 2, 3, 4]])
        self.assertEqual(result.scores, [0.5])
        self.assertEqual(result.class_ids, [0])
        self.assertEqual(result.class_names, ['person'])

    def test_modify(self):
        result = DetectResult()
        result.boxes = [[1, 2, 3, 4]]
        result.scores = [0.5]
        result.class_ids = [0]
        result.class_names = ['person']
        self.assertEqual(result.boxes, [[1, 2, 3, 4]])
        self.assertEqual(result.scores, [0.5])
        self.assertEqual(result.class_ids, [0])
        self.assertEqual(result.class_names, ['person'])


class TestDetectionItem(unittest.TestCase):
    def test_init_default(self):
        item = DetectionItem()
        self.assertEqual(item.status, False)
        self.assertEqual(item.scene, "")
        self.assertEqual(item.coordinate, [])
        self.assertEqual(item.accuracy, 0.0)
        self.assertEqual(item.name, "")

    def test_init_with_params(self):
        item = DetectionItem(status=True, scene="dc", coordinate=[1, 2, 3, 4], accuracy=0.8, name="dc_1")
        self.assertEqual(item.status, True)
        self.assertEqual(item.scene, "dc")
        self.assertEqual(item.coordinate, [1, 2, 3, 4])
        self.assertEqual(item.accuracy, 0.8)
        self.assertEqual(item.name, "dc_1")

    def test_to_dict(self):
        item = DetectionItem(status=True, scene="dc", coordinate=[1, 2, 3, 4], accuracy=0.8, name="dc_1")
        item_dict = item.to_dict()
        self.assertIsInstance(item_dict, dict)
        self.assertEqual(
            item_dict,
            {
                "status": 'true',
                "scene": "dc",
                "coordinate": [1, 2, 3, 4],
                "accuracy": 0.8,
                "name": "dc_1",
                "color": "#20ff4f",
            },
        )

    def test_from_dict(self):
        item_dict = {
            "status": True,
            "scene": "dc",
            "coordinate": [1, 2, 3, 4],
            "accuracy": 0.8,
            "name": "dc_1",
        }
        item = DetectionItem.from_dict(item_dict)
        self.assertEqual(item.status, True)
        self.assertEqual(item.scene, "dc")
        self.assertEqual(item.coordinate, [1, 2, 3, 4])
        self.assertEqual(item.accuracy, 0.8)
        self.assertEqual(item.name, "dc_1")
        item_dict = item.to_dict()
        self.assertEqual(item_dict, item_dict)


if __name__ == '__main__':
    unittest.main()
