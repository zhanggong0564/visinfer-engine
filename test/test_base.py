'''
@Author       : gongzhang4
@Date         : 2026-01-08 05:02:37
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 06:36:01
@FilePath     : test_base.py
@Description  :
'''

import unittest
import requests
import json


class TestBase(unittest.TestCase):
    """测试基础类"""

    def setUp(self):
        self.base_url = "http://0.0.0.0:8090"

    def _make_api_request(self, image_path, url, json_data):
        """发送API请求的辅助方法"""
        data = {"json_data": json.dumps(json_data)}
        with open(image_path, "rb") as image_file:
            files = [("file", ("test_image.jpg", image_file, "image/jpeg"))]
            headers = {}
            response = requests.request("POST", url, headers=headers, data=data, files=files)
        return response.text

    def _validate_response_format(self, response_text):
        """验证标准响应格式"""
        # 解析JSON响应
        response = json.loads(response_text)

        # 基本字段检查
        self.assertIn("code", response)
        self.assertIn("message", response)
        self.assertIn("result", response)

        # 类型检查
        self.assertIsInstance(response["code"], int)
        self.assertIsInstance(response["message"], str)
        self.assertIsInstance(response["result"], dict)

        # 结果字段检查
        result = response["result"]
        self.assertIn("detailList", result)
        self.assertIsInstance(result["detailList"], list)

        self.assertIn("status", result)
        # 修正：status可能是字符串或布尔值
        self.assertTrue(
            isinstance(result["status"], (bool, str)),
            f"status字段应该是布尔值或字符串，实际是: {type(result['status'])}",
        )

        # 详细列表检查
        for item in result["detailList"]:
            self.assertIn("status", item)
            # 修正：status可能是字符串或布尔值
            self.assertTrue(
                isinstance(item["status"], (bool, str)),
                f"detailList中的status字段应该是布尔值或字符串，实际是: {type(item['status'])}",
            )
            self.assertIn("accuracy", item)
            self.assertIsInstance(item["accuracy"], float)
            self.assertIn("scene", item)
            self.assertIsInstance(item["scene"], str)
            self.assertIn("coordinate", item)
            self.assertIsInstance(item["coordinate"], list)
            # 修正：坐标可能是8个值（4个点）而不是4个值,
            self.assertTrue(
                len(item["coordinate"]) in [0, 8], f"coordinate字段应该有0或8个值，实际有: {len(item['coordinate'])}"
            )
            for coord in item["coordinate"]:
                self.assertIsInstance(coord, (int, float))
