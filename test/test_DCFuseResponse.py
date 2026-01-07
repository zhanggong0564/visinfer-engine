'''
单元测试文件 - 符合Python测试规范
@Author       : gongzhang4
@Date         : 2026-01-07 09:05:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 09:31:02
@FilePath     : test_DCFuseResponse.py
@Description  : DCFuse API响应测试
'''

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import requests


class TestDCFuseResponse(unittest.TestCase):
    """DCFuse API响应测试类"""

    def setUp(self):
        """测试前置设置"""
        self.test_image_path = "./images/lQDPJwfWMYLzRv_NC9DND8CwNWP5oG42qP0I3k_Zt4wLAA_4032_3024.jpg"
        self.base_url = "http://0.0.0.0:8090"

    def _make_api_request(self, image_path, url, json_data):
        """发送API请求的辅助方法"""
        data = {"json_data": json.dumps(json_data)}
        with open(image_path, "rb") as image_file:
            files = [("file", ("test_image.jpg", image_file, "image/jpeg"))]
            headers = {}
            response = requests.request("POST", url, headers=headers, data=data, files=files)
        return response.text

    def test_mobile_vision_identification_success(self):
        """测试移动视觉识别API - 成功场景"""
        url = f"{self.base_url}/api/v1/dcfuse_detect/"
        data_dict = {
            "product": "七路无熔丝盒无磁环",  # 产品号
            "type": "WLSH-7",  # 型号
            "modelParams": {"product_model": "七路无熔丝盒无磁环"},  # 嵌套的模型参数  # 型号参数值
        }

        response_text = self._make_api_request(self.test_image_path, url, data_dict)
        print(response_text)

        # 验证响应格式
        self._validate_response_format(response_text)
        print("移动视觉识别测试通过")

    @patch('requests.request')
    def test_mobile_vision_identification_network_error(self, mock_request):
        """测试移动视觉识别API - 网络错误场景"""
        mock_request.side_effect = requests.exceptions.ConnectionError("连接失败")

        url = f"{self.base_url}/api/v1/dcfuse_detect/"
        data_dict = {
            "product": "七路无熔丝盒无磁环",  # 产品号
            "type": "WLSH-7",  # 型号
            "model_params": {"product_model": "七路无熔丝盒无磁环"},  # 嵌套的模型参数  # 型号参数值
        }

        with self.assertRaises(requests.exceptions.ConnectionError):
            self._make_api_request(self.test_image_path, url, data_dict)

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
            # 修正：坐标可能是8个值（4个点）而不是4个值
            self.assertTrue(
                len(item["coordinate"]) in [4, 8], f"coordinate字段应该有4或8个值，实际有: {len(item['coordinate'])}"
            )
            for coord in item["coordinate"]:
                self.assertIsInstance(coord, (int, float))


class TestResponseValidation(unittest.TestCase):
    """响应验证功能测试"""

    def test_valid_response(self):
        """测试有效响应验证"""
        valid_response = {
            "code": 1,
            "message": "成功",
            "result": {
                "detailList": [
                    {"coordinate": [1.0, 2.0, 3.0, 4.0], "status": True, "scene": "检测场景", "accuracy": 0.95}
                ],
                "status": True,
            },
        }

        response_text = json.dumps(valid_response)
        validator = TestDCFuseResponse()
        validator._validate_response_format(response_text)

    def test_invalid_response_missing_field(self):
        """测试缺失字段的响应"""
        invalid_response = {
            "code": 1,
            "message": "成功",
            # 缺少result字段
        }

        response_text = json.dumps(invalid_response)
        validator = TestDCFuseResponse()

        with self.assertRaises(AssertionError):
            validator._validate_response_format(response_text)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
