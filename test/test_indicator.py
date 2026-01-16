'''
单元测试文件 - 符合Python测试规范
@Author       : gongzhang4
@Date         : 2026-01-07 09:05:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-16 06:52:37
@FilePath     : test_indicator.py
@Description  : 指示灯检测 API响应测试
'''

from test_base import TestBase
import json
import unittest


class TestIndicatorResponse(TestBase):

    def setUp(self):
        super().setUp()
        self.test_image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/demo/1/A0SW0030/IMG_20251007_102900_336.jpg"
        self.url = f"{self.base_url}/api/v1/indicator_light_detect/"
        self.data_dict = {
            "product": "六无熔丝盒无磁环",  # 产品号
            "type": "A0SW0030",  # 型号
            "modelParams": {"type": 1, "register": False},  # 嵌套的模型参数  # 型号参数值
        }

    def test_indicator_detection_success(self):
        """测试指示灯检测API - 成功场景"""
        response_text = self._make_api_request(self.test_image_path, self.url, self.data_dict)
        print("response_text:", response_text)
        # 验证响应格式
        self._validate_response_format(response_text)
        print("指示灯检测测试通过")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
