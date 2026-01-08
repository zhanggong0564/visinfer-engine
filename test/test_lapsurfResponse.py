'''
单元测试文件 - 符合Python测试规范
@Author       : gongzhang4
@Date         : 2026-01-07 09:05:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 06:36:57
@FilePath     : test_lapsurfResponse.py
@Description  : DCFuse API响应测试
'''

from test_base import TestBase
import json
import unittest


class TestLapSurfResponse(TestBase):

    def setUp(self):
        super().setUp()
        self.test_image_path = "./images/IMG_20251118_153457_263.jpg"
        self.url = f"{self.base_url}/api/v1/lap_surf_detect/"
        self.data_dict = {}

    def test_mobile_vision_identification_success(self):
        """测试移动视觉识别API - 成功场景"""
        response_text = self._make_api_request(self.test_image_path, self.url, self.data_dict)
        print(response_text)
        # 验证响应格式
        self._validate_response_format(response_text)
        print("移动视觉识别测试通过")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
