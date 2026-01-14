'''
@Author       : gongzhang4
@Date         : 2026-01-08 05:49:22
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-14 06:52:52
@FilePath     : test_dc_surfResponse.py
@Description  :
'''

'''
单元测试文件 - 符合Python测试规范
@Author       : gongzhang4
@Date         : 2026-01-07 09:05:49
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 05:47:44
@FilePath     : test_lapsurfResponse.py
@Description  : DCFuse API响应测试
'''

from test_base import TestBase
import json
import unittest


class TestDCFuseResponse(TestBase):

    def setUp(self):
        super().setUp()
        self.test_image_path = "./images/20260106145926_0pcejci1.jpeg"
        self.url = f"{self.base_url}/api/v1/dcfuse_detect/"
        self.data_dict = {
            "product": "六无熔丝盒无磁环",  # 产品号
            "type": "WLSH-7",  # 型号
            "modelParams": {"product_model": "六路无熔丝盒无磁环"},  # 嵌套的模型参数  # 型号参数值
        }

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
