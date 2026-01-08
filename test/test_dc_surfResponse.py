'''
@Author       : gongzhang4
@Date         : 2026-01-08 05:49:22
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-08 05:49:23
@FilePath     : test_lapsurfResponse copy.py
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
        self.test_image_path = "./images/lQDPJwfWMYLzRv_NC9DND8CwNWP5oG42qP0I3k_Zt4wLAA_4032_3024.jpg"
        self.url = f"{self.base_url}/api/v1/dcfuse_detect/"
        self.data_dict = {
            "product": "七路无熔丝盒无磁环",  # 产品号
            "type": "WLSH-7",  # 型号
            "modelParams": {"product_model": "七路无熔丝盒无磁环"},  # 嵌套的模型参数  # 型号参数值
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
