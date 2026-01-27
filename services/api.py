'''
@Author       : gongzhang4
@Date         : 2026-01-23 09:26:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 05:27:35
@FilePath     : api.py
@Description  :
'''

from config import settings
from .base.onnx_base import BaseOnnxInfer
from typing import Dict


class ApiFactory:
    _instance = None
    _registry: Dict[str, type] = {}

    def register(self, api_type: str):
        def decorator(api_class: type):
            self._registry[api_type] = api_class
            return api_class

        return decorator

    def get_scenarios(self, api_type: str) -> BaseOnnxInfer:
        if api_type in self._registry:
            api_class = self._registry[api_type]
            return api_class(settings)
        else:
            raise ValueError(f"API type {api_type} not registered")

    def list_scenarios(self) -> list:
        return list(self._registry.keys())


detection_factory = ApiFactory()
# if __name__ == '__main__':
#     detector = ApiFactory.create_api("dc_fuse")
