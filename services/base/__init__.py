'''
@Author       : gongzhang4
@Date         : 2026-01-27 03:07:47
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 03:08:53
@FilePath     : __init__.py
@Description  :
'''

from .onnx_base import BaseOnnxInfer
from .business_logic_base import BusinessLogicBase


__all__ = ["BaseOnnxInfer", "BusinessLogicBase"]
