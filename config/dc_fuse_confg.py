'''
@Author       : gongzhang4
@Date         : 2026-01-23 06:42:36
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-23 07:15:03
@FilePath     : dc_fuse_confg.py
@Description  : 直流熔丝场景的配置
'''


class DcFuseConfig:
    model_path: str = "./weights/dc_fuse_v5.onnx"
    confThreshold: float = 0.6
