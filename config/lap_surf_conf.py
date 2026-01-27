'''
@Author       : gongzhang4
@Date         : 2026-01-27 09:26:06
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:26:07
@FilePath     : lap_surf_conf.py
@Description  :
'''


class LapSufConfig:
    model_path: str = "./weights/LapJointSurfRec_v2.onnx"
    confThreshold: float = 0.4
