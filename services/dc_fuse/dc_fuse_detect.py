'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:48:03
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 07:11:26
@FilePath     : dc_fuse_detect.py
@Description  :
'''

from ..yolo import YoloOnnxInfer


class DCFuseDetector(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, nc=12, confThreshold=confThreshold, nmsThreshold=nmsThreshold, task=task)
        self.id2name = {
            0: "brass_plate_6",
            1: "lower_crossbeam_screw_10",
            2: "metal_piece_4",
            3: "no_lower_crossbeam_screw_10",
            4: "no_nut2",
            5: "no_screw_1",
            6: "no_small_screw_8",
            7: "no_upper_crossbeam_screw_9",
            8: "nut_2",
            9: "screw_1",
            10: "small_screw_8",
            11: "upper_crossbeam_screw_9",
        }
