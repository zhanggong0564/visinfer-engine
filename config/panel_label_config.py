'''
@Author       : gongzhang4
@Date         : 2026-03-02 08:01:03
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-28 05:37:35
@FilePath     : panel_label_config.py
@Description  :
'''


class PanelLabelConfig:
    model_path = "./weights/panel_label/best_v1.onnx"
    orient_model_path = "./weights/panel_label/PP-LCNet_x1_0_textline_ori"
    text_recognition_model_path = "./weights/panel_label/PP-OCRv5_server_rec_plane_infer"
    confThreshold = 0.7
    nmsThreshold = 0.8
