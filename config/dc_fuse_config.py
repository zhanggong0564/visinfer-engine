'''
@Description : 直流熔丝场景配置（服务内场景形态，独立配置文件，不侵入框架 Settings）
'''


class DcFuseConfig:
    """直流熔丝检测配置。

    服务形态下不挂到 config/config.py 的全局 Settings（那是框架共享配置），
    单独成文件由场景业务逻辑直接 import，保持"加场景不动框架"。
    """
    model_path = "./weights/dc_fuse/det_yolo_v5.onnx"
    confThreshold = 0.6
