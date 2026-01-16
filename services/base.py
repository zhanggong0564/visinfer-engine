'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:16:55
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 07:25:36
@FilePath     : base.py
@Description  :
'''

import numpy as np
import onnxruntime
from .utils import *
from utils import vision_logger


# 设置onnxruntime日志级别
onnxruntime.set_default_logger_severity(3)


class BaseOnnxInfer:
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, providers=None):
        self.model_path = model_path
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.r = None
        self.dw = None
        self.dh = None

        # 初始化模型
        self.init_model(model_path, confThreshold)

    def get_input_details(self):
        """获取模型输入信息"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_model_shape = model_inputs[0].shape

    def get_output_details(self):
        """获取模型输出信息"""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def init_model(self, model_path, conf_th):
        """初始化ONNX模型"""
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        vision_logger.info(f"使用的执行提供程序: {self.session.get_providers()}")
        self.get_input_details()
        self.get_output_details()
        self.session.run(self.output_names, {self.input_names[0]: np.zeros(self.input_model_shape, dtype=np.float32)})
        vision_logger.info("warmup done success")

    def preprocess(self, im):
        """预处理输入图像

        Args:
            im (np.ndarray): 输入图像

        Returns:
            np.ndarray: 处理后的图像
        """
        img, self.r, self.dw, self.dh = letterbox(im=im, auto=False, new_shape=self.input_model_shape[2:])
        im = np.stack([img])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im).astype(np.float32)
        im /= 255.0  # 归一化到0-1
        return im

    def post_process(self, output_data):
        """后处理模型输出结果

        注意：这个方法需要在子类中具体实现

        Args:
            result: 模型输出结果

        Returns:
            dict: 处理后的结果，包含rect、score、cls等键值对
        """
        raise NotImplementedError("post_process method must be implemented in subclass")

    def infer(self, img):
        """执行推理过程

        Args:
            img (np.ndarray): 输入图像

        Returns:
            dict: 推理结果
        """
        self.image_src_shape = img.shape
        img = self.preprocess(img)
        self.input_model_shape = img.shape
        outputs = self.session.run(self.output_names, {self.input_names[0]: img})
        result = self.post_process(outputs)
        return result
