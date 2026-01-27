'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:16:55
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 03:07:28
@FilePath     : onnx_base.py
@Description  :
'''

import numpy as np
import onnxruntime
from ..utils.utils import *
from utils import vision_logger
import time
from ..data_base import DetectResult

# 设置onnxruntime日志级别
onnxruntime.set_default_logger_severity(3)


class BaseOnnxInfer:
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, providers=None):
        self.model_path = model_path
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        available_providers = onnxruntime.get_available_providers()
        self.providers = (
            providers or ["CUDAExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        self.r = None
        self.dw = None
        self.dh = None

        # 初始化模型
        self._initialize_session(model_path)
        self._warmup()

    @property
    def input_model_shape(self):
        return self._input_model_shape

    def _get_input_details(self):
        """获取模型输入信息"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self._input_model_shape = model_inputs[0].shape

    def _get_output_details(self):
        """获取模型输出信息"""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def _initialize_session(self, model_path):
        """初始化ONNX模型"""
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers, sess_options=sess_options)
        vision_logger.info(f"使用的执行提供程序: {self.session.get_providers()}")
        self._get_input_details()
        self._get_output_details()
        vision_logger.info(f"输入维度: {self._input_model_shape}, 输出数量: {len(self.output_names)}")

    def _warmup(self):
        """预热模型"""
        dummy_input = np.zeros(self._input_model_shape, dtype=np.float32)
        start = time.perf_counter()
        self.session.run(self.output_names, {self.input_names[0]: dummy_input})
        end = time.perf_counter()
        vision_logger.info(f"预热时间: {end - start:.4f}秒")

    def preprocess(self, im):
        raise NotImplementedError("preprecess method must be implemented in subclass")

    def post_process(self, output_data) -> DetectResult:
        """后处理模型输出结果

        注意：这个方法需要在子类中具体实现

        Args:
            result: 模型输出结果

        Returns:
            DetectResult: 处理后的结果，包含rect、score、cls等键值对
        """
        raise NotImplementedError("post_process method must be implemented in subclass")

    def infer(self, img: np.ndarray) -> DetectResult:
        """执行推理过程

        Args:
            img (np.ndarray): 输入图像

        Returns:
            DetectResult: 推理结果
        """
        try:
            self.image_src_shape = img.shape
            img = self.preprocess(img)
            outputs = self.session.run(self.output_names, {self.input_names[0]: img})
            result = self.post_process(outputs)
            return result
        except Exception as e:
            vision_logger.error(f"推理过程中发生错误: {e}")
            return DetectResult()
