'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:16:55
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-23 12:46:31
@FilePath     : onnx_base.py
@Description  :
'''

import numpy as np
import onnxruntime
from ..utils.utils import *
from utils import vision_logger
import time
from schemas.data_base import DetectResult
from schemas.exceptions import ModelInferenceError

# 设置onnxruntime日志级别
onnxruntime.set_default_logger_severity(3)


class BaseOnnxInfer:
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, providers=None):
        self.model_path = model_path
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        available_providers = onnxruntime.get_available_providers()
        if providers is None:
            # 优先 CUDA，但始终保留 CPU 兜底：遇到 CUDA 不支持的算子时可回退而非直接报错
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

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
        """预热模型。

        ONNX 动态轴会以字符串（如 'batch'）或 -1/None 出现在 input shape 里，
        直接 np.zeros 会崩。这里把 batch 维兜底为 1，其余非法维若无法定形则
        跳过预热（仅 warning），避免动态尺寸模型一加载就报错。
        """
        concrete_shape = []
        for idx, dim in enumerate(self._input_model_shape):
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            elif idx == 0:
                concrete_shape.append(1)  # 动态 batch 维兜底为 1
            else:
                vision_logger.warning(
                    f"模型输入含动态维度 {self._input_model_shape}，跳过预热"
                )
                return
        dummy_input = np.zeros(concrete_shape, dtype=np.float32)
        start = time.perf_counter()
        self.session.run(self.output_names, {self.input_names[0]: dummy_input})
        end = time.perf_counter()
        vision_logger.info(f"预热时间: {end - start:.4f}秒")

    def preprocess(self, im):
        """预处理输入图像（子类实现）。

        Returns:
            tuple: (模型输入张量 np.ndarray, PreprocMeta)
        """
        raise NotImplementedError("preprecess method must be implemented in subclass")

    def post_process(self, output_data, meta) -> DetectResult:
        """后处理模型输出结果（子类实现）。

        Args:
            output_data: 模型原始输出
            meta (PreprocMeta): 预处理产生的每请求缩放/原图元数据

        Returns:
            DetectResult: 处理后的结果
        """
        raise NotImplementedError("post_process method must be implemented in subclass")

    def infer(self, img: np.ndarray) -> DetectResult:
        """执行推理过程（无状态：每请求态走局部 meta，不写 self）。

        Args:
            img (np.ndarray): 输入图像

        Returns:
            DetectResult: 推理结果
        """
        try:
            # 仅分割任务后处理需要原图，其余存引用即可，避免每请求复制整张大图
            ori_img = img.copy() if getattr(self, "task", None) == "seg" else img
            start = time.time()
            tensor, meta = self.preprocess(img)
            meta.ori_img = ori_img
            end = time.time()
            vision_logger.debug("YOLO预处理时间: {:.4f}秒", end - start)
            start = time.time()
            outputs = self.session.run(self.output_names, {self.input_names[0]: tensor})
            end = time.time()
            vision_logger.debug("YOLO推理时间: {:.4f}秒", end - start)
            start = time.time()
            result = self.post_process(outputs, meta)
            end = time.time()
            vision_logger.debug("YOLO后处理时间: {:.4f}秒", end - start)
            return result
        except Exception as e:
            # 推理失败必须向上暴露，避免被静默吞成空结果，导致线上故障无法区分
            vision_logger.error(f"推理过程中发生错误: {e}")
            raise ModelInferenceError("模型推理失败", original_error=str(e))
