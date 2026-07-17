'''
@Author       : gongzhang4
@Date         : 2026-01-07 06:16:55
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-23 12:46:31
@FilePath     : vision_infer.py
@Description  : 后端无关的视觉模型推理模板
'''

import numpy as np
from services.inference import InferenceRunner
from utils import vision_logger
import time
from schemas.data_base import DetectResult
from schemas.exceptions import ModelInferenceError

class BaseVisionInfer:
    def __init__(
        self,
        runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
    ):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.runner = runner
        self.input_names = [item.name for item in self.runner.input_infos]
        self.output_names = [item.name for item in self.runner.output_infos]
        self._input_model_shape = list(self.runner.input_infos[0].shape)
        self.providers = list(self.runner.providers)
        vision_logger.info(
            f"输入维度: {self._input_model_shape}, 输出数量: {len(self.output_names)}"
        )

    @property
    def input_model_shape(self):
        return self._input_model_shape

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
            vision_logger.debug("模型预处理时间: {:.4f}秒", end - start)
            start = time.time()
            outputs = self.runner.run({self.input_names[0]: tensor})
            end = time.time()
            vision_logger.debug("模型推理时间: {:.4f}秒", end - start)
            start = time.time()
            result = self.post_process(outputs, meta)
            end = time.time()
            vision_logger.debug("模型后处理时间: {:.4f}秒", end - start)
            return result
        except ModelInferenceError:
            raise
        except Exception as e:
            # 推理失败必须向上暴露，避免被静默吞成空结果，导致线上故障无法区分
            vision_logger.error(f"推理过程中发生错误: {e}")
            raise ModelInferenceError("模型推理失败", original_error=str(e))

    def close(self) -> None:
        self.runner.close()
