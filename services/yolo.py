'''
@Author       : gongzhang4
@Date         : 2026-01-27 02:06:28
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-27 03:19:34
@FilePath     : yolo.py
@Description  :
'''

from .base import BaseOnnxInfer
from .utils import *
from collections import defaultdict
from schemas.data_base import DetectResult


class YoloOnnxInfer(BaseOnnxInfer):
    def __init__(self, model_path, nc, confThreshold=0.5, nmsThreshold=0.5, task="det"):
        super().__init__(model_path, confThreshold=confThreshold, nmsThreshold=nmsThreshold)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.agnostic = False
        self.nc = nc
        self.filter_classes = None
        self.task = task

    def preprocess(self, im):
        """预处理输入图像

        Args:
            im (np.ndarray): 输入图像

        Returns:
            np.ndarray: 处理后的图像
        """
        img, self.r, self.dw, self.dh = letterbox(im=im, auto=False, new_shape=self._input_model_shape[2:])
        im = np.stack([img])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im).astype(np.float32)
        im /= 255.0  # 归一化到0-1
        return im

    def post_process(self, preds):
        """后处理输出"""
        p = non_max_suppression_v8(
            preds[0],
            task=self.task,
            conf_thres=self.confThreshold,
            iou_thres=self.nmsThreshold,
            classes=self.filter_classes,
            agnostic=self.agnostic,
            multi_label=False,
            nc=self.nc,
        )
        image_shape = self.image_src_shape[:2]
        input_shape = self.input_model_shape[2:]
        if self.task == "seg":
            protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            mask_in = p[0][:, 6:]
            bboxes = p[0][:, :4]
            masks = process_mask(protos, mask_in, bboxes, input_shape)
            masks = [scale_masks(mask, (image_shape[1], image_shape[0]), self.r, self.dw, self.dh) for mask in masks]
            mask_polygons = [segment for mask in masks for segment in masks2segments(mask)]
        pred = p[0]
        pred[:, :4] = scale_boxes(input_shape, pred[:, :4], image_shape, xywh=False)
        pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
        bbox = pred[:, :4]  # xywh
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(bbox)
        # else:
        #     bbox = xywh2xyxy(bbox)
        detect_result = DetectResult(
            bbox.tolist(),
            pred[:, -2].tolist(),
            pred[:, -1].tolist(),
            [self.id2name[int(cls)] for cls in pred[:, -1]],
            masks=masks if self.task == "seg" else [],
            mask_polygons=mask_polygons if self.task == "seg" else [],
            ori_img=self.ori_img,
        )
        return detect_result
