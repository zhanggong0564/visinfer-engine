'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-02 07:40:42
@FilePath     : panel_label_detect.py
@Description  : 面板标签检测
'''

from ..yolo import YoloOnnxInfer
from ..utils import *
import numpy as np
from schemas.data_base import DetectResult
from paddleocr import TextLineOrientationClassification, TextRecognition, PaddleOCR
from .utils import Points_to_Mask
from typing import List
from dataclasses import dataclass, field


@dataclass
class PanellabelItem:
    Points: List[np.ndarray] = field(default_factory=list)
    index: List[int] = field(default_factory=list)
    class_id: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)

    def save_img(self, image, save_path):
        for i, point in enumerate(self.Points):
            image = cv2.polylines(image, [point], True, (0, 255, 0), 2)
            image = cv2.putText(
                image, self.texts[i], (point[0][0], point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
            )
        cv2.imwrite(save_path, image)


class PanelLabelDetect(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="seg"):
        super().__init__(model_path, 2, confThreshold, nmsThreshold, task)
        self.id2name = {
            0: "line",
            1: "QFU",
        }


class OCRPipeline:
    def __init__(self, detect_model_path, orient_model_path, confThreshold=0.5, nmsThreshold=0.5, task="seg"):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task)
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
            textline_orientation_model_dir=orient_model_path,
        )

    def infer(self, image, sort_by="xy") -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        points_qfu = mask_polygons[class_ids == 1]
        points_line = mask_polygons[class_ids == 0]

        # panel_label_item = PanellabelItem()

        mask_rois, sorted_idxs = Points_to_Mask(image, points_line, sort_by=sort_by)
        for i, mask_roi in enumerate(mask_rois):
            cv2.imwrite(f"./demo/vis/mask_roi_{i}.jpg", mask_roi)
        rec_preds = self.ocr.predict(mask_rois, use_textline_orientation=True, text_det_unclip_ratio=3)
        texts = [pred['rec_texts'][0] for pred in rec_preds]
        positions = [np.int64(cv2.boxPoints(cv2.minAreaRect(mask_polygon))) for mask_polygon in mask_polygons]
        ori_index = [np.where(class_ids == 0)[0][sorted_idx] for sorted_idx in sorted_idxs]
        positions = [positions[idx] for idx in ori_index]
        roi_classes_ids = class_ids[ori_index]
        confidences = results.confidences[ori_index]
        panel_label_item = PanellabelItem(
            Points=positions,
            index=ori_index,
            class_id=roi_classes_ids.tolist(),
            texts=texts,
            confidence=confidences.tolist(),
        )

        return panel_label_item
