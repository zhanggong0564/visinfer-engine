'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 09:02:18
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
from itertools import chain
import time
from utils import vision_logger
from .product_type import PRODUCT_guideline


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
    def __init__(
        self, detect_model_path, orient_model_path, text_recognition_model_path, confThreshold=0.5, nmsThreshold=0.5
    ):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
            textline_orientation_model_dir=orient_model_path,
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir=text_recognition_model_path,
        )

    def infer(self, image) -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        points_line = []
        points_qfu = []
        if 1 in class_ids:
            points_qfu = mask_polygons[class_ids == 1]
        if 0 in class_ids:
            points_line = mask_polygons[class_ids == 0]

        # panel_label_item = PanellabelItem()
        start = time.time()
        mask_rois, sorted_idxs = Points_to_Mask(image, points_line)
        end = time.time()
        vision_logger.info(f"Points_to_Mask: {end - start:.4f}秒")
        start = time.time()
        rec_preds = self.ocr.predict(
            mask_rois,
            use_textline_orientation=True,
            text_det_unclip_ratio=2.0,
            text_det_box_thresh=0.4,
            text_rec_score_thresh=0.7,
            text_det_limit_side_len=64 * 2,
            # max_side_limit=40000,
        )
        end = time.time()
        vision_logger.info(f"ocr.predict: {end - start:.4f}秒")
        # for i, mask_roi in enumerate(mask_rois):
        #     cv2.imwrite(f"./demo/vis/mask_roi_{i}.jpg", mask_roi)
        # start = time.time()
        # texts = [pred['rec_texts'][0] for pred in rec_preds]
        texts = []
        for pred in rec_preds:
            rec_texts = pred.get('rec_texts') if len(pred.get('rec_texts')) > 0 else ['none']
            rec_scores = pred.get('rec_scores') if len(pred.get('rec_scores')) > 0 else [0]
            if not rec_texts or not rec_scores or len(rec_texts) != len(rec_scores):
                continue

            pairs = [(t, s) for t, s in zip(rec_texts, rec_scores) if isinstance(t, str) and t.strip() != ""]
            if not pairs:
                continue
            # 取分数最高的文本
            best_text, best_score = max(pairs, key=lambda x: float(x[1]))
            texts.append(best_text)

        # texts = [pred['rec_texts'][0] for pred in rec_preds if pred.get('rec_texts') and len(pred['rec_texts']) > 0]
        positions = [
            np.int64(cv2.boxPoints(cv2.minAreaRect(np.array(mask_polygon, dtype=np.float32))))
            for mask_polygon in mask_polygons
        ]
        ori_index = [np.where(class_ids == 0)[0][sorted_idx] for sorted_idx in sorted_idxs]
        positions = [list(chain.from_iterable(positions[idx].tolist())) for idx in ori_index]
        roi_classes_ids = class_ids[ori_index]
        confidences = [results.scores[idx] for idx in ori_index]
        panel_label_item = PanellabelItem(
            Points=positions,
            index=ori_index,
            class_id=roi_classes_ids.tolist(),
            texts=texts,
            confidence=confidences,
        )

        return panel_label_item


class OCRPipelineCrop:
    def __init__(self, detect_model_path, orient_model_path, confThreshold=0.5, nmsThreshold=0.5):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
            textline_orientation_model_dir=orient_model_path,
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir='/data/zhanggong/workspace/project/move_vsion/mobile_vision/weights/panel_label/PP-OCRv5_server_rec_plane_infer',
        )

    def infer(self, image, sort_by="xy") -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        points_qfu = mask_polygons[class_ids == 1]
        points_line = mask_polygons[class_ids == 0]

        # panel_label_item = PanellabelItem()
        start = time.time()
        mask_rois, sorted_idxs = Points_to_Mask(image, points_line, sort_by=sort_by)
        end = time.time()
        vision_logger.info(f"Points_to_Mask: {end - start:.4f}秒")
        start = time.time()
        return mask_rois
