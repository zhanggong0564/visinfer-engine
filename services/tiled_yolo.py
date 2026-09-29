"""YOLO extension with preprocessing crops, sequential inference and result merging."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from schemas.data_base import DetectResult
from schemas.inference_context import PreprocMeta
from services.base.detection import Detection, DetectionResult
from services.base.geometry import CoordinateSpace, Region, xyxy_region
from services.inference import InferenceRunner
from services.yolo import YoloInfer
from services.yolo_ops import prepare_yolo_input, restore_yolo_boxes, run_yolo_nms


@dataclass
class TiledPreprocMeta:
    """Request-local original shape and each tile's region/resize metadata."""

    src_shape: tuple[int, ...]
    tiles: tuple[tuple[Region, PreprocMeta], ...]
    ori_img: np.ndarray | None = None


def region_box(region: Region) -> tuple[float, float, float, float]:
    if region.space != CoordinateSpace.PIXEL or not region.polygon:
        raise ValueError("tile detection requires a nonempty pixel-space region")
    xs, ys = zip(*region.polygon)
    box = min(xs), min(ys), max(xs), max(ys)
    if not all(isfinite(value) for value in box):
        raise ValueError("tile detection coordinates must be finite")
    return box


def quarter_regions(height: int, width: int, overlap: int = 200) -> list[Region]:
    """Four crops with total central overlap; small images remain one crop."""
    if min(height, width) <= 0 or overlap < 0:
        raise ValueError("image dimensions must be positive and overlap nonnegative")
    if min(height, width) <= max(overlap, 1):
        return [xyxy_region([0, 0, width, height])]
    before, after = overlap // 2, overlap - overlap // 2
    xs = ((0, width // 2 + after), (width // 2 - before, width))
    ys = ((0, height // 2 + after), (height // 2 - before, height))
    return [xyxy_region([left, top, right, bottom])
            for left, right in xs for top, bottom in ys]


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def deduplicate_detections(
    detections: Sequence[Detection],
) -> list[Detection]:
    """Suppress any positive-area same-class overlap, ranked by area*score."""
    ranked = []
    for det in detections:
        box = region_box(det.region)
        area = _area(box)
        if area <= 0 or det.mask is not None:
            raise ValueError("tile detection requires positive-area boxes without masks")
        if not isfinite(det.score) or not 0 <= det.score <= 1:
            raise ValueError("detection score must be finite and in [0, 1]")
        ranked.append((det, box, area))
    ranked.sort(key=lambda item: (
        item[2] * item[0].score, item[0].score, item[2]), reverse=True)
    kept = []
    for det, box, area in ranked:
        duplicate = False
        for previous, other, _ in kept:
            if (det.class_id, det.class_name) != (previous.class_id, previous.class_name):
                continue
            intersection = (
                max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
                * max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
            )
            if intersection > 0:
                duplicate = True
                break
        if not duplicate:
            kept.append((det, box, area))
    return [det for det, _, _ in kept]


def merge_tile_results(
    tiles: Sequence[tuple[Region, DetectionResult]],
) -> DetectionResult:
    """Translate tile-local boxes to original pixels, then deduplicate."""
    detections = []
    for region, result in tiles:
        left, top, _, _ = region_box(region)
        for det in result.detections:
            x1, y1, x2, y2 = region_box(det.region)
            detections.append(Detection(
                xyxy_region([x1 + left, y1 + top, x2 + left, y2 + top]),
                det.score, det.class_id, det.class_name, mask=det.mask,
            ))
    return DetectionResult(detections=deduplicate_detections(detections))


class TiledYoloInfer(YoloInfer):
    def __init__(
        self,
        nc: int,
        runner: InferenceRunner,
        confThreshold: float = 0.5,
        nmsThreshold: float = 0.5,
        task: str = "det",
        *,
        tiled_inference: bool = False,
        tile_overlap: int = 200,
    ) -> None:
        if tiled_inference and task != "det":
            raise ValueError("tiled YOLO only supports box detection")
        if tile_overlap < 0:
            raise ValueError("tile overlap must be nonnegative")
        super().__init__(nc, runner, confThreshold, nmsThreshold, task)
        self.tiled_inference = tiled_inference
        self.tile_overlap = tile_overlap

    def preprocess(
        self, im: np.ndarray,
    ) -> tuple[np.ndarray, PreprocMeta] | tuple[list[np.ndarray], TiledPreprocMeta]:
        if not self.tiled_inference:
            return super().preprocess(im)
        tensors, tiles = [], []
        for region in quarter_regions(*im.shape[:2], self.tile_overlap):
            left, top, right, bottom = (int(value) for value in region_box(region))
            tensor, meta = prepare_yolo_input(
                im[top:bottom, left:right], self.input_model_shape[2:],
            )
            tensors.append(tensor)
            tiles.append((region, meta))
        return tensors, TiledPreprocMeta(src_shape=im.shape, tiles=tuple(tiles))

    def model_inference(
        self, model_inputs: np.ndarray | list[np.ndarray],
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        if not self.tiled_inference:
            return super().model_inference(model_inputs)
        # The split model accepts batch=1, so preserve one runner call per tile.
        return [self.runner.run({self.input_names[0]: tensor}) for tensor in model_inputs]

    def post_process(
        self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        meta: PreprocMeta | TiledPreprocMeta,
    ) -> DetectResult:
        if not self.tiled_inference:
            return super().post_process(outputs, meta)
        if not isinstance(meta, TiledPreprocMeta) or len(outputs) != len(meta.tiles):
            raise ValueError("tile outputs and preprocessing metadata must match")
        tiles = []
        for output, (region, tile_meta) in zip(outputs, meta.tiles):
            selected = run_yolo_nms(
                output[0], task="det", conf_threshold=self.confThreshold,
                iou_threshold=self.nmsThreshold, classes=self.filter_classes,
                agnostic=False, nc=self.nc,
            )[0]
            boxes = restore_yolo_boxes(
                selected, self.input_model_shape[2:], tile_meta.src_shape,
            )
            detections = [
                Detection(xyxy_region(row[:4].tolist()), float(row[4]),
                          int(row[5]), self.id2name[int(row[5])])
                for row in boxes
            ]
            tiles.append((region, DetectionResult(detections=detections)))
        merged = merge_tile_results(tiles)
        merged.source_image = meta.ori_img
        return DetectResult(
            boxes=[list(region_box(det.region)) for det in merged.detections],
            scores=[det.score for det in merged.detections],
            class_ids=[det.class_id for det in merged.detections],
            class_names=[det.class_name for det in merged.detections],
            ori_img=merged.source_image,
        )
