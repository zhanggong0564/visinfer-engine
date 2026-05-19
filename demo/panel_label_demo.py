'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:42:41
@LastEditors  : zhanggong
@LastEditTime : 2026-05-15
@FilePath     : panel_label_demo.py
@Description  : 面板标签检测 Demo — 批量推理并可视化
'''

import sys

sys.path.append("..")

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from services.panel_label import OCRPipeline, PanelLabelJudgeApi, PRODUCT_guideline
from config import settings
from schemas import InputParamsBusiness

# 按需测试的产品类型列表
TYPES = [
    "QF2",
    "PE1-A",
    "PE1-B",
    "T1",
    "PH",
    "S1S2",
    "D1",
    "QF1L1",
    "QF1L2",
    "QF1L3",
    "XB3",
    "J28J30",
    "J3",
    "J46",
    "1017KM1_1",
    "1017KM1_2",
    "1017KM3_1",
    "1017KM3_2",
    "201Q1",
    "201X1_1",
    "201X1_2",
    "101FU601FU_1",
    "101FU601FU_2",
    "401A1",
    "401A2",
    "401A4",
    "401A6",
    "1020U1_1",
    "1020U1_2",
    "1022U1_1",
    "1022U1_2",
    "1019U1_1",
    "1019U1_2",
    "901X1",
    "901X2",
]

DATA_DIR = Path("./demo/data/test")
VIS_DIR = Path("./demo/test/vis")


def visualize_results(image_src, results, product_type, dst_path):
    """在图像上绘制 guideline 矩形和检测结果多边形"""
    h, w, _ = image_src.shape

    if product_type in PRODUCT_guideline:
        gx, gy, gw, gh = PRODUCT_guideline[product_type]
        cv2.rectangle(
            image_src,
            (int(gx * w), int(gy * h)),
            (int(gx * w + gw * w), int(gy * h + gh * h)),
            (0, 255, 0),
            2,
        )

    for detail in results.to_dict().get("detailList", []):
        coord = detail.get("coordinate", [])
        if len(coord) != 8:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4 = coord
        points = np.array(
            [
                [
                    [int(x1 * w), int(y1 * h)],
                    [int(x2 * w), int(y2 * h)],
                    [int(x3 * w), int(y3 * h)],
                    [int(x4 * w), int(y4 * h)],
                ]
            ],
            dtype=np.int32,
        )
        is_ok = detail.get("status") == "true"
        color = (0, 255, 0) if is_ok else (0, 0, 255)
        cv2.polylines(image_src, points, True, color, 2)
        cv2.putText(
            image_src,
            detail.get("name", ""),
            (int(x1 * w), int(y1 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    cv2.imwrite(str(dst_path), image_src)


def run(types=None):
    """批量推理给定的产品类型"""
    if types is None:
        types = TYPES

    detector = PanelLabelJudgeApi(settings)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for product_type in types:
        image_dir = DATA_DIR / product_type
        if not image_dir.is_dir():
            print(f"[跳过] 无数据目录: {image_dir}")
            continue

        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            print(f"[跳过] {product_type}: 无图片")
            continue

        print(f"\n{'='*60}")
        print(f"类型: {product_type} | 图片数: {len(image_paths)}")
        print(f"{'='*60}")

        positive = 0
        for image_path in image_paths:
            print(f"  {image_path}")

            image_src = cv2.imread(str(image_path))
            if image_src is None:
                print(f"  无法读取: {image_path.name}")
                continue

            input_params = InputParamsBusiness(
                image=image_src,
                product_type=product_type,
            )
            results = detector.detect(input_params)
            status = results.to_dict()["status"]
            is_ok = status == "true"

            if is_ok:
                positive += 1

            dst_path = VIS_DIR / f"{product_type}_{image_path.stem}_res.jpg"
            visualize_results(image_src, results, product_type, dst_path)

            if not is_ok:
                print(f"    FAIL: vis_path: {dst_path}, src_path: {image_path}")

        accuracy = positive / len(image_paths)
        print(f"  positive_num: {positive}, total_num: {len(image_paths)}, accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    # 屏蔽 vision_logger 的控制台 INFO/DEBUG 输出，终端只显示 WARNING 及以上
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="WARNING",
        colorize=True,
    )
    run(types=["1019U1_2"])
