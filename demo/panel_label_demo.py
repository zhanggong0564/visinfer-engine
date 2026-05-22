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

# 数据目录配置
DATA_DIR = Path("./demo/data/charging_pile_test")
VIS_DIR = Path("./demo/test/vis")


def auto_detect_product_types(data_dir):
    """自动从数据目录中检测产品类型（子文件夹名）"""
    if not data_dir.is_dir():
        print(f"错误: 数据目录不存在: {data_dir}")
        return []

    product_types = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            # 检查是否有 jpg 图片
            if next(item.glob("*.jpg"), None) is not None:
                product_types.append(item.name)

    return product_types


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


def run(types=None, rule="all"):
    """批量推理给定的产品类型，最后输出所有型号的正确率汇总"""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 自动检测产品类型
    if types is None:
        product_types = auto_detect_product_types(DATA_DIR)
        if not product_types:
            print("未检测到任何产品类型")
            return
        print(f"检测到的产品类型 ({len(product_types)} 个): {', '.join(product_types)}")
    else:
        product_types = types

    detector = PanelLabelJudgeApi(settings)

    # 用于存储所有型号的正确率
    accuracy_summary = {}

    for product_type in product_types:
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
            image_src = cv2.imread(str(image_path))
            if image_src is None:
                print(f"  无法读取: {image_path.name}")
                continue

            input_params = InputParamsBusiness(
                image=image_src,
                product_type=product_type,
                rule=rule,
            )
            results = detector.detect(input_params)
            status = results.to_dict()["status"]
            is_ok = status == "true"

            if is_ok:
                positive += 1

            dst_path = VIS_DIR / f"{product_type}_{image_path.stem}_res.jpg"
            visualize_results(image_src, results, product_type, dst_path)

            # if not is_ok:
            print(f"    FAIL: vis_path: {dst_path}, src_path: {image_path}")

        accuracy = positive / len(image_paths) if len(image_paths) > 0 else 0
        accuracy_summary[product_type] = {
            "positive": positive,
            "total": len(image_paths),
            "accuracy": accuracy,
        }
        print(f"  正确数: {positive}, 总数: {len(image_paths)}, 正确率: {accuracy:.2%}")

    # 输出所有型号的正确率汇总
    print(f"\n\n{'='*80}")
    print(f"所有型号正确率汇总 (共 {len(accuracy_summary)} 个型号)")
    print(f"{'='*80}")
    print(f"{'型号':<20} {'正确数':>8} {'总数':>8} {'正确率':>10}")
    print(f"{'-'*80}")

    total_positive = 0
    total_all = 0
    for product_type, stats in sorted(accuracy_summary.items()):
        print(f"{product_type:<20} {stats['positive']:>8} {stats['total']:>8} {stats['accuracy']:>9.2%}")
        total_positive += stats["positive"]
        total_all += stats["total"]

    overall_accuracy = total_positive / total_all if total_all > 0 else 0
    print(f"{'-'*80}")
    print(f"{'总计':<20} {total_positive:>8} {total_all:>8} {overall_accuracy:>9.2%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=["front", "back", "all"], default="all", help="字符比较规则")
    parser.add_argument("--types", nargs="*", default=None, help="指定产品类型，不填则自动检测")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="WARNING",
        colorize=True,
    )
    run(types=args.types, rule=args.rule)
