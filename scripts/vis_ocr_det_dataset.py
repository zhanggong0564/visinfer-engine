"""
可视化 PaddleOCR 检测数据集标注框。

用法：
    # 可视化 train.txt 前 10 张，输出到 output/vis_ocr/
    python scripts/vis_ocr_det_dataset.py

    # 指定标注文件、数据集根目录、输出目录、数量
    python scripts/vis_ocr_det_dataset.py \
        --label_file datasets/ocr_det_dataset_examples/train.txt \
        --data_dir   datasets/ocr_det_dataset_examples \
        --output_dir output/vis_ocr \
        --num        20

    # 可视化全部，显示窗口而不保存
    python scripts/vis_ocr_det_dataset.py --show --num -1
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ── 配色：有效框绿色，忽略框(###)红色 ──────────────────────────────────────────
COLOR_VALID   = (0, 200, 0)    # BGR 绿
COLOR_IGNORED = (0, 0, 220)    # BGR 红
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.45
FONT_THICK    = 1


def draw_annotations(image: np.ndarray, annotations: list) -> np.ndarray:
    """在图像上绘制所有文本检测框及文字内容。"""
    vis = image.copy()
    for ann in annotations:
        text   = ann["transcription"]
        pts    = np.array(ann["points"], dtype=np.int32)
        color  = COLOR_IGNORED if text == "###" else COLOR_VALID

        # 绘制四边形轮廓
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # 在框左上角写文字（忽略框不写内容，避免污染画面）
        if text != "###":
            x, y = pts[0]
            # 半透明背景块，提升可读性
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
            cv2.rectangle(vis, (x, y - th - 4), (x + tw + 2, y), color, -1)
            cv2.putText(vis, text, (x + 1, y - 2),
                        FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)

    return vis


def parse_label_file(label_file: str):
    """逐行解析标注文件，返回 (img_rel_path, annotations) 列表。"""
    records = []
    with open(label_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                print(f"[WARN] 第 {lineno} 行无制表符分隔，跳过: {line[:50]}")
                continue
            img_path, _, ann_str = line.partition("\t")
            annotations = json.loads(ann_str)
            records.append((img_path, annotations))
    return records


def visualize(label_file: str, data_dir: str, output_dir: str,
              num: int, show: bool):
    records = parse_label_file(label_file)
    if num > 0:
        records = records[:num]

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_rel, annotations in records:
        img_path = os.path.join(data_dir, img_rel)
        if not os.path.exists(img_path):
            print(f"[WARN] 图片不存在，跳过: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] 读取失败，跳过: {img_path}")
            continue

        vis = draw_annotations(image, annotations)

        # 统计信息叠加在左上角
        n_valid   = sum(1 for a in annotations if a["transcription"] != "###")
        n_ignored = len(annotations) - n_valid
        info = f"valid:{n_valid}  ignored:{n_ignored}"
        cv2.putText(vis, info, (8, 22), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        if show:
            cv2.imshow("OCR Det Visualizer", vis)
            key = cv2.waitKey(0)
            if key == ord("q") or key == 27:   # Q / ESC 退出
                break
        else:
            stem = os.path.splitext(os.path.basename(img_rel))[0]
            out_path = os.path.join(output_dir, f"{stem}_vis.jpg")
            cv2.imwrite(out_path, vis)
            print(f"[saved] {out_path}")

    if show:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="可视化 PaddleOCR 检测数据集")
    ap.add_argument("--label_file", default="datasets/ocr_det_dataset_examples/train.txt",
                    help="标注文件路径（默认: datasets/ocr_det_dataset_examples/train.txt）")
    ap.add_argument("--data_dir", default="datasets/ocr_det_dataset_examples",
                    help="数据集根目录，图片路径以此为基准（默认: datasets/ocr_det_dataset_examples）")
    ap.add_argument("--output_dir", default="output/vis_ocr",
                    help="可视化结果输出目录（默认: output/vis_ocr）")
    ap.add_argument("--num", type=int, default=10,
                    help="可视化图片数量，-1 表示全部（默认: 10）")
    ap.add_argument("--show", action="store_true",
                    help="用窗口显示而不保存，按 Q/ESC 退出，任意键下一张")
    args = ap.parse_args()

    # 脚本可从项目根目录或 scripts/ 目录运行
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    def resolve(p):
        if os.path.isabs(p):
            return p
        # 优先从当前工作目录解析，其次从项目根
        if os.path.exists(p):
            return p
        candidate = os.path.join(project_root, p)
        if os.path.exists(candidate):
            return candidate
        return p  # 保留原值，后续报错

    label_file = resolve(args.label_file)
    data_dir   = resolve(args.data_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(project_root, args.output_dir)

    if not os.path.exists(label_file):
        print(f"[ERROR] 标注文件不存在: {label_file}", file=sys.stderr)
        sys.exit(1)

    print(f"标注文件 : {label_file}")
    print(f"数据目录 : {data_dir}")
    if not args.show:
        print(f"输出目录 : {output_dir}")
    print(f"可视化数量: {'全部' if args.num < 0 else args.num}")
    print()

    visualize(label_file, data_dir, output_dir, args.num, args.show)

    if not args.show:
        print(f"\n完成，结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
