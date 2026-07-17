"""预览服务端可视化（vis_image）。

用 router 同款的 render_detection_overlay 跑一张图，把结果存成可直接打开的 jpg。
不用起服务也能看到响应里 result.vis_image 的真实样子（含蓝色虚线引导框）。

运行（从仓库根目录、在 ppocr 环境）：
    python demo/preview_vis.py <scene> <图片路径> [产品型号] [输出jpg]

例：
    python demo/preview_vis.py panel_label label.jpg TK2 vis_out.jpg
    python demo/preview_vis.py dc_fuse test.jpg 五路有熔丝盒无磁环 vis_out.jpg

scene 可选：dc_fuse / indicator / lap_surf / line_squeeze / panel_label / plate_screw
"""
import base64
import os
import sys

import cv2
import numpy as np

# 从仓库根运行：把 cwd(仓库根) 放进 sys.path，确保能 import services/schemas/config
sys.path.insert(0, os.getcwd())

# scene -> 触发注册的插件模块名（需先 pip install -e 各插件）
PLUGIN_MODULES = {
    "dc_fuse": "vie_plugin_dc_fuse.plugin",
    "indicator": "vie_plugin_indicator_light.plugin",
    "lap_surf": "vie_plugin_lap_surf.plugin",
    "line_squeeze": "vie_plugin_line_squeeze.plugin",
    "panel_label": "vie_plugin_panel_label.plugin",
    "plate_screw": "vie_plugin_plate_screw.plugin",
}


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "panel_label"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
    product_type = sys.argv[3] if len(sys.argv) > 3 else ""
    out_path = sys.argv[4] if len(sys.argv) > 4 else "vis_out.jpg"

    __import__(PLUGIN_MODULES[scene])  # 导入即触发 @scenario_registry.register
    from services.scenario_registry import scenario_registry
    from routers.visualization import render_detection_overlay
    from schemas.data_base import InputParamsBusiness
    from config import settings

    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")

    # 线标(panel_label)专属：standard_result/guideline 生产环境随请求下发，
    # 示例从本地词典取并经 extra 注入（对齐插件 examples/run.py）。
    params = InputParamsBusiness(image=image, product_type=product_type)
    guides = None
    if scene == "panel_label":
        from vie_plugin_panel_label.product_type import PRODUCT_TYPE, PRODUCT_guideline
        guideline = PRODUCT_guideline.get(product_type)
        params.extra = {
            "standard_result": PRODUCT_TYPE.get(product_type),
            "guideline": guideline,
        }
        if guideline:
            guides = [tuple(guideline)]

    detector = scenario_registry.create(scene)
    result = detector.detect(params)
    result_dict = result if isinstance(result, dict) else result.to_dict()

    detail_list = result_dict.get("detailList", [])
    print(f"detailList 共 {len(detail_list)} 项, 顶层 status={result_dict.get('status')}, guides={guides}")

    vis_b64 = render_detection_overlay(
        image,
        detail_list,
        guides=guides,
        max_side=settings.VIS_MAX_SIDE,
        jpeg_quality=settings.VIS_JPEG_QUALITY,
    )
    if not vis_b64:
        raise SystemExit("vis_image 为空（无原图/绘制失败），检查检测结果")

    raw = base64.b64decode(vis_b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    h, w = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    print(f"可视化已存盘: {out_path}  (base64 长度={len(vis_b64)}, 解码尺寸={w}x{h})")


if __name__ == "__main__":
    main()
