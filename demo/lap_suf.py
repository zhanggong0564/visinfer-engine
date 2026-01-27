'''
@Author       : gongzhang4
@Date         : 2026-01-08 01:55:25
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-27 09:55:55
@FilePath     : lap_suf.py
@Description  :
'''

import sys

sys.path.append("..")

import cv2
import numpy as np
from config import settings
from utils import vision_logger
from services import detection_factory
from services.data_base import InputParamsBusiness


if __name__ == "__main__":
    image_path = r"/data/zhanggong/workspace/project/move_vsion/mobile_vision/test.jpg"
    image_src = cv2.imread(image_path)
    h, w, _ = image_src.shape
    is_rotate = w < h
    if is_rotate:
        # 向左旋转90度
        print("rotate image")
        image = cv2.rotate(image_src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    infer = detection_factory.get_scenarios("lap_surf")
    input = InputParamsBusiness(image=image)
    # infer = LapSurfJudgeApi(settings.lap_surf.model_path, conf_threshold=settings.lap_surf.confThreshold)

    # bboxes, scores, labels, masks = infer(image)
    res = infer.detect(input)
    print(res)

    def rotate_points(res, src_w, src_h):
        w = src_h
        h = src_w

        detailList = res.get("detailList", [])
        for detail in detailList:
            # 归一化坐标还原,并限制wh
            x1, y1, x2, y2, x3, y3, x4, y4 = detail.get("coordinate", [])
            x1, y1, x2, y2, x3, y3, x4, y4 = (
                min(w, max(0, int(x1 * w))),
                min(h, max(0, int(y1 * h))),
                min(w, max(0, int(x2 * w))),
                min(h, max(0, int(y2 * h))),
                min(w, max(0, int(x3 * w))),
                min(h, max(0, int(y3 * h))),
                min(w, max(0, int(x4 * w))),
                min(h, max(0, int(y4 * h))),
            )

            x2 = x3
            y2 = y3

            x_1 = h - y2
            y_1 = x1

            x_2 = h - y1
            y_2 = x2

            x_3 = x_2
            y_3 = y_1

            x_4 = x_1
            y_4 = y_2
            detail["coordinate"] = [
                x_1 / src_w,
                y_1 / src_h,
                x_3 / src_w,
                y_3 / src_h,
                x_2 / src_w,
                y_2 / src_h,
                x_4 / src_w,
                y_4 / src_h,
            ]
        return res

    res_new = rotate_points(res.to_dict(), w, h)
    for detail in res_new.get("detailList", []):
        x1, y1, x2, y2, x3, y3, x4, y4 = detail.get("coordinate", [])
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        x3 = int(x3 * w)
        y3 = int(y3 * h)
        x4 = int(x4 * w)
        y4 = int(y4 * h)

        points = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
        cv2.polylines(image_src, points, True, (0, 255, 0), 2)
        # cv2.polylines(image_src, np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), True, (0, 255, 0), 2)
    cv2.imwrite("lap_surf_rotate.jpg", image_src)

    # vision_logger.info(json.dumps(res, ensure_ascii=False, indent=4))
