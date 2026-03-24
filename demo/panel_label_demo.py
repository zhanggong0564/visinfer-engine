'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:42:41
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-23 12:27:05
@FilePath     : panel_label_demo.py
@Description  :
'''

import sys

sys.path.append("..")
import cv2
from utils import vision_logger
from services import rotate_points
import numpy as np
from services.panel_label import OCRPipeline, PanelLabelJudgeApi
from config import settings
from schemas import InputParamsBusiness
import json
from pathlib import Path

'''
IMG_20260127_150447_087_res.jpg

'''


def visualize_results(image_src, results, dst_path):
    h, w, _ = image_src.shape
    for detail in results.to_dict().get("detailList", []):
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
        status = detail.get("status", "")
        color = (0, 255, 0) if status == "true" else (0, 0, 255)
        cv2.polylines(image_src, points, True, color, 2)
        cv2.putText(image_src, detail.get("name", ""), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.polylines(image_src, np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), True, (0, 255, 0), 2)
    cv2.imwrite(dst_path, image_src)


# for detail in results.to_dict().get("detailList", []):
#     x1, y1, x2, y2, x3, y3, x4, y4 = detail.get("coordinate", [])
#     x1 = int(x1 * w)
#     y1 = int(y1 * h)
#     x2 = int(x2 * w)
#     y2 = int(y2 * h)
#     x3 = int(x3 * w)
#     y3 = int(y3 * h)
#     x4 = int(x4 * w)
#     y4 = int(y4 * h)
#     points = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
#     status = detail.get("status", "")
#     color = (0, 255, 0) if status == "true" else (0, 0, 255)
#     cv2.polylines(image_src, points, True, color, 2)
#     cv2.putText(image_src, detail.get("name", ""), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#     # cv2.polylines(image_src, np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), True, (0, 255, 0), 2)
# cv2.imwrite("panel_label_res.jpg", image_src)


if __name__ == '__main__':
    image_paths = list(Path("./demo/data/panel_label/QF2").glob("*.jpg"))
    positive_num = 0
    total_num = len(image_paths)
    detector = PanelLabelJudgeApi(settings.panel_label)

    for image_path in image_paths:
        # print(image_path)
        # image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision/demo/data/panel_label/QF2/IMG_20260127_150401_214.jpg"
        image_src = cv2.imread(str(image_path))
        # model_path = "./weights/panel_label/best_v1.onnx"
        # orient_model_path = "./weights/panel_label/PP-LCNet_x1_0_textline_ori"
        # confThreshold = 0.4

        # detector = OCRPipeline(model_path, orient_model_path, confThreshold)
        # results = detector.infer(image_src)
        # results.save_img(image_src, "./demo/data/panel_label_result.jpg")

        # for result in results:
        #     # result.print()
        #     print(f"rec_texts: {result['rec_texts']}->orientation: {result['textline_orientation_angles']}")
        input_params = InputParamsBusiness(
            image=image_src,
            product_type="QF2",
        )
        h, w, _ = image_src.shape

        results = detector.detect(input_params)
        dst_path = "./vis/" + image_path.stem + "_res.jpg"

        visualize_results(image_src, results, dst_path)

        # json_str = json.dumps(results.to_dict(), indent=4)
        # print(results.to_dict()["status"])
        if results.to_dict()["status"] == "true":
            positive_num += 1
        else:
            print(f"false dst_path: {dst_path} ,path: {image_path}")
    print(f"positive_num: {positive_num}, total_num: {total_num}, accuracy: {positive_num / total_num}")
