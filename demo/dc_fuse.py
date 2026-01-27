'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:14:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-23 09:34:13
@FilePath     : dc_fuse.py
@Description  :
'''

'''
@Author       : gongzhang4
@Date         : 2025-12-01 07:21:51
@LastEditors  : zhanggong zhanggong1@sungrowpower.com
@LastEditTime : 2025-12-09 02:36:39
@FilePath     : dc_fuse.py
@Description  :
'''

import sys

sys.path.append("..")
from services import DCFuseDetectorAPI
import cv2
from collections import defaultdict
from config import settings
from utils import vision_logger
import json
from services import rotate_points
import numpy as np
from services.data_base import InputParamsBusiness
from services.api import ApiFactory


if __name__ == '__main__':
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision_identification/src/test_image/debug_20251206/1/17617013726191983345470760165376.jpg"
    type_name = "五路有熔丝盒无磁环"
    image = cv2.imread(image_path)
    is_rotate = image.shape[1] > image.shape[0]

    if is_rotate:
        # 向右旋转90度
        print("rotate image")
        image_src = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w, _ = image_src.shape

    # detector = DCFuseDetectorAPI(settings)
    detector = ApiFactory.create_api("dc_fuse")
    input = InputParamsBusiness(image=image, product_type=type_name)
    res = detector.detect(input)
    vision_logger.info(res.to_dict())

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
    cv2.imwrite("dc_fuse_rotate.jpg", image_src)

    # vision_logger.info(json.dumps(res, ensure_ascii=False, indent=2))
