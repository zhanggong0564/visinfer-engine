'''
@Author       : gongzhang4
@Date         : 2026-01-07 07:14:19
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 07:31:20
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


if __name__ == '__main__':
    image_path = "/data/zhanggong/workspace/project/move_vsion/mobile_vision_identification/src/test_image/debug_20251206/1/17617013726191983345470760165376.jpg"
    type_name = "五路有熔丝盒无磁环"
    image = cv2.imread(image_path)
    detector = DCFuseDetectorAPI(model_path=settings.dc_fuse.model_path, conf_threshold=settings.dc_fuse.confThreshold)
    res = detector.detect(image, type_name)
    vision_logger.info(json.dumps(res, ensure_ascii=False, indent=2))
