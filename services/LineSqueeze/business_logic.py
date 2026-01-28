'''
@Author       : gongzhang4
@Date         : 2026-01-17 06:45:33
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-28 07:38:33
@FilePath     : business_logic.py
@Description  :
'''

import numpy as np
from ..utils import sort_boxes
from typing import List, Tuple
from ..data_base import OCRResult, MoMResult, DetectionItem
from ..api import detection_factory
from ..base import BusinessLogicBase
from .yolo import LineSqueezePipeline
from utils import vision_logger


class VerifyLineSequenceUtils(object):
    def __init__(self, nums: int, verify_dc: bool = False, verify_fu: bool = False):
        self.nums = nums
        self.verify_dc = verify_dc
        self.verify_fu = verify_fu

    def __call__(
        self,
        dc_infos: List[str],
        fu_infos: List[str],
        sorted_dc_boxes: List[List[int]],
        sorted_fu_boxes: List[List[int]],
    ) -> Tuple[bool, List[DetectionItem]]:
        res_infos = MoMResult().detailList
        # 判定dc_infos fu_infos 不为空时长度= nums
        if (len(dc_infos) != 0 and len(dc_infos) != self.nums) or (len(fu_infos) != 0 and len(fu_infos) != self.nums):
            return False, res_infos

        if self.verify_dc:
            res_info = self.verify_line_sequence(dc_infos, self.nums)
            if len(sorted_dc_boxes) == 0:
                sorted_dc_boxes = np.array([[] for _ in range(len(res_info) + 1)])
            for res, box in zip(res_info, sorted_dc_boxes):
                res_infos.append(
                    DetectionItem(
                        status=res,
                        scene="dc",
                        coordinate=box[:4],
                        accuracy=float(box[4]) if len(box) != 0 else "",
                    )
                )

        if self.verify_fu:
            res_info = self.verify_line_sequence(fu_infos, self.nums)
            if len(sorted_fu_boxes) == 0:
                sorted_fu_boxes = np.array([[] for _ in range(len(res_info) + 1)])
            for res, box in zip(res_info, sorted_fu_boxes):
                res_infos.append(
                    DetectionItem(
                        status=res,
                        scene="fu",
                        coordinate=box[:4],
                        accuracy=float(box[4]) if len(box) != 0 else "",
                    )
                )
        if self.verify_dc and self.verify_fu:
            return (
                all([res.status for res in res_infos if res.scene == 'dc'])
                and all([res.status for res in res_infos if res.scene == 'fu']),
                res_infos,
            )
        elif self.verify_dc:
            return all([res.status for res in res_infos if res.scene == 'dc']), res_infos
        elif self.verify_fu:
            return all([res.status for res in res_infos if res.scene == 'fu']), res_infos
        else:
            return True, res_infos

    def verify_line_sequence(self, infos: List[str], nums: int) -> bool:
        """
        验证线序是否正确
        """
        res_infos = [False for _ in range(nums)]
        # if len(infos) != nums:
        #     return False
        try:
            if len(infos) != nums:
                for info in infos:
                    info = int(info)
                    res_infos[info - 1] = True
            else:
                for i in range(nums):
                    info = int(infos[i])
                    if (info - 1) >= nums:
                        continue
                    res_infos[info - 1] = (i + 1) == info
        except ValueError:
            print(f"verify_line_sequence error, infos: {infos}")
            return res_infos
        return res_infos


# ProductType = {
#     "五路有熔丝盒有磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
#     "五路有熔丝盒无磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
#     "六路有熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True, verify_fu=True),
#     "六路无熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True),
#     "七路无熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True),
#     "七路有熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True, verify_fu=True),
# }
# VISUAL_SIMILAR_MAP = {
#     's': ['5'],
#     'S': ['5'],
#     'l': [
#         '1',
#     ],
#     'i': ['1'],
#     'I': ['1'],
#     'O': ['0'],
#     'o': ['0'],
#     'b': ['6'],
#     'q': ['9'],
#     'T': ['1'],
#     't': ['1'],
#     'Z': ['2'],
#     'a': ['2'],
#     'A': ['4'],
#     '+': ['3'],
#     "G": ['5'],
#     "B": ['5'],
#     # 可以添加更多容易混淆的字符对
# }


@detection_factory.register("LineSqueeze")
class LineSqueezeDetectApi(BusinessLogicBase):
    ProductType = {
        "五路有熔丝盒有磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
        "五路有熔丝盒无磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
        "六路有熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True, verify_fu=True),
        "六路无熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True),
        "七路无熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True),
        "七路有熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True, verify_fu=True),
    }
    VISUAL_SIMILAR_MAP = {
        's': ['5'],
        'S': ['5'],
        'l': [
            '1',
        ],
        'i': ['1'],
        'I': ['1'],
        'O': ['0'],
        'o': ['0'],
        'b': ['6'],
        'q': ['9'],
        'T': ['1'],
        't': ['1'],
        'Z': ['2'],
        'a': ['2'],
        'A': ['4'],
        '+': ['3'],
        "G": ['5'],
        "B": ['5'],
        # 可以添加更多容易混淆的字符对
    }

    def __init__(self, settings) -> None:
        super().__init__(settings)

    def _initialize_model(self, settings):
        try:
            self.detector = LineSqueezePipeline(
                det_model_path=settings.line_squeeze.ModelPath.det_model_path,
                ocr_model_path=settings.line_squeeze.ModelPath.ocr_model_dir,
                confThreshold=settings.line_squeeze.ConfThreshold.det,
            )
        except Exception as e:
            vision_logger.error(f"加载模型失败: {e}")
            raise e

    def business_logic_post_process(self, result: OCRResult, product_type: str) -> MoMResult:
        """业务逻辑后处理"""
        if product_type not in self.ProductType:
            return MoMResult(
                status=False, error_msg=f"product_type {product_type} not in ProductType", message="检测失败"
            )
        # 对result中的boxes进行排序,text,class_ids同时也排序

        _, sorted_indices = sort_boxes(result.boxes)
        dc_boxes = [result.boxes[i] + [result.scores[i]] for i in sorted_indices if result.class_ids[i] == 1]
        fu_boxes = [result.boxes[i] + [result.scores[i]] for i in sorted_indices if result.class_ids[i] == 0]

        dc_res = [result.text[i][2] for i in sorted_indices if result.class_ids[i] == 1]
        fu_res = [result.text[i][2] for i in sorted_indices if result.class_ids[i] == 0]
        dc_res = self.check_infos(dc_res)
        fu_res = self.check_infos(fu_res)
        res, infos = self.ProductType[product_type](dc_res, fu_res, dc_boxes, fu_boxes)

        return MoMResult(status=res, detailList=infos, message="检测成功")

    def check_infos(self, infos: List[str]) -> List[str]:
        """
        检查线序识别结果是否正确
        """
        corrected = []
        valid_info = ['1', '2', '3', '4', '5', '6', '7']
        for char in infos:
            if char in valid_info:
                corrected.append(char)
            elif char in self.VISUAL_SIMILAR_MAP and self.VISUAL_SIMILAR_MAP[char][0] in valid_info:
                corrected.append(self.VISUAL_SIMILAR_MAP[char][0])
            else:
                corrected.append(char)
        return corrected


# class LineSqueezeRecognition:
#     def __init__(self, ROIDet_OnnxPath: str, OCR_model_dir: str) -> None:
#         self.roi_det = RoiDet(
#             ROIDet_OnnxPath, 2, input_model_shape=(1, 3, 1280, 1280), providers=['CUDAExecutionProvider']
#         )
#         self.ocr = TextRecognition(model_dir=OCR_model_dir, model_name='en_PP-OCRv5_mobile_rec')
#         # self.ocr = PaddleOCR(
#         #     use_doc_orientation_classify=False,
#         #     use_doc_unwarping=False,
#         #     use_textline_orientation=False,
#         #     device='gpu',
#         #     lang='en',
#         #     ##识别模型路径
#         #     text_recognition_model_dir=OCR_model_dir,
#         #     # #检测模型路径
#         #     # det_model_dir=OCR_model_dir,
#         # )
#         self.classes2names = {0: "fu_line", 1: "dc_line"}

#     ##判断线序是否正确
#     def verify_line_sequence(self, image: np.ndarray, types: str) -> Dict[str, Dict[str, str]]:
#         '''
#         input:np.array,shape=(h,w,c)
#         output:
#         [
#         {
#             "coordinate":[
#               0.2296152114868164,
#               0.12855976819992065,
#               0.2626335322856903,
#               0.17189684510231018
#               ]#线序的坐标信息,
#             "status": True,False,
#             "scene": "dc" #类别信息,
#             "accuracy": 0.95 #识别准确率,
#         },
#         ]

#         '''
#         if types not in ProductType:
#             return {"status": False, "error": f"types {types} not in ProductType"}
#         # import time

#         # start = time.time()
#         results = self.roi_det.infer(image)

#         # image_vis = vis_box_mask(image.copy(), results)
#         # cv2.imwrite('image_vis.jpg', image_vis)
#         # end = time.time()
#         # print(f'roi_det cost time: {end - start} s')
#         classes = results['cls']
#         score = results['score']
#         rect = np.concatenate((np.array(results['rect']), np.array(score).reshape(-1, 1)), axis=1)
#         # 使用列表推导式根据类别将box分配到不同列表
#         dc_boxes = [box for cls, box in zip(classes, rect) if cls == 1]
#         fu_boxes = [box for cls, box in zip(classes, rect) if cls == 0]
#         sorted_dc_boxes = sort_boxes(dc_boxes)
#         sorted_fu_boxes = sort_boxes(fu_boxes)
#         # dc_rois = [
#         #     image[int(dc_box[1]) + 20 : int(dc_box[3]) - 20, int(dc_box[0]) - 10 : int(dc_box[2]) + 10]
#         #     for dc_box in sorted_dc_boxes
#         # ]
#         # fu_rois = [
#         #     image[int(fu_box[1]) + 20 : int(fu_box[3]) - 20, int(fu_box[0]) - 10 : int(fu_box[2]) + 10]
#         #     for fu_box in sorted_fu_boxes
#         # ]
#         dc_rois = [
#             image[int(dc_box[1]) + 10 : int(dc_box[3]) - 10, int(dc_box[0]) : int(dc_box[2])]
#             for dc_box in sorted_dc_boxes
#         ]
#         fu_rois = [
#             image[int(fu_box[1]) + 10 : int(fu_box[3]) - 10, int(fu_box[0]) : int(fu_box[2])]
#             for fu_box in sorted_fu_boxes
#         ]
#         # dc_rois = [roi[int(roi.shape[0] * 0.2) : int(roi.shape[0] * 0.8), :] for roi in dc_rois]
#         # fu_rois = [roi[int(roi.shape[0] * 0.2) : int(roi.shape[0] * 0.8), :] for roi in fu_rois]
#         ##可视化fu/dc
#         # for i, roi in enumerate(dc_rois):
#         #     cv2.imwrite(f'dc_{i}.jpg', roi)
#         # for i, roi in enumerate(fu_rois):
#         #     cv2.imwrite(f'fu_{i}.jpg', roi)
#         dc_res = [res['rec_text'][2] for res in self.ocr.predict(input=dc_rois) if len(res['rec_text']) > 2]
#         fu_res = [res['rec_text'][2] for res in self.ocr.predict(input=fu_rois) if len(res['rec_text']) > 2]

#         # print(f'dc_res: {dc_res}')
#         # print(f'fu_res: {fu_res}')

#         dc_res = self.check_infos(dc_res)
#         fu_res = self.check_infos(fu_res)
#         # print(f'dc_res_after: {dc_res}')
#         # print(f'fu_res_after: {fu_res}')
#         norm_dc_boxes = []
#         for i, box in enumerate(sorted_dc_boxes):
#             box[:4] = box[:4] / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
#             norm_dc_boxes.append(box)
#         norm_fu_boxes = []
#         for i, box in enumerate(sorted_fu_boxes):
#             box[:4] = box[:4] / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
#             norm_fu_boxes.append(box)

#         res, infos = ProductType[types](dc_res, fu_res, norm_dc_boxes, norm_fu_boxes)

#         # for i, box in enumerate(sorted_dc_boxes + sorted_fu_boxes):
#         #     infos[i]['coordinate'] = box
#         result = {"status": res, "detailList": infos}

#         return result

#     def check_infos(self, infos: List[str]) -> List[str]:
#         """
#         检查线序识别结果是否正确
#         """
#         corrected = []
#         valid_info = ['1', '2', '3', '4', '5', '6', '7']
#         for char in infos:
#             if char in valid_info:
#                 corrected.append(char)
#             elif char in VISUAL_SIMILAR_MAP and VISUAL_SIMILAR_MAP[char][0] in valid_info:
#                 corrected.append(VISUAL_SIMILAR_MAP[char][0])
#             else:
#                 corrected.append(char)
#         return corrected
