'''
@Author       : gongzhang4
@Date         : 2026-01-29 12:25:07
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-02-07 08:06:00
@FilePath     : business_logic.py
@Description  :
'''

from .tools import check_box_containment
from ..api import detection_factory
from ..base import BusinessLogicBase
from .plate_screw_detect import PlateScrewDetect
from utils import vision_logger
from schemas.data_base import MoMResult, DetectResult, DetectionItem
from schemas.exceptions import ModelInferenceError
from collections import defaultdict
from schemas import MessageType


def select_box(box_info, image_width, image_height):
    """
    针对铁片好的box和螺丝的box进行交叉判断和筛选
    :param results:
    :return:
    """
    all_metal_plate_info = []
    all_metal_screw_info = []
    for label in box_info.keys():
        new_box_list = []
        for one_box in box_info[label]:
            new_box_info = one_box + [label]
            new_box_list.append(new_box_info)
        if "metal_plate" in label:  #### 先筛选出铁片
            all_metal_plate_info = all_metal_plate_info + new_box_list
        if "metal_screw" in label:  #### 再筛选出螺丝
            all_metal_screw_info = all_metal_screw_info + new_box_list

    ######  判断铁片和螺丝间的包含关系
    all_result_after_contain = []
    for one_plate in all_metal_plate_info:
        one_plate_box = one_plate[0]
        one_plate_contain_info = {}
        one_plate_contain_info["screw"] = []
        one_plate_contain_info["plate"] = [one_plate]
        ####  遍历所有螺丝位置进行判断
        for screw_id, one_screw in enumerate(all_metal_screw_info):
            one_screw_box = one_screw[0]
            contain_res = check_box_containment(
                one_plate_box, one_screw_box, img_width=image_width, img_height=image_height
            )
            if contain_res == 1:  ####  说明 这个螺丝再这个铁片里面
                one_plate_contain_info["screw"].append(one_screw)

        all_result_after_contain.append(one_plate_contain_info)
    return all_result_after_contain


@detection_factory.register("plate_screw")
class PlateScrewJudgeApi(BusinessLogicBase):
    def __init__(self, settings):
        super().__init__(settings)

    def _initialize_model(self, settings):
        try:
            self.detector = PlateScrewDetect(settings.plate_screw.model_path, settings.plate_screw.confThreshold)
        except Exception as e:
            vision_logger.error(f"加载模型失败: {e}")
            raise ModelInferenceError(
                "plate_screw 模型加载失败",
                scenario="plate_screw",
                original_error=e,
            )

    def business_post_process(self, ctx):
        result = ctx.raw_result
        res = defaultdict(list)
        for box, cls, score, name in zip(result.boxes, result.class_ids, result.scores, result.class_names):
            res[name].append([box, score])
        results_contain = select_box(res, ctx.w, ctx.h)
        ctx.result = self._judge_result(results_contain, True)

    def _judge_result(self, results_contain, return_box=True):
        judge_result_info = MoMResult()
        #####  先判断 场景七--铁片 是否存在缺失
        judge_result_info.status = True

        try:
            for one_result in results_contain:
                for key, value in one_result.items():
                    for box_info in value:
                        box = box_info[0]
                        conf = box_info[1]
                        label = box_info[2]
                        status = False if 'no' in label else True
                        if not status:
                            judge_result_info.status = False
                        temp = DetectionItem(
                            status=status,
                            scene=label,
                            coordinate=[box[0], box[1], box[2], box[3]],
                            accuracy=conf,
                        )
                        if return_box:
                            judge_result_info.detailList.append(temp)
            judge_result_info.message = MessageType.SUCCESS.value
        except Exception as e:
            judge_result_info.status = False
            judge_result_info.error_msg = str(e)
        return judge_result_info
