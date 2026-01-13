'''
@Author       : gongzhang4
@Date         : 2026-01-13 05:18:18
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-13 07:30:24
@FilePath     : business_logic.py
@Description  :
'''

from .yolo import yolo11ONNX
from .tools import check_box_containment


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


class PlateScrewJudgeApi:
    def __init__(self, model_path, conf_threshold=0.5):
        self.detector = self.load_model(model_path, conf_threshold)

    def load_model(self, model_path, conf_threshold):
        return yolo11ONNX(model_path, nc=4, confThreshold=conf_threshold)

    def detect(self, im):
        outputs = self.detector.infer(im)
        self.image_width, self.image_height = im.shape[1], im.shape[0]
        # vis = vis_box_mask(im, outputs)
        # cv2.imwrite("vis.jpg", vis)
        results = self.postprocess(outputs)
        return results

    def postprocess(self, outputs):
        results_contain = select_box(outputs, self.image_width, self.image_height)
        res = self._judge_result(results_contain, True)
        return res

        # return results

    def _judge_result(self, results_contain, return_box=True):
        judge_result_info = {}
        #####  先判断 场景七--铁片 是否存在缺失
        judge_result_info["detailList"] = []
        judge_result_info['status'] = True
        judge_result_info['error_msg'] = ""

        try:
            for one_result in results_contain:
                for key, value in one_result.items():
                    for box_info in value:
                        box = box_info[0]
                        conf = box_info[1]
                        label = box_info[2]
                        status = False if 'no' in label else True
                        if not status:
                            judge_result_info['status'] = False
                        temp = {
                            "coordinate": [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]],
                            "scene": label,
                            "accuracy": conf,
                            "status": "true" if status else "false",
                        }
                        if return_box:
                            judge_result_info["detailList"].append(temp)
            judge_result_info['message'] = "检测成功"
        except Exception as e:
            judge_result_info['status'] = False
            judge_result_info['error_msg'] = str(e)
        judge_result_info['status'] = "true" if judge_result_info['status'] else "false"
        return judge_result_info
