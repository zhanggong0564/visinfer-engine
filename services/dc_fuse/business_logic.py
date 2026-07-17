"""直流熔丝检测业务逻辑（服务内场景形态，适配新框架模板方法基类）。

判定规则（ResultJudge）原样保留自旧框架；后处理用新模板钩子 business_post_process(ctx)：
读 ctx.raw_result → 写 ctx.result（MoMResult），坐标输出像素值，归一化交给基类 normalize_hook。
模型路径走场景内 config.py（旧框架取 settings.dc_fuse.*，新框架 Settings 不含此项）。
"""

from collections import defaultdict

from services.scenario_registry import scenario_registry
from services.base import BusinessLogicBase
from services.inference import (
    OnnxRuntimeOptions,
    RunnerSpec,
    create_inference_runner,
)
from schemas.data_base import MoMResult, DetectionItem, MessageType
from schemas.exceptions import ProductNotRegisteredError, ModelInferenceError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .config import DcFuseConfig
from .dc_fuse_detect import DCFuseDetector


class ResultJudge:
    def __init__(
        self,
        ways=5,
        is_detect_metal_piece=True,
        is_detect_upper_screw=False,
        is_detect_nut=False,
        is_detectscrew=True,
        is_small_screw=False,
        metal_piece_num=4,
    ):
        self.ways = ways
        self.is_detect_metal_piece = is_detect_metal_piece
        self.is_detect_upper_screw = is_detect_upper_screw
        self.is_detect_nut = is_detect_nut
        self.is_detectscrew = is_detectscrew
        self.is_small_screw = is_small_screw
        self.metal_piece_num2 = metal_piece_num

    def __call__(self, det_info):
        screw = det_info.get("screw_1", [])
        nut = det_info.get("nut_2", [])
        brass_plate = det_info.get("brass_plate_6", [])
        metal_piece = det_info.get("metal_piece_4", [])
        no_screw = det_info.get("no_screw_1", [])
        upper_screw = det_info.get("upper_crossbeam_screw_9", [])
        lower_screw = det_info.get("lower_crossbeam_screw_10", [])
        no_upper_screw = det_info.get("no_upper_crossbeam_screw_9", [])
        no_lower_screw = det_info.get("no_lower_crossbeam_screw_10", [])
        small_screw = det_info.get("small_screw_8", [])
        res = {
            "screw": True,
            "nut": True,
            "metal_piece": True,
            "upper_screw": True,
            "lower_screw": True,
            "brass_plate": True,
            "small_screw": True,
        }
        if self.is_detectscrew:
            if (len(screw) != self.ways * 2) and (len(no_screw) > 0):
                res["screw"] = False
        if self.is_small_screw:
            if len(small_screw) != self.ways:
                res["small_screw"] = False
        if self.is_detect_nut:
            if len(nut) != self.ways * 2:
                res["nut"] = False
        if self.is_detect_metal_piece:
            if self.metal_piece_num2 == 2:
                if len(metal_piece) != 2:
                    res["metal_piece"] = False
            else:
                if not (len(metal_piece) == 4 or len(metal_piece) == 6):
                    res["metal_piece"] = False
        if self.is_detect_upper_screw:
            if (len(upper_screw) != 2 or len(lower_screw) != 2) and (
                len(no_upper_screw) > 0 or len(no_lower_screw) > 0
            ):
                res["upper_screw"] = False
        if len(brass_plate) != self.ways:
            res["brass_plate"] = False
        return {k: v for k, v in res.items() if self._is_detection_enabled(k)}

    def _is_detection_enabled(self, key: str) -> bool:
        """检查指定检测项是否启用"""
        detection_map = {
            "screw": self.is_detectscrew,
            "nut": self.is_detect_nut,
            "metal_piece": self.is_detect_metal_piece,
            "upper_screw": self.is_detect_upper_screw,
            "lower_screw": self.is_detect_upper_screw,
            "brass_plate": True,
            "small_screw": self.is_small_screw,
        }
        return detection_map.get(key, False)


@scenario_registry.register("dc_fuse")
class DCFuseDetectorAPI(BusinessLogicBase):
    SUPPORTED_TYPES = {
        "五路有熔丝盒有磁环": ResultJudge(
            ways=5, is_detectscrew=True, is_small_screw=True, is_detect_metal_piece=True, is_detect_upper_screw=True
        ),
        "五路有熔丝盒无磁环": ResultJudge(ways=5, is_detectscrew=True, is_detect_nut=True, is_detect_metal_piece=True),
        "六路无熔丝盒无磁环": ResultJudge(ways=6, is_detectscrew=False, is_detect_metal_piece=True, is_detect_nut=True),
        "六路有熔丝盒无磁环": ResultJudge(ways=6, is_detectscrew=True, is_detect_nut=True, is_detect_metal_piece=True),
        "七路无熔丝盒无磁环": ResultJudge(ways=7, is_detectscrew=False, is_detect_nut=True),
        "七路有熔丝盒无磁环": ResultJudge(
            ways=7, is_detect_metal_piece=True, is_detectscrew=True, is_detect_nut=True, metal_piece_num=2
        ),
    }

    # 判定项 -> 该项对应的检测标签（含 no_ 前缀），用于回填 detailList，无每请求状态故置类属性
    label_mapping = {
        "screw": ["screw_1", "no_screw_1"],
        "nut": ["nut_2"],
        "small_screw": ["small_screw_8", "no_small_screw_8"],
        "brass_plate": ["brass_plate_6"],
        "metal_piece": ["metal_piece_4"],
        "upper_screw": ["upper_crossbeam_screw_9", "no_upper_crossbeam_screw_9"],
        "lower_screw": ["lower_crossbeam_screw_10", "no_lower_crossbeam_screw_10"],
    }

    def _initialize_model(self, settings):
        cfg = DcFuseConfig()
        runner = None
        try:
            runner = create_inference_runner(
                RunnerSpec(
                    scenario="dc_fuse",
                    onnx_path=cfg.model_path,
                ),
                OnnxRuntimeOptions.from_settings(settings),
            )
            self.detector = DCFuseDetector(
                cfg.confThreshold,
                runner=runner,
            )
        except Exception as e:
            if runner is not None:
                try:
                    runner.close()
                except Exception as close_error:
                    vision_logger.warning(
                        f"dc_fuse 初始化回滚清理失败: {close_error}"
                    )
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError("dc_fuse 模型加载失败", scenario="dc_fuse", original_error=e)

    def business_post_process(self, ctx: InferenceContext) -> None:
        product_type = ctx.product_type
        if product_type not in self.SUPPORTED_TYPES:
            raise ProductNotRegisteredError(
                f"产品型号 '{product_type}' 未在 dc_fuse SUPPORTED_TYPES 中注册",
                product_type=product_type,
                scenario="dc_fuse",
            )
        result = ctx.raw_result  # DetectResult
        result_judge = self.SUPPORTED_TYPES[product_type]
        det_info = defaultdict(list)
        for bbox, score, name in zip(result.boxes, result.scores, result.class_names):
            det_info[name].append({"bbox": bbox, "score": score})
        judge_result = result_judge(det_info)
        # 坐标输出像素 xyxy，归一化由基类 normalize_hook 统一处理（NORMALIZE 默认 True）
        mom_result = MoMResult(status=True, message=MessageType.SUCCESS.value)
        for label, is_pass in judge_result.items():
            if not is_pass:
                mom_result.status = False
            for sub_label in self.label_mapping.get(label, []):
                for det in det_info.get(sub_label, []):
                    mom_result.detailList.append(
                        DetectionItem(status=is_pass, scene=sub_label,
                                      coordinate=det["bbox"], accuracy=det["score"])
                    )
        ctx.result = mom_result
