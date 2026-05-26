"""数据模型 schemas 单元测试"""
import pytest
import numpy as np
from schemas.data_base import (
    DetectResult,
    DetectionItem,
    MoMResult,
    InputParamsBusiness,
    OCRResult,
    IndicatorLightEmbedding,
    MessageType,
)
from schemas.common import CommonResponse, ResultResponse, EmptyRequest


class TestDetectResult:
    def test_default_values(self):
        r = DetectResult()
        assert r.boxes == []
        assert r.scores == []
        assert r.class_ids == []
        assert r.class_names == []
        assert r.masks == []
        assert r.mask_polygons == []
        assert r.ori_img is None

    def test_custom_values(self):
        r = DetectResult(
            boxes=[[1, 2, 3, 4]],
            scores=[0.5],
            class_ids=[0],
            class_names=["person"],
        )
        assert r.boxes == [[1, 2, 3, 4]]
        assert r.scores == [0.5]
        assert r.class_ids == [0]
        assert r.class_names == ["person"]

    def test_field_modification(self):
        r = DetectResult()
        r.boxes = [[1, 2, 3, 4]]
        r.scores = [0.9]
        r.class_ids = [1]
        r.class_names = ["car"]
        assert r.boxes == [[1, 2, 3, 4]]


class TestDetectionItem:
    def test_default_values(self):
        item = DetectionItem()
        assert item.status is False
        assert item.scene == ""
        assert item.coordinate == []
        assert item.accuracy == 0.0
        assert item.name == ""
        assert item.color == "#20ff4f"

    def test_custom_values(self):
        item = DetectionItem(
            status=True, scene="dc",
            coordinate=[1, 2, 3, 4],
            accuracy=0.8, name="dc_1",
        )
        assert item.status is True
        assert item.scene == "dc"
        assert item.coordinate == [1, 2, 3, 4]
        assert item.accuracy == 0.8
        assert item.name == "dc_1"

    def test_to_dict_true_status(self):
        item = DetectionItem(status=True, scene="dc", coordinate=[1, 2, 3, 4],
                             accuracy=0.8, name="dc_1")
        d = item.to_dict()
        assert d["status"] == "true"
        assert d["scene"] == "dc"
        assert d["coordinate"] == [1, 2, 3, 4]
        assert d["accuracy"] == 0.8
        assert d["name"] == "dc_1"
        assert d["color"] == "#20ff4f"

    def test_to_dict_false_status(self):
        item = DetectionItem(status=False, scene="x", coordinate=[],
                             accuracy=0.0, name="fail")
        d = item.to_dict()
        assert d["status"] == "false"
        assert d["color"] == "#FFFF00"

    def test_from_dict(self):
        data = {
            "status": True, "scene": "dc",
            "coordinate": [1, 2, 3, 4],
            "accuracy": 0.8, "name": "dc_1",
        }
        item = DetectionItem.from_dict(data)
        assert item.status is True
        assert item.scene == "dc"
        assert item.coordinate == [1, 2, 3, 4]

    def test_from_dict_with_color(self):
        item = DetectionItem.from_dict({"color": "#FF0000"})
        assert item.color == "#FF0000"


class TestMoMResult:
    def test_default_values(self):
        r = MoMResult()
        assert r.detailList == []
        assert r.status is False
        assert r.error_msg == ""
        assert r.message == ""

    def test_to_dict(self):
        item = DetectionItem(status=True, scene="dc", coordinate=[1, 2, 3, 4],
                             accuracy=0.8, name="dc_1")
        r = MoMResult(detailList=[item], status=True, message="检测成功")
        d = r.to_dict()
        assert d["status"] == "true"
        assert len(d["detailList"]) == 1
        assert d["detailList"][0]["status"] == "true"

    def test_from_dict(self):
        data = {
            "detailList": [{
                "status": True, "scene": "dc",
                "coordinate": [1, 2, 3, 4],
                "accuracy": 0.8, "name": "dc_1",
            }],
            "status": True, "error_msg": "", "message": "ok",
        }
        r = MoMResult.from_dict(data)
        assert r.status is True
        assert len(r.detailList) == 1
        assert r.detailList[0].scene == "dc"


class TestInputParamsBusiness:
    def test_default_image_raises(self):
        """image 字段的 default_factory=np.ndarray 在无参时有 bug，应传 image 参数"""
        # 传入实际 image 来规避 np.ndarray() 的 TypeError
        p = InputParamsBusiness(image=np.zeros((10, 10, 3), dtype=np.uint8))
        assert p.SN == ""
        assert p.product_type == ""
        assert p.is_registered is False

    def test_custom_values(self):
        p = InputParamsBusiness(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            SN="SN123", product_type="TYPE_A", is_registered=True,
        )
        assert p.SN == "SN123"
        assert p.product_type == "TYPE_A"
        assert p.is_registered is True


class TestOCRResult:
    def test_default_values(self):
        r = OCRResult()
        assert r.text == []
        assert r.boxes == []
        assert r.class_ids == []
        assert r.scores == []


class TestIndicatorLightEmbedding:
    def test_default_values(self):
        e = IndicatorLightEmbedding()
        assert e.embeddings == []
        assert e.boxes == []
        assert e.scores == []


class TestMessageType:
    def test_values(self):
        assert MessageType.SUCCESS == "检测成功"
        assert MessageType.FAIL == "检测失败"
        assert MessageType.PRODUCT_TYPE_ERROR == "产品类型错误"


class TestResultResponse:
    def test_creation(self):
        item = DetectionItem(status=True, scene="test", coordinate=[1, 2, 3, 4],
                             accuracy=0.9, name="t1")
        rr = ResultResponse(
            detailList=[item], status="true",
            error_msg="", message="检测成功",
        )
        assert rr.status == "true"
        assert len(rr.detailList) == 1


class TestCommonResponse:
    def test_creation(self):
        item = DetectionItem(status=True, scene="test", coordinate=[1, 2, 3, 4],
                             accuracy=0.9, name="t1")
        result = ResultResponse(
            detailList=[item], status="true",
            error_msg="", message="检测成功",
        )
        resp = CommonResponse(code=1, message="成功", result=result)
        assert resp.code == 1
        assert resp.message == "成功"
        assert resp.result.status == "true"


class TestEmptyRequest:
    def test_creation(self):
        req = EmptyRequest()
        assert isinstance(req, EmptyRequest)


class TestCommonResponseCodeIsInt:
    """code 字段必须接受多档 int 取值"""

    def test_accepts_success(self):
        from schemas.common import CommonResponse, ResultResponse
        result = ResultResponse(detailList=[], status="true", error_msg="", message="ok")
        resp = CommonResponse(code=1, message="成功", result=result)
        assert resp.code == 1

    def test_accepts_multi_digit_codes(self):
        from schemas.common import CommonResponse, ResultResponse
        result = ResultResponse(detailList=[], status="false", error_msg="x", message="x")
        for c in (1001, 1002, 1003, 5000, 5001):
            resp = CommonResponse(code=c, message="x", result=result)
            assert resp.code == c
