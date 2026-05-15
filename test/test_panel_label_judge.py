"""PanelLabelJudgeApi 面板标签业务逻辑单元测试"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from services.panel_label.business_logic import (
    PanelLabelJudgeApi,
    PanelInfo,
    ErrorType,
    PanellabelItem,
)
from schemas.data_base import DetectionItem


@pytest.fixture
def mock_product_type():
    return {
        "TYPE1": ["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"],
    }


@pytest.fixture
def mock_guideline():
    return {
        "TYPE1": (0.1, 0.1, 0.8, 0.8),  # x, y, w, h 归一化
    }


@pytest.fixture
def judge():
    with patch.object(
        PanelLabelJudgeApi, "_initialize_model", return_value=None
    ):
        judge = PanelLabelJudgeApi(MagicMock())
        judge.detector = MagicMock()
        judge.w = 1000
        judge.h = 800
        return judge


class TestPanelLabelAnalyze:
    def test_perfect_match(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            observed = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"],
                confidence=[0.95, 0.88, 0.72],
            )
            result = judge.analyze(observed, "TYPE1")
            assert result.result is True
            assert result.message == ErrorType.OK.value

    def test_count_mismatch(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            observed = PanellabelItem(
                Points=[[10, 20, 30, 40, 50, 60, 70, 80]],
                index=[0],
                class_id=[0],
                texts=["LINE1/aaa"],  # 只有1个，标准有3个
                confidence=[0.95],
            )
            result = judge.analyze(observed, "TYPE1")
            assert result.result is False
            assert result.message == ErrorType.MISSING.value

    def test_content_mismatch(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            observed = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=["LINE1/xxx", "WRONG/yyy", "LINE3/zzz"],
                confidence=[0.9, 0.8, 0.7],
            )
            result = judge.analyze(observed, "TYPE1")
            assert result.result is False
            assert result.message == ErrorType.MISMATCH.value
            assert 1 in result.error_indexs

    def test_case_insensitive_match(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            observed = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=["line1/xxx", "Line2/yyy", "LINE3/zzz"],  # 大小写不同
                confidence=[0.9, 0.8, 0.7],
            )
            result = judge.analyze(observed, "TYPE1")
            assert result.result is True
            assert result.message == ErrorType.OK.value

    def test_only_compare_before_slash(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            observed = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=[
                    "LINE1/different_suffix",
                    "LINE2/other",
                    "LINE3/whatever",
                ],
                confidence=[0.9, 0.8, 0.7],
            )
            result = judge.analyze(observed, "TYPE1")
            assert result.result is True


class TestGuidelineFilter:
    def test_points_inside_rect_kept(self, judge, mock_guideline):
        judge.w = 1000
        judge.h = 800
        with patch(
            "services.panel_label.business_logic.PRODUCT_guideline",
            mock_guideline,
        ):
            results = PanellabelItem(
                Points=[
                    [200, 200, 300, 200, 300, 300, 200, 300],  # 在 (100,80,800,640) 内
                ],
                index=[0],
                class_id=[0],
                texts=["KEEP"],
                confidence=[0.9],
            )
            filtered = judge.guideline_filter(results, "TYPE1")
            assert len(filtered.Points) == 1
            assert filtered.texts == ["KEEP"]

    def test_points_outside_rect_excluded(self, judge, mock_guideline):
        judge.w = 1000
        judge.h = 800
        with patch(
            "services.panel_label.business_logic.PRODUCT_guideline",
            mock_guideline,
        ):
            results = PanellabelItem(
                Points=[
                    [950, 750, 960, 750, 960, 760, 950, 760],  # 在 roi 外
                ],
                index=[0],
                class_id=[0],
                texts=["DISCARD"],
                confidence=[0.9],
            )
            filtered = judge.guideline_filter(results, "TYPE1")
            assert len(filtered.Points) == 0


class TestBusinessLogicPostProcess:
    def test_success_scenario(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            results = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"],
                confidence=[0.95, 0.88, 0.72],
            )
            mom = judge.business_logic_post_process(results, "TYPE1")
            assert mom.status is True
            assert mom.message == ErrorType.OK.value
            assert len(mom.detailList) == 3
            for item in mom.detailList:
                assert item.status is True

    def test_mismatch_items_have_false_status(self, judge, mock_product_type):
        with patch(
            "services.panel_label.business_logic.PRODUCT_TYPE", mock_product_type
        ):
            results = PanellabelItem(
                Points=[
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [100, 200, 300, 400, 500, 600, 700, 800],
                    [10, 200, 300, 200, 300, 400, 10, 400],
                ],
                index=[0, 1, 2],
                class_id=[0, 0, 0],
                texts=["LINE1/xxx", "WRONG_DATA", "LINE3/zzz"],
                confidence=[0.95, 0.88, 0.72],
            )
            mom = judge.business_logic_post_process(results, "TYPE1")
            assert mom.status is False
            # 第1个和第3个正确，第2个错误
            assert mom.detailList[0].status is True
            assert mom.detailList[1].status is False
            assert mom.detailList[2].status is True


class TestPanelInfo:
    def test_default_values(self):
        info = PanelInfo()
        assert info.result is False
        assert info.message == ErrorType.UNKNOWN.value
        assert info.standard_result == []
        assert info.observed_result == []
        assert info.error_indexs == []

    def test_custom_values(self):
        info = PanelInfo(
            result=True,
            product_type="T1",
            standard_result=["A", "B"],
            observed_result=["a", "b"],
            observed_result_points=[[1, 2, 3, 4, 5, 6, 7, 8]],
            message=ErrorType.OK.value,
        )
        assert info.result is True
        assert info.product_type == "T1"
        assert info.standard_result == ["A", "B"]
