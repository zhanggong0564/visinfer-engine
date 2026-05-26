"""panel_label 业务逻辑单元测试"""
import pytest
from unittest.mock import patch, MagicMock
from schemas.exceptions import ProductNotRegisteredError


@pytest.fixture
def api_instance():
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
    with patch("services.panel_label.business_logic.OCRPipeline"):
        from services.panel_label.business_logic import PanelLabelJudgeApi
        from config import settings
        api = PanelLabelJudgeApi(settings)
        api.w = 1000
        api.h = 1000
        yield api


class TestProductTypeValidation:
    def test_unregistered_product_type_raises(self, api_instance):
        from services.panel_label.panel_label_detect import PanellabelItem
        bogus_result = PanellabelItem()
        with pytest.raises(ProductNotRegisteredError) as exc_info:
            api_instance.business_logic_post_process(bogus_result, "NON_EXISTENT_TYPE_999")
        assert "NON_EXISTENT_TYPE_999" in exc_info.value.error_msg
        assert exc_info.value.context.get("product_type") == "NON_EXISTENT_TYPE_999"
        assert exc_info.value.context.get("scenario") == "panel_label"
