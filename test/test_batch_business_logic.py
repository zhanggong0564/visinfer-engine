import numpy as np
import pytest

from services.base import BatchBusinessLogicBase


class _BatchBusiness(BatchBusinessLogicBase):
    def _initialize_model(self, settings):
        self.events = []

    def validate_batch(self, images, request_params):
        self.events.append(("validate", len(images), request_params))

    def inspect_batch(self, images, request_params):
        self.events.append(("inspect", [name for name, _ in images]))
        return {"status": "PASS"}


def test_batch_business_logic_runs_validation_before_inspection():
    business = _BatchBusiness(settings=object())
    images = [("sample-1.jpg", np.zeros((2, 2, 3), dtype=np.uint8))]

    result = business.inspect(images, {"selected": "plug"})

    assert result == {"status": "PASS"}
    assert business.events == [
        ("validate", 1, {"selected": "plug"}),
        ("inspect", ["sample-1.jpg"]),
    ]


def test_batch_business_logic_rejects_single_image_detect_entry():
    business = _BatchBusiness(settings=object())

    with pytest.raises(RuntimeError, match="应调用 inspect"):
        business.detect(object())
