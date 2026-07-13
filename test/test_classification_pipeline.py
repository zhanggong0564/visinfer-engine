from unittest.mock import Mock

import numpy as np
import pytest

from services.base import BaseClassificationPipeline, ClassificationResult, TensorInfo


class FakeRunner:
    input_infos = (TensorInfo("images", (None, 3, 32, 32), "tensor(float)"),)

    def __init__(self, outputs):
        self.run = Mock(return_value=outputs)


class EmptyMetadataRunner:
    input_infos = ()


class StubClassifier(BaseClassificationPipeline):
    def preprocess(self, images):
        return np.stack(images)


def image():
    return np.zeros((3, 32, 32), dtype=np.float32)


def two_image_model(output):
    return StubClassifier(FakeRunner(outputs=[output]), labels=("zero", "one"))


def test_classification_result_preserves_arrays():
    class_ids = np.array([1, 0])
    scores = np.array([0.9, 0.8], dtype=np.float32)

    result = ClassificationResult(class_ids=class_ids, scores=scores)

    assert result.class_ids is class_ids
    assert result.scores is scores


def test_classifier_returns_top1_in_compatibility_shape():
    runner = FakeRunner(
        outputs=[np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)]
    )
    model = StubClassifier(runner, labels=("zero", "one"))

    assert model.predict([image(), image()]) == [
        {"class_ids": [1], "scores": [pytest.approx(0.9)]},
        {"class_ids": [0], "scores": [pytest.approx(0.8)]},
    ]
    runner.run.assert_called_once()
    inputs = runner.run.call_args.args[0]
    assert set(inputs) == {"images"}
    assert inputs["images"].shape == (2, 3, 32, 32)


def test_classifier_empty_input_does_not_call_runner():
    runner = FakeRunner(outputs=[])
    model = StubClassifier(runner, labels=("zero", "one"))

    assert model.predict([]) == []
    runner.run.assert_not_called()


def test_classifier_rejects_runner_without_input_metadata():
    with pytest.raises(ValueError, match="at least one input"):
        StubClassifier(EmptyMetadataRunner(), labels=("zero", "one"))


def test_classifier_rejects_runner_without_outputs():
    runner = FakeRunner(outputs=[])
    model = StubClassifier(runner, labels=("zero", "one"))

    with pytest.raises(RuntimeError, match="at least one output"):
        model.predict([image()])


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
    ],
)
def test_classifier_rejects_invalid_output(output):
    with pytest.raises(ValueError, match="classification output"):
        two_image_model(output).predict([image(), image()])
