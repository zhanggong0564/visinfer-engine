import numpy as np
import pytest

from services.base.ctc_recognition_pipeline import (
    BaseCtcRecognitionPipeline,
    CtcRecognitionResult,
)
from services.inference import TensorInfo


class _Runner:
    input_infos = (TensorInfo("images", (None, 3, 48, None), "tensor(float)"),)

    def __init__(self, logits=None):
        self.logits = logits
        self.last_input = None

    def run(self, inputs):
        self.last_input = inputs["images"]
        return [self.logits]


class _Recognizer(BaseCtcRecognitionPipeline):
    def preprocess_image(self, image, target_width):
        return np.zeros((3, self.input_height, target_width), np.float32)


class _BatchRecognizer(BaseCtcRecognitionPipeline):
    def __init__(self, *args, batch_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_shape = batch_shape

    def preprocess_batch(self, images):
        shape = self.batch_shape or (
            len(images),
            3,
            self.input_height,
            self._target_width(images),
        )
        return np.zeros(shape, np.float32)


def _probabilities(indices, classes, confidence=0.9):
    logits = np.full((1, len(indices), classes), (1 - confidence) / (classes - 1), np.float32)
    for timestep, index in enumerate(indices):
        logits[0, timestep, index] = confidence
    return logits


def _decoder(characters=("A", "B")):
    return _Recognizer(_Runner(), characters, input_height=48)


def test_ctc_decode_removes_blank_and_consecutive_duplicates():
    logits = _probabilities([1, 1, 0, 1, 2], classes=3, confidence=0.9)

    result = _decoder().decode(logits)[0]

    assert result.text == "AAB"
    assert result.score == pytest.approx(0.9)


def test_ctc_decode_empty_sequence_has_zero_confidence():
    result = _decoder().decode(_probabilities([0, 0], classes=3))[0]

    assert result == CtcRecognitionResult(text="", score=0.0)


@pytest.mark.parametrize("logits", [np.zeros((4, 3)), np.zeros((1, 4, 4, 3))])
def test_ctc_rejects_non_three_dimensional_output(logits):
    with pytest.raises(ValueError, match="character dictionary"):
        _decoder().decode(logits)


def test_ctc_rejects_character_dimension_mismatch():
    with pytest.raises(ValueError, match="character dictionary"):
        _decoder(characters=("A",)).decode(np.zeros((1, 4, 3), np.float32))


def test_recognizer_uses_widest_ratio_for_batch_width():
    runner = _Runner(np.zeros((2, 4, 3), np.float32))
    model = _Recognizer(runner, ("A", "B"), input_height=48)

    model.predict([
        np.zeros((20, 40, 3), np.uint8),
        np.zeros((20, 100, 3), np.uint8),
    ])

    assert runner.last_input.shape == (2, 3, 48, 240)


def test_recognizer_limits_dynamic_width_to_max_width():
    runner = _Runner(np.zeros((1, 4, 3), np.float32))
    model = _Recognizer(runner, ("A", "B"), input_height=48, max_width=160)

    model.predict([np.zeros((20, 100, 3), np.uint8)])

    assert runner.last_input.shape == (1, 3, 48, 160)


def test_recognizer_returns_empty_list_without_running_backend():
    runner = _Runner()
    model = _Recognizer(runner, ("A", "B"), input_height=48)

    assert model.predict([]) == []
    assert runner.last_input is None


def test_recognizer_rejects_invalid_image_shape():
    with pytest.raises(ValueError, match="image"):
        _decoder().predict([np.zeros((20,), np.uint8)])


def test_recognizer_rejects_output_batch_dimension_mismatch():
    runner = _Runner(np.zeros((1, 4, 3), np.float32))
    model = _Recognizer(runner, ("A", "B"), input_height=48)

    images = [np.zeros((20, 40, 3), np.uint8)] * 2
    with pytest.raises(ValueError, match="batch dimension"):
        model.predict(images)


def test_recognizer_accepts_preprocess_batch_override_without_preprocess_image():
    runner = _Runner(np.zeros((1, 4, 3), np.float32))
    model = _BatchRecognizer(runner, ("A", "B"), input_height=48)

    model.predict([np.zeros((20, 100, 3), np.uint8)])

    assert runner.last_input.shape == (1, 3, 48, 240)


def test_recognizer_rejects_invalid_preprocess_batch_override_shape():
    runner = _Runner(np.zeros((1, 4, 3), np.float32))
    model = _BatchRecognizer(
        runner,
        ("A", "B"),
        input_height=48,
        batch_shape=(1, 3, 48, 239),
    )

    with pytest.raises(ValueError, match="preprocessed batch shape"):
        model.predict([np.zeros((20, 100, 3), np.uint8)])
