"""Backend-independent dynamic-width CTC recognition pipeline."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from services.inference import InferenceRunner


@dataclass(frozen=True)
class CtcRecognitionResult:
    """One greedily decoded CTC recognition result."""

    text: str
    score: float


class BaseCtcRecognitionPipeline:
    """Provide dynamic-width batching and greedy CTC decoding."""

    def __init__(
        self,
        runner: InferenceRunner,
        characters: Sequence[str],
        input_height: int,
        max_width: int | None = None,
    ) -> None:
        if not runner.input_infos:
            raise ValueError("runner must declare at least one input")
        if input_height <= 0:
            raise ValueError("input_height must be positive")
        if max_width is not None and max_width <= 0:
            raise ValueError("max_width must be positive")

        self.runner = runner
        self.characters = tuple(characters)
        self.input_height = input_height
        self.max_width = max_width
        self.input_name = runner.input_infos[0].name

    def preprocess_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Preprocess a batch; subclasses may override this method."""
        target_width = self._target_width(images)
        tensors = [self.preprocess_image(image, target_width) for image in images]
        return np.stack(tensors)

    def preprocess_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """Preprocess one image to CHW; subclasses may override this method."""
        raise NotImplementedError(
            "subclass must implement preprocess_batch or preprocess_image"
        )

    def decode(self, logits: np.ndarray) -> list[CtcRecognitionResult]:
        """Greedily decode CTC logits, using class index zero as blank."""
        if logits.ndim != 3 or logits.shape[2] != len(self.characters) + 1:
            raise ValueError("CTC output does not match character dictionary")

        indices = logits.argmax(axis=2)
        scores = logits.max(axis=2)
        results: list[CtcRecognitionResult] = []
        for row_indices, row_scores in zip(indices, scores):
            keep = [
                index
                for index, value in enumerate(row_indices)
                if value != 0
                and (index == 0 or value != row_indices[index - 1])
            ]
            text = "".join(
                self.characters[int(row_indices[index]) - 1] for index in keep
            )
            score = float(np.mean(row_scores[keep])) if keep else 0.0
            results.append(CtcRecognitionResult(text=text, score=score))
        return results

    def predict(self, images: Sequence[np.ndarray]) -> list[CtcRecognitionResult]:
        """Preprocess, infer and decode a batch of images."""
        if not images:
            return []

        self._target_width(images)
        tensor = np.asarray(self.preprocess_batch(images))
        expected_width = self._target_width(images)
        expected_shape = (len(images), 3, self.input_height, expected_width)
        if tensor.ndim != 4 or tensor.shape != expected_shape:
            raise ValueError(
                f"preprocessed batch shape must be {expected_shape}, got {tensor.shape}"
            )

        outputs = self.runner.run({self.input_name: tensor})
        if not outputs:
            raise ValueError("runner returned no CTC output")
        logits = np.asarray(outputs[0])
        if logits.ndim == 3 and logits.shape[0] != len(images):
            raise ValueError(
                "CTC output batch dimension does not match input images"
            )
        return self.decode(logits)

    def _target_width(self, images: Sequence[np.ndarray]) -> int:
        ratios = []
        for image in images:
            if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
                raise ValueError("each image must be a two- or three-dimensional array")
            height, width = image.shape[:2]
            if height <= 0 or width <= 0:
                raise ValueError("image dimensions must be positive")
            ratios.append(width / height)

        target_width = max(1, int(np.ceil(self.input_height * max(ratios))))
        if self.max_width is not None:
            target_width = min(target_width, self.max_width)
        return target_width

    def close(self) -> None:
        self.runner.close()
