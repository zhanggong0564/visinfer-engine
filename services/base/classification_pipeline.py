"""Backend-independent classification pipeline contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from services.inference import InferenceRunner


@dataclass(frozen=True)
class ClassificationResult:
    """One top-1 classification prediction."""

    class_id: int
    score: float


class BaseClassificationPipeline(ABC):
    """Common classification flow independent of model architecture."""

    def __init__(self, runner: InferenceRunner, labels: Sequence[str]) -> None:
        if not runner.input_infos:
            raise ValueError("classification runner must expose at least one input")
        self.runner = runner
        self.labels = tuple(labels)
        self.input_name = runner.input_infos[0].name

    @abstractmethod
    def preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Convert input images into the model input tensor."""

    def predict(
        self, images: Sequence[np.ndarray]
    ) -> list[ClassificationResult]:
        """Return one backend-independent top-1 result per image."""
        if not images:
            return []

        tensor = self.preprocess(images)
        outputs = self.runner.run({self.input_name: tensor})
        if not outputs:
            raise RuntimeError("classification runner must return at least one output")
        logits = outputs[0]
        expected_shape = (len(images), len(self.labels))
        if logits.ndim != 2 or logits.shape != expected_shape:
            raise ValueError("classification output shape does not match contract")

        class_ids = logits.argmax(axis=1)
        rows = np.arange(len(images))
        return [
            ClassificationResult(class_id=int(class_id), score=float(score))
            for class_id, score in zip(class_ids, logits[rows, class_ids])
        ]

    def close(self) -> None:
        self.runner.close()
