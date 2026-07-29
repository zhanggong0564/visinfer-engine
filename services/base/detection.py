"""Backend-independent object-detection result models."""

from dataclasses import dataclass, field

import numpy as np

from .geometry import Region


@dataclass(frozen=True)
class Detection:
    region: Region
    score: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    source_image: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
