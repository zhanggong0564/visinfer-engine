"""Common OCR result models shared by scene plugins."""

from dataclasses import dataclass, field

from .geometry import Region


@dataclass(frozen=True)
class OCRToken:
    """One detected text region and its independent model scores."""

    text: str | None
    region: Region
    recognition_score: float | None = None
    detection_score: float | None = None

    @property
    def polygon(self) -> list[list[float]]:
        """Legacy list representation used by existing OCR geometry code."""

        return [[x, y] for x, y in self.region.polygon]

    @property
    def confidence(self) -> float:
        """Legacy alias for recognition confidence."""

        return self.recognition_score or 0.0

    @property
    def center(self) -> tuple[float, float]:
        return self.region.center


@dataclass(frozen=True)
class OCRResult:
    tokens: tuple[OCRToken, ...] = field(default_factory=tuple)
