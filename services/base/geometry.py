"""Common geometry value objects used by inference results."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = tuple[Point, ...]


class CoordinateSpace(str, Enum):
    PIXEL = "PIXEL"
    NORMALIZED = "NORMALIZED"


@dataclass(frozen=True)
class Region:
    """A polygon with an explicit coordinate space."""

    polygon: Polygon
    space: CoordinateSpace = CoordinateSpace.PIXEL

    @property
    def center(self) -> Point:
        if not self.polygon:
            raise ValueError("region polygon cannot be empty")
        xs, ys = zip(*self.polygon)
        return sum(xs) / len(xs), sum(ys) / len(ys)


def xyxy_region(
    box: tuple[float, float, float, float] | list[float],
    space: CoordinateSpace = CoordinateSpace.PIXEL,
) -> Region:
    """Convert an ``x1, y1, x2, y2`` box to a four-point region."""

    if len(box) != 4:
        raise ValueError("xyxy box must contain four values")
    x1, y1, x2, y2 = (float(value) for value in box)
    return Region(
        polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        space=space,
    )
