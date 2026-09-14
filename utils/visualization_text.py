"""Chinese text support without changing the shared overlay layout."""

from functools import lru_cache
from importlib.resources import files

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=8)
def legend_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(files("utils").joinpath("fonts/NotoSansCJKsc-Regular.otf")), size
    )


def measure_text(text: str, size: int) -> int:
    return int(legend_font(size).getlength(text))


def draw_text(canvas: np.ndarray, text: str, center_left: tuple[int, int], size: int) -> None:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(
        center_left, text, font=legend_font(size), fill=(20, 20, 20), anchor="lm",
    )
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
