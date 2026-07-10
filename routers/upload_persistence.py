"""上传原图格式识别、解码与原子暂存原语。"""

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def detect_image_extension(payload: bytes) -> str:
    """按文件头返回规范扩展名，只接受 JPEG、PNG 和 BMP。"""
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(_PNG_SIGNATURE):
        return ".png"
    if payload.startswith(b"BM"):
        return ".bmp"
    raise ValueError("不支持的图片格式，仅支持 JPEG、PNG、BMP")


def decode_image(payload: bytes) -> np.ndarray:
    """把原始图片字节解码为 BGR ndarray。"""
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("图片解码失败")
    return image


@dataclass
class StagedImageWrite:
    temporary_path: str
    final_path: str

    @classmethod
    def write(cls, payload: bytes, final_path: str) -> "StagedImageWrite":
        image_dir = os.path.dirname(final_path) or "."
        os.makedirs(image_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".vie-upload-",
            suffix=".tmp",
            dir=image_dir,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return cls(temporary_path=temporary_path, final_path=final_path)

    def commit(self) -> None:
        os.replace(self.temporary_path, self.final_path)

    def discard(self) -> None:
        try:
            os.unlink(self.temporary_path)
        except FileNotFoundError:
            pass


def write_bytes_atomically(payload: bytes, final_path: str) -> None:
    staged = StagedImageWrite.write(payload, final_path)
    try:
        staged.commit()
    except Exception:
        staged.discard()
        raise
