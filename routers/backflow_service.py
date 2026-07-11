"""数据回流路径解析与落盘服务。"""

from dataclasses import dataclass
import json
import os
import re
from typing import Callable, Optional

from config import settings
from routers.upload_persistence import write_bytes_atomically
from utils import vision_logger


DATA_DIR = os.path.abspath(settings.DATA_DIR)
UNKNOWN_MODEL_DIR = "_unknown_model"
SAFE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class BackflowTarget:
    scene_dir: str
    model_dir: str
    save_stem: str


TargetResolver = Callable[[str, Optional[str]], BackflowTarget]


class BackflowService:
    def __init__(
        self,
        detector_type: str,
        target_resolver: TargetResolver,
        data_dir: str = DATA_DIR,
    ) -> None:
        self.detector_type = detector_type
        self.target_resolver = target_resolver
        self.data_dir = data_dir

    @staticmethod
    def sanitize_dir_name(name: str) -> str:
        sanitized = re.sub(r"[\\/]+", "_", str(name)).strip().strip(".")
        return sanitized or UNKNOWN_MODEL_DIR

    @staticmethod
    def safe_client_filename(filename: str) -> str:
        normalized = (filename or "unknown.jpg").replace("\\", "/")
        return normalized.rsplit("/", 1)[-1] or "unknown.jpg"

    @staticmethod
    def safe_path(root: str, *parts: str) -> str:
        root_path = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(root_path, *parts))
        if os.path.commonpath((root_path, candidate)) != root_path:
            raise ValueError("数据回流路径越界")
        return candidate

    @staticmethod
    def classify_result(result_dict: dict) -> str:
        status = result_dict.get("status")
        if status is True or (
            isinstance(status, str) and status.strip().lower() == "true"
        ):
            return "ok"
        return "ng"

    def resolve_paths(
        self,
        original_filename: str,
        received_at: str,
        fallback_product_type: Optional[str],
        verdict_dir: str,
        image_extension: Optional[str] = None,
    ) -> dict[str, str]:
        target = self.target_resolver(original_filename, fallback_product_type)
        if image_extension is not None:
            if image_extension not in SAFE_IMAGE_EXTENSIONS:
                raise ValueError(f"不支持的回流图片扩展名: {image_extension}")
            extension = image_extension
        else:
            safe_name = self.safe_client_filename(original_filename)
            extension = os.path.splitext(safe_name)[1].lower()
            if extension not in SAFE_IMAGE_EXTENSIONS:
                extension = ".jpg"

        scene_dir = self.sanitize_dir_name(target.scene_dir)
        model_name = self.sanitize_dir_name(target.model_dir)
        save_stem = self.sanitize_dir_name(target.save_stem)
        model_dir = self.safe_path(
            self.data_dir,
            scene_dir,
            received_at[:10],
            model_name,
            verdict_dir,
        )
        image_dir = self.safe_path(model_dir, "images")
        record_dir = self.safe_path(model_dir, "records")
        return {
            "scene_dir": scene_dir,
            "model_dir": model_name,
            "save_stem": save_stem,
            "verdict_dir": verdict_dir,
            "image_dir": image_dir,
            "record_dir": record_dir,
            "image_path": self.safe_path(image_dir, f"{save_stem}{extension}"),
            "record_path": self.safe_path(record_dir, f"{save_stem}.json"),
        }

    def persist_record(
        self,
        original_filename: str,
        raw_json: str,
        result_dict: dict,
        latency_ms: float,
        received_at: str,
        fallback_product_type: Optional[str] = None,
        raw_image_bytes: Optional[bytes] = None,
        image_extension: Optional[str] = None,
    ) -> None:
        """保存原始图片和结果记录，落盘失败不影响请求接口。"""
        try:
            is_error_record = isinstance(result_dict, dict) and "error" in result_dict
            verdict_dir = (
                "pending" if is_error_record else self.classify_result(result_dict)
            )
            paths = self.resolve_paths(
                original_filename,
                received_at,
                fallback_product_type,
                verdict_dir,
                image_extension=image_extension,
            )
            os.makedirs(paths["record_dir"], exist_ok=True)

            try:
                if not is_error_record:
                    pending_paths = self.resolve_paths(
                        original_filename,
                        received_at,
                        fallback_product_type,
                        "pending",
                        image_extension=image_extension,
                    )
                    if os.path.exists(pending_paths["image_path"]):
                        os.makedirs(paths["image_dir"], exist_ok=True)
                        os.replace(
                            pending_paths["image_path"],
                            paths["image_path"],
                        )
                        vision_logger.info(
                            "数据回流图片移动完成 src={} dst={}",
                            pending_paths["image_path"],
                            paths["image_path"],
                        )
                    elif raw_image_bytes is not None:
                        write_bytes_atomically(raw_image_bytes, paths["image_path"])
                    else:
                        vision_logger.warning(
                            "数据回流缺少 pending 原图且无原始字节 filename={}",
                            original_filename,
                        )
                elif raw_image_bytes is not None and not os.path.exists(
                    paths["image_path"]
                ):
                    write_bytes_atomically(raw_image_bytes, paths["image_path"])
            except Exception as exc:
                vision_logger.warning(
                    "数据回流图片落盘失败 filename={}: {}",
                    original_filename,
                    exc,
                )

            try:
                request_params = json.loads(raw_json)
            except json.JSONDecodeError:
                request_params = raw_json

            record = {
                "received_at": received_at,
                "detector_type": self.detector_type,
                "original_filename": original_filename,
                "saved_scene_dir": paths["scene_dir"],
                "saved_model_dir": paths["model_dir"],
                "verdict": "error" if is_error_record else verdict_dir,
                "request_params": request_params,
                "latency_ms": latency_ms,
                "result": result_dict,
            }
            with open(paths["record_path"], "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
        except Exception as exc:
            vision_logger.warning(
                "数据回流落盘失败 filename={}: {}", original_filename, exc
            )
