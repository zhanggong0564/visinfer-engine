"""直流熔丝内置兼容场景的类型化配置。"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class DcFuseConfig(SceneSettings):
    """直流熔丝检测配置。

    服务形态下不挂到 config/config.py 的全局 Settings（那是框架共享配置），
    单独成文件由场景业务逻辑直接 import，保持"加场景不动框架"。
    """
    model_config = SettingsConfigDict(
        env_prefix="DC_FUSE_",
        env_file=".env",
        extra="ignore",
    )

    model_path: str = "./weights/dc_fuse/det_yolo_v5.onnx"
    conf_threshold: float = Field(default=0.6, ge=0, le=1)

    @property
    def confThreshold(self) -> float:
        return self.conf_threshold
