"""Shared settings base for scene plugins."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SceneSettings(BaseSettings):
    """Load typed scene defaults with environment and ``.env`` overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )
