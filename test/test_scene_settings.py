from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class ExampleSceneSettings(SceneSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXAMPLE_SCENE_",
        env_file=".env",
        extra="ignore",
    )

    threshold: float = 0.5


def test_scene_settings_loads_prefixed_environment(monkeypatch):
    monkeypatch.setenv("EXAMPLE_SCENE_THRESHOLD", "0.75")

    assert ExampleSceneSettings().threshold == 0.75
