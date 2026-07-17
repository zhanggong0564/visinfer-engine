"""ScenarioRegistry scene construction and rollback contracts."""

import pytest

from services.scenario_registry import ScenarioRegistry


class _Settings:
    pass


def test_registry_instances_do_not_share_registrations():
    first = ScenarioRegistry(_Settings())
    second = ScenarioRegistry(_Settings())

    first.register("scene")(object)

    assert first.list_scenarios() == ["scene"]
    assert second.list_scenarios() == []


def test_create_injects_settings_and_returns_a_fresh_instance():
    settings = _Settings()
    registry = ScenarioRegistry(settings)

    @registry.register("scene")
    class Scene:
        def __init__(self, received_settings):
            self.settings = received_settings

    first = registry.create("scene")
    second = registry.create("scene")

    assert isinstance(first, Scene)
    assert first.settings is settings
    assert second is not first


def test_create_rejects_unregistered_scenario():
    registry = ScenarioRegistry(_Settings())

    with pytest.raises(ValueError, match="not registered"):
        registry.create("missing")


def test_snapshot_and_full_restore_replace_registry_state():
    registry = ScenarioRegistry(_Settings())
    original = type("Original", (), {})
    replacement = type("Replacement", (), {})
    registry.register("scene")(original)
    snapshot = registry.snapshot()

    registry.register("scene")(replacement)
    registry.register("new")(object)
    registry.restore(snapshot)

    assert registry.snapshot() == {"scene": original}


def test_selective_restore_resets_only_named_scenarios():
    registry = ScenarioRegistry(_Settings())
    original = type("Original", (), {})
    replacement = type("Replacement", (), {})
    registry.register("scene")(original)
    snapshot = registry.snapshot()

    registry.register("scene")(replacement)
    registry.register("new")(object)
    registry.restore(snapshot, scenario_names={"scene", "new"})

    assert registry.snapshot() == {"scene": original}
