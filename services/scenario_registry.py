"""Scene registration and construction."""

from collections.abc import Iterable, Mapping
from typing import Any, Type

from config import settings

from .base.business_logic_base import BusinessLogicBase


# Cython evaluates this alias while initializing the compiled module. Using
# typing.Type keeps the Python 3.10 wheel importable after Cython compilation.
ScenarioType = Type[BusinessLogicBase]


class ScenarioRegistry:
    """Store scene types and create business-logic instances on demand."""

    def __init__(self, application_settings: Any) -> None:
        self._settings = application_settings
        self._registry: dict[str, ScenarioType] = {}

    def register(self, scenario: str):
        def decorator(scene_type: ScenarioType) -> ScenarioType:
            self._registry[scenario] = scene_type
            return scene_type

        return decorator

    def create(self, scenario: str) -> BusinessLogicBase:
        try:
            scene_type = self._registry[scenario]
        except KeyError as exc:
            raise ValueError(f"Scenario {scenario} not registered") from exc
        return scene_type(self._settings)

    def list_scenarios(self) -> list[str]:
        return list(self._registry)

    def snapshot(self) -> dict[str, ScenarioType]:
        return dict(self._registry)

    def restore(
        self,
        snapshot: Mapping[str, ScenarioType],
        scenario_names: Iterable[str] | None = None,
    ) -> None:
        if scenario_names is None:
            self._registry = dict(snapshot)
            return
        for scenario in scenario_names:
            if scenario in snapshot:
                self._registry[scenario] = snapshot[scenario]
            else:
                self._registry.pop(scenario, None)


scenario_registry = ScenarioRegistry(settings)
