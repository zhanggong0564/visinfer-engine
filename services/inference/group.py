"""Lifecycle management for a related group of inference runners."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from utils import vision_logger

from .backends import OnnxRuntimeOptions
from .contract import InferenceRunner
from .factory import RunnerSpec, create_inference_runner
from .status import RuntimeStatusRegistry


@dataclass(frozen=True)
class RunnerDefinition:
    spec: RunnerSpec
    onnx_options: OnnxRuntimeOptions


class InferenceRunnerGroup(Mapping[str, InferenceRunner]):
    """Create and release multiple named runners as one lifecycle unit."""

    def __init__(
        self,
        definitions: Sequence[RunnerDefinition],
        status_registry: RuntimeStatusRegistry | None = None,
    ) -> None:
        if not definitions:
            raise ValueError("runner group 不能为空")
        roles = [definition.spec.model_role for definition in definitions]
        if any(not role or not role.strip() for role in roles):
            raise ValueError("runner group 中的 model_role 不能为空")
        if len(set(roles)) != len(roles):
            raise ValueError("runner group 中的 model_role 不能重复")

        self._runners: dict[str, InferenceRunner] = {}
        try:
            for definition in definitions:
                role = definition.spec.model_role
                assert role is not None
                self._runners[role] = create_inference_runner(
                    definition.spec,
                    definition.onnx_options,
                    status_registry=status_registry,
                )
        except Exception:
            self.close()
            raise

    def __getitem__(self, role: str) -> InferenceRunner:
        return self._runners[role]

    def __iter__(self) -> Iterator[str]:
        return iter(self._runners)

    def __len__(self) -> int:
        return len(self._runners)

    def close(self) -> None:
        runners, self._runners = self._runners, {}
        for role, runner in reversed(tuple(runners.items())):
            try:
                runner.close()
            except Exception as exc:
                vision_logger.warning(
                    "runner group 关闭失败 role={}: {}", role, exc
                )
