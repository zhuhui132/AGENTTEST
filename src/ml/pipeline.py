"""轻量级特征工程流水线。"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence


PipelineStep = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


@dataclass
class PipelineResult:
    """流水线执行结果，包含中间特征与总结信息。"""

    output: Any
    artifacts: Dict[str, Any] = field(default_factory=dict)
    durations: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineStepDefinition:
    """描述单个流水线步骤。"""

    name: str
    callback: PipelineStep
    enabled: bool = True


class FeaturePipeline:
    """按顺序执行一组特征转换或推理步骤。"""

    def __init__(
        self,
        *,
        name: str = "pipeline",
        catch_exceptions: bool = True,
    ) -> None:
        self.name = name
        self._catch_exceptions = catch_exceptions
        self._steps: List[PipelineStepDefinition] = []

    def add_step(self, name: str, step: PipelineStep) -> None:
        self._steps.append(PipelineStepDefinition(name=name, callback=step))

    def enable_step(self, name: str, enabled: bool = True) -> None:
        for definition in self._steps:
            if definition.name == name:
                definition.enabled = enabled
                return
        raise ValueError(f"未找到名为 {name} 的步骤")

    def extend(self, steps: Iterable[tuple[str, PipelineStep]]) -> None:
        for name, step in steps:
            self.add_step(name, step)

    def run(self, data: Any) -> PipelineResult:
        """同步执行流水线；如存在异步步骤会自动开启事件循环。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(data))
        else:
            if loop.is_running():
                raise RuntimeError(
                    "FeaturePipeline.run 在已有事件循环中调用，"
                    "请改用 await pipeline.run_async(...)"
                )
            return loop.run_until_complete(self.run_async(data))

    async def run_async(self, data: Any) -> PipelineResult:
        artifacts: Dict[str, Any] = {}
        durations: Dict[str, float] = {}
        errors: Dict[str, str] = {}
        output = data

        for definition in self._steps:
            if not definition.enabled:
                continue

            start = time.perf_counter()
            try:
                result = definition.callback(output)
                if inspect.isawaitable(result):
                    output = await result  # type: ignore[assignment]
                else:
                    output = result
                artifacts[definition.name] = output
            except Exception as exc:  # noqa: BLE001
                errors[definition.name] = str(exc)
                if not self._catch_exceptions:
                    raise
            finally:
                durations[definition.name] = max(time.perf_counter() - start, 0.0)

        return PipelineResult(
            output=output,
            artifacts=artifacts,
            durations=durations,
            errors=errors,
        )

