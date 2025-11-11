"""集成测试：FeaturePipeline 与模型注册器"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.pipeline import FeaturePipeline, PipelineStepDefinition  # noqa: E402
from src.ml.registry import ModelRegistry  # noqa: E402


@pytest.mark.integration
def test_feature_pipeline_async_steps_and_errors():
    pipeline = FeaturePipeline(name="integration-pipeline")

    async def step_one(data: list[int]) -> list[int]:
        await asyncio.sleep(0.005)
        return [x + 1 for x in data]

    def step_two(data: list[int]) -> list[int]:
        return [x * 2 for x in data]

    def failing_step(data: list[int]) -> list[int]:
        raise ValueError("fake error")

    pipeline.add_step("step_one", step_one)
    pipeline.add_step("step_two", step_two)
    pipeline.add_step("failing", failing_step)
    pipeline.enable_step("failing", enabled=True)

    result = asyncio.run(pipeline.run_async([1, 2, 3]))

    assert result.output == [4, 6, 8]
    assert result.artifacts["step_two"] == [4, 6, 8]
    assert "failing" in result.errors
    assert result.durations["step_one"] > 0


@pytest.mark.integration
def test_model_registry_default_and_async():
    registry = ModelRegistry()

    async def async_model(value: int) -> int:
        await asyncio.sleep(0.001)
        return value + 10

    registry.register("sync", lambda x: x * 2)
    registry.register("async", async_model)
    registry.set_default("sync")

    assert registry.invoke(None, 5) == 10

    result = asyncio.run(registry.invoke_async("async", 5))
    assert result == 15

