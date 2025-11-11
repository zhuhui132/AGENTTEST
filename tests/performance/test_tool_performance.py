"""性能测试：工具系统与 Mock LLM"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.tools import ToolSystem  # noqa: E402
from src.llm.mock_llm import MockLLM  # noqa: E402
from src.core.types import LLMConfig  # noqa: E402


@pytest.mark.performance
@pytest.mark.asyncio
async def test_tool_parallel_performance():
    tools = ToolSystem()

    async def delayed(value: int) -> int:
        await asyncio.sleep(0.01)
        return value + 1

    for i in range(10):
        await tools.register_callable(f"tool_{i}", delayed, description=f"延迟工具{i}")

    plans = [{"name": f"tool_{i}", "parameters": {"value": i}} for i in range(10)]
    start = time.perf_counter()
    results = await tools.call_tools_parallel(plans)
    duration = time.perf_counter() - start

    assert all(res.success for res in results)
    assert duration < 0.2  # 10 个并发任务应该在 200ms 内完成


@pytest.mark.performance
def test_mock_llm_throughput():
    llm = MockLLM(LLMConfig(model_name="mock", max_tokens=128))
    asyncio.run(llm.initialize())

    start = time.perf_counter()
    for _ in range(200):
        asyncio.run(llm.generate("这是一个测试提示"))
    duration = time.perf_counter() - start

    throughput = 200 / duration
    assert throughput > 500  # 每秒至少 500 次调用

