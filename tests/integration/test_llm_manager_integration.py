"""LLM Manager 与 ToolSystem 集成测试"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.base import LLMWithTools  # noqa: E402
from src.llm.manager import LLMManager  # noqa: E402
from src.core.types import LLMConfig  # noqa: E402
from src.utils.tools import ToolSystem  # noqa: E402
from src.core.types import ToolResult  # noqa: E402


class _ToolAwareLLM(LLMWithTools):
    async def _initialize_model(self) -> None:
        return

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        return f"回复：{prompt.splitlines()[0]}"

    async def _generate_stream_impl(self, prompt: str, config: LLMConfig):
        yield await self._generate_impl(prompt, config)

    async def _embed_impl(self, texts):
        return [float(len(texts))] if isinstance(texts, str) else [[float(len(t))] for t in texts]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_manager_register_and_metrics(monkeypatch):
    metrics_events = []

    tools = ToolSystem()

    async def mock_executor(name: str, params: dict) -> ToolResult:
        return ToolResult(tool_name=name, success=True, result=params.get("value"))

    await tools.register_callable("adder", lambda value: value + 1, description="加一")

    manager = LLMManager()
    manager.register("tool-aware", lambda cfg: _ToolAwareLLM(cfg))
    manager.set_default("tool-aware")
    manager.set_metrics_callback(lambda event, payload: metrics_events.append({"event": event, **payload}))

    llm = await manager.get()
    assert isinstance(llm, LLMWithTools)
    llm.set_tool_executor(mock_executor)

    reply = await llm.generate_with_tools("adder", tools=["adder"])
    assert reply.startswith("回复")
    assert metrics_events and metrics_events[0]["event"] == "generate"

    await manager.shutdown()

