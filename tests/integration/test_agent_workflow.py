"""集成测试：Agent 生命周期与子系统协作"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.factory import AgentFactory  # noqa: E402
from src.core.types import AgentMessage  # noqa: E402
from src.memory.memory import MemorySystem  # noqa: E402
from src.rag.rag import RAGSystem  # noqa: E402
from src.utils.tools import ToolSystem  # noqa: E402
from src.utils.metrics import AccuracyMetrics  # noqa: E402


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_factory_creates_working_agent():
    factory = AgentFactory()
    agent = await factory.create_agent()
    try:
        response = await agent.process_message("请计算 2 + 3")
        assert "mock reply" in response.content
        assert response.metadata["session_id"] == agent.session_id
        assert any(call["tool_name"] == "calculator" for call in response.tool_calls)

        metrics = AccuracyMetrics()
        score = metrics.factual_accuracy(response.content, "结果是 5")
        assert score > 0.2
    finally:
        await agent.shutdown()


@pytest.mark.integration
def test_memory_rag_tool_components_cooperate():
    memory = MemorySystem()
    rag = RAGSystem()
    tools = ToolSystem()

    async def _prepare():
        await memory.add_memory("用户喜欢阅读科幻小说", metadata={"tag": "preference"})
        await rag.add_document("科幻书单", "推荐《三体》《沙丘》", metadata={"category": "book"})
        await tools.register_callable("echo", lambda text: text, description="回显工具")

    asyncio.run(_prepare())

    stats = asyncio.run(memory.stats())
    assert stats["total_memories"] == 1

    docs = asyncio.run(rag.retrieve("书单"))
    assert docs and docs[0].title == "科幻书单"

    result = asyncio.run(tools.call_tool_async("echo", {"text": "hello"}))
    assert result.success and result.result == "hello"

