"""针对 Agent 关键子系统的单元测试"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.agent import IntelligentAgent  # noqa: E402
from src.core.types import AgentMessage, MemoryConfig, RAGConfig, ToolConfig, LLMConfig  # noqa: E402
from src.memory.memory import MemorySystem  # noqa: E402
from src.rag.rag import RAGSystem  # noqa: E402
from src.utils.tools import ToolSystem  # noqa: E402
from src.utils.context import ContextManager  # noqa: E402
from src.llm.base import BaseLLM  # noqa: E402
from src.llm.manager import LLMManager  # noqa: E402
from src.llm.mock_llm import MockLLM  # noqa: E402
from src.core.types import PerformanceMetrics  # noqa: E402
from src.core.exceptions import LLMError  # noqa: E402
from src.utils.metrics import AccuracyMetrics  # noqa: E402


@pytest.mark.asyncio
async def test_memory_system_metadata_and_stats():
    memory = MemorySystem(MemoryConfig(max_memories=10, similarity_threshold=0.1))
    await memory.add_memory(
        "用户喜欢看科幻电影",
        metadata={"role": "user", "domain": "movie"},
    )
    await memory.add_memory(
        "助手建议每周整理笔记",
        metadata={"role": "assistant", "domain": "productivity"},
    )

    results = await memory.retrieve("科幻")
    assert results and results[0].metadata["domain"] == "movie"

    by_role = await memory.retrieve_by_metadata("role", "assistant")
    assert len(by_role) == 1

    stats = await memory.stats()
    assert stats["total_memories"] == 2
    assert stats["per_type"]["episodic"] == 2


@pytest.mark.asyncio
async def test_rag_system_chunking_and_filters():
    rag = RAGSystem(RAGConfig(chunk_size=40, chunk_overlap=10, similarity_threshold=0.0))
    long_content = "上海是金融中心，外滩很美" * 5
    doc_id = await rag.add_document(
        "上海出差攻略",
        long_content,
        metadata={"topic": "travel"},
    )
    await rag.update_document(doc_id, {"tags": ["travel"]})

    documents = await rag.retrieve("金融 外滩", filters={"tag": "travel"})
    assert documents
    assert any("上海" in doc.content for doc in documents)

    stats = await rag.stats()
    assert stats["total_documents"] >= 1
    assert stats["per_tag"]["travel"] >= 1


@pytest.mark.asyncio
async def test_tool_system_parallel_execution():
    tools = ToolSystem(ToolConfig(max_parallel_tools=2))

    async def async_increment(value: int) -> int:
        return value + 1

    def double(value: int) -> int:
        return value * 2

    await tools.register_callable("inc", async_increment, description="加一", schema={
        "type": "function",
        "function": {
            "name": "inc",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        },
    })
    await tools.register_callable("double", double, description="乘二", schema={
        "type": "function",
        "function": {
            "name": "double",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        },
    })

    plans = [
        {"name": "inc", "parameters": {"value": 1}},
        {"name": "double", "parameters": {"value": 3}},
    ]
    results = await tools.call_tools_parallel(plans)
    payload = [res.result for res in results]
    assert payload == [2, 6]

    usage = tools.get_tool_usage_stats()
    assert usage["inc"]["calls"] == 1 and usage["inc"]["success"] == 1
    assert usage["double"]["calls"] == 1 and usage["double"]["success"] == 1


@pytest.mark.asyncio
async def test_context_manager_custom_hooks():
    def custom_summarizer(messages: List[AgentMessage]) -> str:
        return f"共{len(messages)}条记录"

    def extract_numbers(message: AgentMessage) -> List[str]:
        digits = "".join(ch if ch.isdigit() else " " for ch in message.content).split()
        return [d for d in digits if d]

    def classify_intent(message: AgentMessage) -> str | None:
        if "请" in message.content:
            return "request"
        return None

    manager = ContextManager(
        summarizer=custom_summarizer,
        entity_extractors=(extract_numbers,),
        intent_classifier=classify_intent,
    )

    messages = [
        AgentMessage(role="user", content="请记录任务 123"),
        AgentMessage(role="assistant", content="好的，已经记录"),
    ]
    context = await manager.build_context(messages)
    assert context.summary == "共2条记录"
    assert "123" in context.key_entities
    assert context.user_intent == "request"

    recent = await manager.get_recent_messages(context.conversation_id, limit=1)
    assert len(recent) == 1

    await manager.attach_additional_info(context.conversation_id, priority="high")
    updated = await manager.get_context(context.conversation_id)
    assert updated and updated.additional_info["priority"] == "high"


class _EchoLLM(BaseLLM):
    async def _initialize_model(self) -> None:
        return

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        return prompt.splitlines()[0]

    async def _generate_stream_impl(self, prompt: str, config: LLMConfig):
        yield prompt

    async def _embed_impl(self, texts):
        if isinstance(texts, str):
            return [float(len(texts))]
        return [[float(len(text))] for text in texts]


class _BrokenLLM(BaseLLM):
    async def _initialize_model(self) -> None:
        return

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        raise LLMError("LLM 生成失败")

    async def _generate_stream_impl(self, prompt: str, config: LLMConfig):
        raise LLMError("LLM 流式生成失败")

    async def _embed_impl(self, texts):
        raise LLMError("LLM 嵌入失败")


class _MathLLM(BaseLLM):
    async def _initialize_model(self) -> None:
        return

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        return "计算结果为 2"

    async def _generate_stream_impl(self, prompt: str, config: LLMConfig):
        yield await self._generate_impl(prompt, config)

    async def _embed_impl(self, texts):
        return [1.0]


@pytest.mark.asyncio
async def test_llm_manager_metrics_callback():
    events: List[Dict[str, Any]] = []

    manager = LLMManager()
    manager.register("echo", lambda cfg: _EchoLLM(cfg))
    manager.set_default("echo")
    manager.set_metrics_callback(lambda event, payload: events.append({"event": event, **payload}))

    llm = await manager.get()
    reply = await llm.generate("这是提示")
    assert reply == "这是提示"
    assert events and events[0]["event"] == "generate"


@pytest.mark.asyncio
async def test_intelligent_agent_process_message():
    agent = IntelligentAgent(
        memory_system=MemorySystem(MemoryConfig(max_memories=20)),
        rag_system=RAGSystem(RAGConfig(similarity_threshold=0.0)),
        tools=ToolSystem(ToolConfig(max_parallel_tools=2)),
        context_manager=ContextManager(),
        llm=MockLLM(LLMConfig(model_name="mock")),
    )
    await agent.initialize()

    response = await agent.process_message("请计算 1 + 1")
    assert response.content.startswith("[mock reply]")
    assert response.metadata["llm_metrics"]
    assert any(call["tool_name"] == "calculator" for call in response.tool_calls)

    await agent.shutdown()


@pytest.mark.asyncio
async def test_agent_fallback_on_llm_error():
    agent = IntelligentAgent(
        memory_system=MemorySystem(),
        rag_system=RAGSystem(RAGConfig(similarity_threshold=0.0)),
        tools=ToolSystem(),
        context_manager=ContextManager(),
        llm=_BrokenLLM(LLMConfig(model_name="broken")),
    )
    await agent.initialize()

    response = await agent.process_message("请帮我计算 3 + 4")
    assert "LLM 暂不可用" in response.content
    assert response.metadata["llm_prompt"]["tool_calls"] == ["calculator"]

    await agent.shutdown()


@pytest.mark.asyncio
async def test_agent_response_accuracy_metric():
    agent = IntelligentAgent(
        memory_system=MemorySystem(),
        rag_system=RAGSystem(RAGConfig(similarity_threshold=0.0)),
        tools=ToolSystem(),
        context_manager=ContextManager(),
        llm=_MathLLM(LLMConfig(model_name="math")),
    )
    await agent.initialize()

    response = await agent.process_message("请告诉我 1 + 1 的结果")
    metrics = AccuracyMetrics()
    score = metrics.factual_accuracy(response.content, "计算结果为 2")
    assert score >= 0.5

    await agent.shutdown()

