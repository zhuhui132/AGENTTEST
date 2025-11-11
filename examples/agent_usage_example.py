"""基于文档结构的 Agent 功能综合演示."""

import asyncio
import os
import sys
from typing import Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.agent import IntelligentAgent  # noqa: E402
from src.core.types import (  # noqa: E402
    AgentConfig,
    AgentMessage,
    MemoryConfig,
    MemoryType,
    RAGConfig,
)
from src.memory.memory import MemorySystem  # noqa: E402
from src.rag.rag import RAGSystem  # noqa: E402
from src.utils.tools import ToolSystem  # noqa: E402


async def _secure_calculator(*, expression: str) -> float:
    """轻量级四则运算工具，拒绝非法字符。"""
    allow = set("0123456789+-*/(). ")
    if not set(expression) <= allow:
        raise ValueError("表达式包含非法字符")
    return eval(expression, {"__builtins__": {}})  # noqa: S307


async def _weather_lookup(*, city: str) -> str:
    """模拟天气查询，可用于演示工具调用。"""
    samples = {
        "北京": "晴，3~12℃，注意早晚温差。",
        "上海": "多云，8~15℃，建议携带薄外套。",
        "深圳": "阵雨，18~24℃，记得带伞。",
    }
    return samples.get(city, f"暂未收录 {city} 的天气数据")


async def build_agent() -> Tuple[IntelligentAgent, MemorySystem, RAGSystem, ToolSystem]:
    """依据文档推荐的配置初始化 Agent 及其子系统。"""
    memory = MemorySystem(MemoryConfig(max_memories=64, cleanup_interval=3600))
    rag = RAGSystem(RAGConfig(max_documents=120, similarity_threshold=0.2))
    tools = ToolSystem()

    await tools.register_callable(
        "calculator",
        _secure_calculator,
        description="执行基础四则运算",
        schema={
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "根据表达式计算结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "支持 + - * / 的数学表达式",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    )

    await tools.register_callable(
        "weather_lookup",
        _weather_lookup,
        description="查询预置城市的天气信息",
        schema={
            "type": "function",
            "function": {
                "name": "weather_lookup",
                "description": "根据城市名称返回天气概述",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市中文名称"}
                    },
                    "required": ["city"],
                },
            },
        },
    )

    agent = IntelligentAgent(
        config=AgentConfig(name="DocsDemoAgent", memory_enabled=True, tools_enabled=True),
        memory_system=memory,
        rag_system=rag,
        tools=tools,
    )
    await agent.initialize()
    return agent, memory, rag, tools


async def seed_memory(memory: MemorySystem) -> None:
    """写入示例记忆，并展示检索与统计。"""
    print("\n=== 记忆系统演示 ===")
    await memory.add_memory(
        "用户姓名是陈晨，来自上海",
        importance=3.0,
        metadata={"source": "profile", "topic": "personal"},
        memory_type=MemoryType.EPISODIC,
    )
    await memory.add_memory(
        "用户偏好川菜，尤其喜欢冒菜",
        importance=2.5,
        metadata={"source": "preference", "topic": "food"},
        memory_type=MemoryType.SEMANTIC,
    )

    memories = await memory.retrieve("用户信息", limit=5)
    for item in memories:
        print("- 记忆:", item.content, "| 重要性:", item.importance)

    stats = await memory.stats()
    print("记忆统计:", stats)


async def seed_documents(rag: RAGSystem) -> None:
    """为 RAG 索引加载基础文档，并演示检索结果。"""
    print("\n=== RAG 系统演示 ===")
    await rag.add_document(
        title="上海出差指南",
        content="上海金融区集中在陆家嘴，建议提前预约会议室并关注天气。",
        source="internal-wiki",
        metadata={"topic": "travel", "city": "上海"},
    )
    await rag.add_document(
        title="数据科学家工具清单",
        content="推荐携带高性能笔记本、VPN、常用数据集和调试脚本。",
        source="internal-wiki",
        metadata={"topic": "work"},
    )

    docs = await rag.retrieve("上海 出差", limit=3)
    for doc in docs:
        print(f"- 文档: {doc.title} | 来源: {doc.source}")


async def demo_tool_direct_call(tools: ToolSystem) -> None:
    """展示如何在 Agent 外部直接调用已注册工具。"""
    print("\n=== 工具系统演示 ===")
    print("已注册工具:", list(tools.list_tools()))

    weather = await tools.call_tool_async("weather_lookup", {"city": "上海"})
    print("天气查询结果:", weather.result)

    calc = await tools.call_tool_async("calculator", {"expression": "24 * 3 / 2"})
    print("计算结果:", calc.result)


async def demo_conversation(agent: IntelligentAgent) -> None:
    """结合记忆、检索与工具的一次多轮对话。"""
    print("\n=== 多轮对话演示 ===")
    prompts = [
        "你好，我第一次使用这个 Agent。",
        "请记住，我叫陈晨，下周要去上海出差。",
        "帮我计算 24 * 3 / 2 的结果。",
        "根据我的行程，给一个准备清单。",
    ]

    for text in prompts:
        message = AgentMessage(role="user", content=text)
        response = await agent.process_message(message)
        print("\n用户:", text)
        print("Agent:", response.content)
        print("置信度:", response.confidence)
        print("引用文档:", response.sources or "无")
        print("工具调用:", response.tool_calls or "无")

    print("\nAgent 状态:", agent.get_status())


async def main() -> None:
    agent, memory, rag, tools = await build_agent()

    try:
        await seed_memory(memory)
        await seed_documents(rag)
        await demo_tool_direct_call(tools)
        await demo_conversation(agent)
    finally:
        await agent.shutdown()
        print("\nAgent 已安全关闭。")


if __name__ == "__main__":
    asyncio.run(main())
