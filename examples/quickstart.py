"""IntelligentAgent 快速上手示例."""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.agent import IntelligentAgent  # noqa: E402
from src.core.types import AgentMessage  # noqa: E402

ASYNC_HELLO_PROMPT = "你好，请简要介绍一下你的能力。"
ASYNC_EXPAND_PROMPT = "我想扩展你的功能，有哪些建议？"


async def main() -> None:
    """运行两个最小示例对话，展示异步 Agent 的基本用法。"""
    agent = IntelligentAgent()
    await agent.initialize()

    try:
        print("=== IntelligentAgent Quickstart ===")

        greeting = await agent.process_message(ASYNC_HELLO_PROMPT)
        print("\n[示例1] 基础问候对话")
        print("用户:", ASYNC_HELLO_PROMPT)
        print("Agent:", greeting.content)
        print("响应耗时(s):", greeting.processing_time)

        message = AgentMessage(role="user", content=ASYNC_EXPAND_PROMPT)
        extension = await agent.process_message(message)
        print("\n[示例2] 扩展能力建议")
        print("用户:", message.content)
        print("Agent:", extension.content)
        print("引用来源:", extension.sources or "无")
        print("置信度:", extension.confidence)

        print("\nAgent 状态:", agent.get_status())
    finally:
        await agent.shutdown()
        print("Agent 已关闭。")


if __name__ == "__main__":
    asyncio.run(main())
