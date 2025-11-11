"""工具系统简要测试（保持向后兼容）"""
from __future__ import annotations

import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.tools import ToolSystem  # noqa: E402
from src.core.types import ToolConfig  # noqa: E402
from src.core.exceptions import ToolError  # noqa: E402


@pytest.mark.asyncio
async def test_register_and_call_tool():
    tool_system = ToolSystem()

    async def multiply(value: int) -> int:
        return value * 2

    await tool_system.register_callable(
        "double",
        multiply,
        description="返回输入值的两倍",
        schema={
            "type": "function",
            "function": {
                "name": "double",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
            },
        },
    )

    result = await tool_system.call_tool_async("double", {"value": 21})
    assert result.success is True
    assert result.result == 42


@pytest.mark.asyncio
async def test_tool_register_limit_and_validation():
    tool_system = ToolSystem(ToolConfig(max_tools=1))

    await tool_system.register_callable("first", lambda: None)

    with pytest.raises(ToolError):
        await tool_system.register_callable("second", lambda: None)

    await tool_system.register_callable(
        "adder",
        lambda value: value + 1,
        replace=True,
        schema={
            "type": "function",
            "function": {
                "name": "adder",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
            },
        },
    )

    with pytest.raises(ToolError):
        await tool_system.call_tool_async("adder", {})


@pytest.mark.asyncio
async def test_tool_health_and_description():
    tool_system = ToolSystem()

    async def healthy_tool(number: int) -> int:
        return number

    await tool_system.register_callable(
        "healthy",
        healthy_tool,
        description="健康检查工具",
    )

    health = await tool_system.ensure_all_tools_healthy()
    assert health["healthy"] is True

    await tool_system.call_tool_async("healthy", {"number": 5})
    info = await tool_system.describe_tool("healthy")
    assert info["description"] == "健康检查工具"
    assert info["usage"]["calls"] == 1
