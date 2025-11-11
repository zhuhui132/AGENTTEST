"""
工具系统组件
提供统一的工具注册、参数校验与执行入口。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Sequence

from ..core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from ..core.interfaces import BaseTool
from ..core.types import ToolConfig, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistration:
    """工具注册信息"""

    name: str
    instance: BaseTool
    description: str = ""


class _CallableTool(BaseTool):
    """将简单函数包装为 BaseTool 的适配器"""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        description: str | None = None,
        schema: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._func = func
        self._description = description or ""
        self._schema = schema or {
            "type": "function",
            "function": {
                "name": name,
                "description": self._description or f"简易工具：{name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        self.validate_parameters(parameters)

        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**parameters)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._func(**parameters),
                )

            return ToolResult(
                tool_name=self._name,
                success=True,
                result=result,
                metadata=config or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具 %s 执行失败", self._name)
            return ToolResult(
                tool_name=self._name,
                success=False,
                error=str(exc),
                metadata=config or {},
            )

    def get_schema(self) -> Dict[str, Any]:
        return self._schema

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        schema_props = self._schema.get("function", {}).get("parameters", {}).get(
            "properties", {}
        )
        required = set(
            self._schema.get("function", {})
            .get("parameters", {})
            .get("required", [])
        )

        missing = [key for key in required if key not in parameters]
        if missing:
            raise ToolError(f"缺少必要参数: {', '.join(missing)}")

        # 基础类型检查
        for key, definition in schema_props.items():
            if key in parameters and "type" in definition:
                expected_type = definition["type"]
                value = parameters[key]
                if expected_type == "number" and not isinstance(value, (int, float)):
                    raise ToolError(f"参数 {key} 应为数值类型")
                if expected_type == "string" and not isinstance(value, str):
                    raise ToolError(f"参数 {key} 应为字符串类型")

        return True

    async def health_check(self) -> bool:
        return True


class ToolSystem:
    """工具系统管理类"""

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config = config or ToolConfig()
        self._tools: Dict[str, ToolRegistration] = {}
        self._lock = asyncio.Lock()
        self._usage_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "success": 0, "failure": 0, "total_time": 0.0}
        )
        logger.debug("工具系统初始化完成: max_tools=%s", self.config.max_tools)

    def list_tools(self) -> Iterable[str]:
        """列出所有已注册的工具名称"""
        return tuple(self._tools.keys())

    async def register_tool(
        self,
        name: str,
        tool: BaseTool,
        *,
        description: str = "",
        replace: bool = False,
    ) -> None:
        """注册工具实例"""
        async with self._lock:
            if name not in self._tools and len(self._tools) >= self.config.max_tools:
                raise ToolError(
                    f"已注册工具数量({len(self._tools)})达到上限 {self.config.max_tools}"
                )
            if name in self._tools and not replace:
                raise ToolError(f"工具 {name} 已存在，如需覆盖请设置 replace=True")

            self._tools[name] = ToolRegistration(name, tool, description)
            logger.info("注册工具: %s", name)

    async def register_callable(
        self,
        name: str,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        description: str | None = None,
        schema: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """使用普通函数快捷注册工具"""
        adapter = _CallableTool(name, func, description, schema)
        await self.register_tool(name, adapter, description=description or "", replace=replace)

    async def unregister_tool(self, name: str) -> None:
        """注销工具"""
        async with self._lock:
            self._tools.pop(name, None)
            logger.info("注销工具: %s", name)

    async def call_tool_async(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """异步调用工具"""
        async with self._lock:
            registration = self._tools.get(name)

        if registration is None:
            raise ToolNotFoundError(f"工具 {name} 未注册")

        tool = registration.instance
        params = parameters or {}
        timeout = config.get("timeout") if config else None
        timeout = timeout or self.config.default_timeout

        try:
            tool.validate_parameters(params)
            started_at = time.perf_counter()
            if timeout and timeout > 0:
                result = await asyncio.wait_for(tool.execute(params, config=config), timeout=timeout)
            else:
                result = await tool.execute(params, config=config)
            elapsed = time.perf_counter() - started_at
            self._record_usage(registration.name, success=True, elapsed=elapsed)
        except ToolError:
            self._record_usage(registration.name, success=False)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具 %s 执行过程中发生异常", name)
            self._record_usage(registration.name, success=False)
            raise ToolExecutionError(str(exc)) from exc

        return result

    def call_tool(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """同步调用工具"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.call_tool_async(name, parameters, config=config)
            )

        raise ToolError("当前存在运行中的事件循环，请使用 call_tool_async 调用工具")

    async def call_tools_parallel(
        self,
        plans: Sequence[Dict[str, Any]],
        *,
        return_exceptions: bool = False,
    ) -> List[ToolResult | Exception]:
        """根据计划并发执行多个工具调用"""
        if not plans:
            return []

        semaphore = asyncio.Semaphore(self.config.max_parallel_tools) if self.config.parallel_execution else None

        async def _run_plan(plan: Dict[str, Any]) -> ToolResult | Exception:
            async def _execute() -> ToolResult:
                return await self.call_tool_async(
                    plan["name"],
                    plan.get("parameters"),
                    config=plan.get("config"),
                )

            if semaphore is None:
                try:
                    return await _execute()
                except Exception as exc:  # noqa: BLE001
                    if return_exceptions:
                        return exc
                    raise

            async with semaphore:
                try:
                    return await _execute()
                except Exception as exc:  # noqa: BLE001
                    if return_exceptions:
                        return exc
                    raise

        return await asyncio.gather(*[_run_plan(plan) for plan in plans], return_exceptions=return_exceptions)

    async def ensure_all_tools_healthy(self) -> Dict[str, bool]:
        """执行健康检查"""
        results: Dict[str, bool] = {}
        async with self._lock:
            registrations = list(self._tools.values())

        for registration in registrations:
            try:
                results[registration.name] = await registration.instance.health_check()
            except Exception:  # noqa: BLE001
                logger.exception("工具 %s 健康检查失败", registration.name)
                results[registration.name] = False

        return results

    async def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """获取指定工具的参数模式"""
        async with self._lock:
            registration = self._tools.get(name)
        if not registration:
            raise ToolNotFoundError(f"工具 {name} 未注册")
        return registration.instance.get_schema()

    async def validate_tool_parameters(self, name: str, parameters: Dict[str, Any]) -> bool:
        """验证外部传入参数是否符合工具要求"""
        async with self._lock:
            registration = self._tools.get(name)
        if not registration:
            raise ToolNotFoundError(f"工具 {name} 未注册")
        return registration.instance.validate_parameters(parameters)

    async def describe_tool(self, name: str) -> Dict[str, Any]:
        """返回工具描述信息"""
        async with self._lock:
            registration = self._tools.get(name)
        if not registration:
            raise ToolNotFoundError(f"工具 {name} 未注册")
        return {
            "name": registration.name,
            "description": registration.description,
            "schema": registration.instance.get_schema(),
            "usage": self._usage_stats.get(name, {}),
        }

    def get_tool_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """返回所有工具的使用统计"""
        return {name: dict(stats) for name, stats in self._usage_stats.items()}

    def _record_usage(self, name: str, *, success: bool, elapsed: float = 0.0) -> None:
        stats = self._usage_stats[name]
        stats["calls"] += 1
        if success:
            stats["success"] += 1
            stats["total_time"] += elapsed
        else:
            stats["failure"] += 1


# 默认提供一个简单的计算器工具，便于快速体验
async def _default_calculator(**parameters: Any) -> Any:
    expression = parameters.get("expression")
    if not isinstance(expression, str):
        raise ToolError("计算器工具需要字符串表达式参数 expression")

    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ToolError("表达式包含非法字符")

    return eval(expression, {"__builtins__": {}})  # noqa: S307


async def create_default_tool_system() -> ToolSystem:
    """创建包含默认工具的 ToolSystem"""
    system = ToolSystem()
    await system.register_callable(
        "calculator",
        _default_calculator,
        description="计算简单的算术表达式",
        schema={
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "需要计算的算术表达式，例如 '1+2*3'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    )
    return system


__all__ = ["ToolSystem", "create_default_tool_system"]
