"""LLM 管理器：集中管理模型实例与配置。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Optional

from ..core.exceptions import LLMError
from ..core.types import LLMConfig
from .base import BaseLLM
from .mock_llm import MockLLM


LLMFactory = Callable[[LLMConfig], Awaitable[BaseLLM] | BaseLLM]


class LLMManager:
    """支持按名称注册/获取 LLM，并统一管理生命周期。"""

    def __init__(self) -> None:
        self._providers: Dict[str, LLMFactory] = {}
        self._instances: Dict[str, BaseLLM] = {}
        self._default_name: Optional[str] = None
        self._metrics_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.register("mock", lambda cfg: MockLLM(cfg))

    def register(self, name: str, factory: LLMFactory) -> None:
        if name in self._providers:
            raise LLMError(f"LLM 提供者 {name} 已注册")
        self._providers[name] = factory

    def set_default(self, name: str) -> None:
        if name not in self._providers and name not in self._instances:
            raise LLMError(f"未找到 LLM 提供者: {name}")
        self._default_name = name

    def set_metrics_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._metrics_callback = callback

    async def get(self, name: Optional[str] = None, config: Optional[LLMConfig] = None) -> BaseLLM:
        target_name = name or self._default_name or "mock"
        if target_name in self._instances:
            return self._instances[target_name]

        factory = self._providers.get(target_name)
        if not factory:
            raise LLMError(f"未注册的 LLM: {target_name}")

        cfg = config or LLMConfig(model_name=target_name)
        instance = factory(cfg)
        if asyncio.iscoroutine(instance) or isinstance(instance, Awaitable):
            instance = await instance  # type: ignore[assignment]
        if not isinstance(instance, BaseLLM):
            raise LLMError(f"LLM 工厂 {target_name} 返回了无效实例")

        if self._metrics_callback:
            instance.set_metrics_callback(self._metrics_callback)

        await instance.initialize()
        self._instances[target_name] = instance
        return instance

    async def shutdown(self) -> None:
        for instance in self._instances.values():
            shutdown = getattr(instance, "shutdown", None)
            if callable(shutdown):
                await shutdown()
        self._instances.clear()

    @asynccontextmanager
    async def session(self, name: Optional[str] = None, config: Optional[LLMConfig] = None):
        llm = await self.get(name, config)
        try:
            yield llm
        finally:
            # session 模式不自动关闭，留给上层控制
            pass

