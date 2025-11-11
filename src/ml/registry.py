"""通用模型注册器，可在测试环境中快速切换策略。"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional


class ModelRegistry:
    """维护模型名称与可调用对象的映射。"""

    def __init__(self) -> None:
        self._models: Dict[str, Callable[..., Any]] = {}
        self._default: Optional[str] = None

    def register(self, name: str, model: Callable[..., Any], *, replace: bool = False) -> None:
        if name in self._models and not replace:
            raise ValueError(f"模型 {name} 已注册")
        self._models[name] = model

    def unregister(self, name: str) -> None:
        self._models.pop(name, None)

    def has_model(self, name: str) -> bool:
        return name in self._models

    def set_default(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"模型 {name} 未注册")
        self._default = name

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        model_name = name or self._default
        if not model_name:
            raise KeyError("尚未注册默认模型")
        if model_name not in self._models:
            raise KeyError(f"模型 {model_name} 未注册")

        result = self._models[model_name](*args, **kwargs)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)  # type: ignore[arg-type]
            else:
                if loop.is_running():
                    raise RuntimeError("检测到正在运行的事件循环，请改用 invoke_async")
                return loop.run_until_complete(result)  # type: ignore[arg-type]
        return result

    async def invoke_async(self, name: str, *args: Any, **kwargs: Any) -> Any:
        model_name = name or self._default
        if not model_name:
            raise KeyError("尚未注册默认模型")
        if model_name not in self._models:
            raise KeyError(f"模型 {model_name} 未注册")

        result = self._models[model_name](*args, **kwargs)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            return await result  # type: ignore[return-value]
        return result

    def available_models(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._models)

