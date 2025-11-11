"""轻量级 Mock LLM，用于本地调试与单元测试。"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Union

from .base import BaseLLM
from ..core.types import LLMConfig


class MockLLM(BaseLLM):
    """返回可预测结果的轻量模型，方便无网络环境下运行。"""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        predefined_responses: dict[str, str] | None = None,
    ) -> None:
        super().__init__(config or LLMConfig(model_name="mock"))
        self._predefined_responses = predefined_responses or {}

    async def _initialize_model(self) -> None:  # noqa: D401
        # Mock 模型无需真实初始化，但保留异步接口以保持兼容
        await asyncio.sleep(0)

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        if prompt in self._predefined_responses:
            return self._predefined_responses[prompt]
        preview = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        return f"[mock reply] {preview}".strip()

    async def _generate_stream_impl(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> AsyncGenerator[str, None]:
        reply = await self._generate_impl(prompt, config)
        for chunk in reply.split():
            yield chunk
            await asyncio.sleep(0)

    async def _embed_impl(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        def _embed(text: str) -> List[float]:
            # 简单的 bag-of-chars 统计
            return [float(text.count(char)) for char in ("a", "e", "i", "o", "u")]

        if isinstance(texts, str):
            return _embed(texts)

        return [_embed(t) for t in texts]

