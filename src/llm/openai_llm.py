"""OpenAI LLM 适配层。

该实现仅依赖 Python 标准库，通过 HTTP API 调用 OpenAI 兼容的接口。
如需使用官方 SDK，可在 `_generate_impl` 内替换为对应客户端。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Optional

try:
    import aiohttp
except ImportError as _exc:  # pragma: no cover - 环境缺少 aiohttp 时的降级路径
    aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_IMPORT_ERROR = _exc
else:  # pragma: no cover - 正常路径不需要覆盖率
    _AIOHTTP_IMPORT_ERROR = None

from .base import BaseLLM
from ..core.exceptions import LLMError
from ..core.types import LLMConfig

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class OpenAILLM(BaseLLM):
    """简化的 OpenAI 接入实现，兼容任何 OpenAI API 格式的服务端。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: Optional[str] = None,
        endpoint: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        super().__init__(config)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._endpoint = endpoint or DEFAULT_OPENAI_ENDPOINT
        self._model_alias = model_alias or config.model_name
        self._session: "aiohttp.ClientSession | None" = None

    async def _initialize_model(self) -> None:
        if aiohttp is None:
            raise LLMError(f"缺少 aiohttp 依赖，无法初始化 OpenAILLM: {_AIOHTTP_IMPORT_ERROR}")
        if not self._api_key:
            raise LLMError("OPENAI_API_KEY 未设置，无法初始化 OpenAILLM")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        if not self._session:
            raise LLMError("OpenAILLM 尚未初始化")

        payload = {
            "model": self._model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stop": config.stop_sequences or None,
        }

        async with self._session.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise LLMError(f"OpenAI 请求失败: {resp.status} {detail}")

            data = await resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:  # noqa: B904
                raise LLMError(f"OpenAI 响应解析失败: {data}") from exc

    async def _generate_stream_impl(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> AsyncGenerator[str, None]:
        if not self._session:
            raise LLMError("OpenAILLM 尚未初始化")

        payload = {
            "model": self._model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        async with self._session.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise LLMError(f"OpenAI 请求失败: {resp.status} {detail}")

            async for line in resp.content:
                chunk = line.decode("utf-8").strip()
                if not chunk or chunk == "data: [DONE]":
                    continue
                if chunk.startswith("data:"):
                    chunk = chunk[len("data:") :].strip()
                try:
                    payload = json.loads(chunk)
                    delta = payload["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue

    async def _embed_impl(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        # OpenAI embedding 接口依赖另一个 endpoint，这里给出最小实现
        if not self._session:
            raise LLMError("OpenAILLM 尚未初始化")

        if isinstance(texts, str):
            texts = [texts]

        payload: Dict[str, object] = {
            "model": self._model_alias,
            "input": texts,
        }

        async with self._session.post(
            self._endpoint.replace("chat/completions", "embeddings"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise LLMError(f"OpenAI Embeddings 请求失败: {resp.status} {detail}")

            data = await resp.json()
            vectors = [item["embedding"] for item in data.get("data", [])]
            return vectors[0] if len(vectors) == 1 else vectors

    async def shutdown(self) -> None:
        if aiohttp is None:
            return
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

