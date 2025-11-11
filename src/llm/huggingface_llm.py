"""HuggingFace Hub 文本生成/推理适配器。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Optional

try:
    import aiohttp
except ImportError as _exc:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_IMPORT_ERROR = _exc

from .base import BaseLLM
from ..core.exceptions import LLMError
from ..core.types import LLMConfig

DEFAULT_HF_ENDPOINT = "https://api-inference.huggingface.co/models/{model}"


class HuggingFaceLLM(BaseLLM):
    """通过 HuggingFace Inference API 调用文本生成/嵌入。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_token: Optional[str] = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(config)
        self._api_token = api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._endpoint_template = endpoint or DEFAULT_HF_ENDPOINT
        self._session: "aiohttp.ClientSession | None" = None

    async def _initialize_model(self) -> None:
        if aiohttp is None:
            raise LLMError(f"缺少 aiohttp 依赖，无法初始化 HuggingFaceLLM: {_AIOHTTP_IMPORT_ERROR}")  # type: ignore[name-defined]
        if not self._api_token:
            raise LLMError("HUGGINGFACEHUB_API_TOKEN 未设置，无法初始化 HuggingFaceLLM")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        if not self._session:
            raise LLMError("HuggingFaceLLM 尚未初始化")

        endpoint = self._endpoint_template.format(model=config.model_name)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "return_full_text": False,
            },
        }

        async with self._session.post(
            endpoint,
            headers={"Authorization": f"Bearer {self._api_token}"},
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise LLMError(f"HuggingFace 请求失败: {resp.status} {detail}")

            data = await resp.json()
            try:
                if isinstance(data, list):
                    return data[0]["generated_text"]
                return data["generated_text"]
            except (KeyError, IndexError, TypeError) as exc:  # noqa: B904
                raise LLMError(f"HuggingFace 响应解析失败: {data}") from exc

    async def _generate_stream_impl(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> AsyncGenerator[str, None]:
        # HuggingFace Inference API 暂不提供流式接口，退化为单次调用
        reply = await self._generate_impl(prompt, config)
        for chunk in reply.split():
            yield chunk
            await asyncio.sleep(0)

    async def _embed_impl(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        if not self._session:
            raise LLMError("HuggingFaceLLM 尚未初始化")

        if isinstance(texts, str):
            texts = [texts]

        endpoint = self._endpoint_template.format(model=self.config.model_name) + "/embeddings"
        payload: Dict[str, object] = {"inputs": texts}

        async with self._session.post(
            endpoint,
            headers={"Authorization": f"Bearer {self._api_token}"},
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise LLMError(f"HuggingFace Embeddings 请求失败: {resp.status} {detail}")

            data = await resp.json()
            vectors = data.get("embeddings", [])
            return vectors[0] if len(vectors) == 1 else vectors

    async def shutdown(self) -> None:
        if aiohttp is None:
            return
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

