"""
大语言模型基础接口
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Awaitable, Callable, Union, List, Dict, Any, Optional, Sequence
import asyncio
import time
import logging

from ..core.interfaces import BaseLLM as IBaseLLM
from ..core.types import LLMConfig, ToolResult
from ..core.exceptions import LLMError, TimeoutError


class BaseLLM(IBaseLLM):
    """大语言模型基类"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._metrics_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def set_metrics_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """注册指标回调，供调用者收集性能数据"""
        self._metrics_callback = callback

    async def initialize(self) -> bool:
        """初始化模型"""
        try:
            self.logger.info("Initializing LLM: %s", self.model_name)
            await self._initialize_model()
            self._initialized = True
            self.logger.info("LLM initialized successfully: %s", self.model_name)
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to initialize LLM %s: %s", self.model_name, e)
            raise LLMError(f"LLM initialization failed: {e}") from e

    async def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> str:
        """生成文本"""
        if not self._initialized:
            raise LLMError("LLM not initialized")

        final_config = self._merge_configs(config)

        started_at = time.perf_counter()
        try:
            result = await self._generate_with_timeout(prompt, final_config)
            self._emit_metrics(
                "generate",
                {
                    "prompt_length": len(prompt),
                    "duration": time.perf_counter() - started_at,
                    "max_tokens": final_config.max_tokens,
                },
            )
            return result
        except asyncio.TimeoutError as exc:  # noqa: B904
            raise TimeoutError(f"Generation timeout for model {self.model_name}") from exc
        except Exception as e:  # noqa: BLE001
            self.logger.error("Generation failed: %s", e)
            raise LLMError(f"Generation failed: {e}") from e

    async def generate_batch(
        self,
        prompts: Sequence[str],
        config: Optional[LLMConfig] = None,
    ) -> List[str]:
        """批量生成文本，按顺序返回结果"""
        results: List[str] = []
        for prompt in prompts:
            results.append(await self.generate(prompt, config=config))
        return results

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        if not self._initialized:
            raise LLMError("LLM not initialized")

        final_config = self._merge_configs(config)

        try:
            async for chunk in self._generate_stream_with_timeout(prompt, final_config):
                yield chunk
        except Exception as e:  # noqa: BLE001
            self.logger.error("Stream generation failed: %s", e)
            raise LLMError(f"Stream generation failed: {e}") from e

    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """生成嵌入"""
        if not self._initialized:
            raise LLMError("LLM not initialized")

        started_at = time.perf_counter()
        try:
            result = await self._embed_with_timeout(texts)
            self._emit_metrics(
                "embed",
                {
                    "items": len(texts) if isinstance(texts, list) else 1,
                    "duration": time.perf_counter() - started_at,
                },
            )
            return result
        except Exception as e:  # noqa: BLE001
            self.logger.error("Embedding failed: %s", e)
            raise LLMError(f"Embedding failed: {e}") from e

    async def count_tokens(self, text: str) -> int:
        """计算token数量"""
        if not self._initialized:
            raise LLMError("LLM not initialized")

        try:
            return await self._count_tokens_impl(text)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Token counting failed: %s", e)
            return len(text) // 4  # 简单的回退估计

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "initialized": self._initialized,
            "config": self.config.__dict__,
        }

    @abstractmethod
    async def _initialize_model(self):
        """初始化模型的具体实现"""
        pass

    @abstractmethod
    async def _generate_impl(self, prompt: str, config: LLMConfig) -> str:
        """生成文本的具体实现"""
        pass

    @abstractmethod
    async def _generate_stream_impl(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        """流式生成的具体实现"""
        pass

    @abstractmethod
    async def _embed_impl(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """生成嵌入的具体实现"""
        pass

    async def _count_tokens_impl(self, text: str) -> int:
        """计算token数量的具体实现"""
        # 默认实现：简单字符计数
        return len(text)

    def _merge_configs(self, config: Optional[LLMConfig]) -> LLMConfig:
        """合并配置"""
        if config is None:
            return self.config

        # 创建新配置对象
        merged = LLMConfig()

        # 复制基础配置
        for field in self.config.__dataclass_fields__:
            setattr(merged, field, getattr(self.config, field))

        # 覆盖传入的配置
        for field in config.__dataclass_fields__:
            value = getattr(config, field)
            if value is not None:
                setattr(merged, field, value)

        return merged

    async def _generate_with_timeout(self, prompt: str, config: LLMConfig) -> str:
        """带超时的生成"""
        timeout = config.timeout or self.config.timeout

        if timeout <= 0:
            return await self._generate_impl(prompt, config)

        return await asyncio.wait_for(
            self._generate_impl(prompt, config),
            timeout=timeout
        )

    async def _generate_stream_with_timeout(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        """带超时的流式生成"""
        timeout = config.timeout or self.config.timeout

        if timeout <= 0:
            async for chunk in self._generate_stream_impl(prompt, config):
                yield chunk
            return

        async for chunk in asyncio.wait_for(
            self._generate_stream_impl(prompt, config),
            timeout=timeout
        ):
            yield chunk

    async def _embed_with_timeout(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """带超时的嵌入生成"""
        timeout = 30.0  # 嵌入操作默认超时时间

        return await asyncio.wait_for(
            self._embed_impl(texts),
            timeout=timeout
        )

    def _emit_metrics(self, event: str, payload: Dict[str, Any]) -> None:
        if self._metrics_callback:
            try:
                self._metrics_callback(event, payload)
            except Exception:  # noqa: BLE001
                self.logger.exception("Metrics callback failed for event %s", event)


class LLMWithTools(BaseLLM):
    """支持工具调用的LLM"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]] = None,
    ):
        super().__init__(config)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._tool_cache: Dict[str, Dict[str, Any]] = {}
        self._tool_executor = tool_executor

    def set_tool_executor(
        self,
        executor: Callable[[str, Dict[str, Any]], Awaitable[ToolResult]],
    ) -> None:
        """注入外部工具执行器（如 ToolSystem.call_tool_async）"""
        self._tool_executor = executor

    def register_tool(self, name: str, tool_func: Callable[..., Any], description: str = ""):
        """注册工具"""
        self.tools[name] = {
            "function": tool_func,
            "description": description,
            "parameters": self._extract_tool_params(tool_func),
        }
        self.logger.info("Tool registered: %s", name)

    def get_tools(self) -> Dict[str, Any]:
        """获取所有工具"""
        return self.tools

    async def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        config: Optional[LLMConfig] = None
    ) -> str:
        """带工具的生成"""
        available_tools: Dict[str, Dict[str, Any]] = {}
        if tools:
            for tool_name in tools:
                if tool_name in self.tools:
                    available_tools[tool_name] = self.tools[tool_name]
        else:
            available_tools = self.tools

        tool_calls = await self._parse_tool_calls(prompt, available_tools)
        tool_results = await self._execute_tool_calls(tool_calls)
        final_prompt = self._build_prompt_with_results(prompt, tool_calls, tool_results)
        return await self.generate(final_prompt, config)

    async def _parse_tool_calls(self, prompt: str, tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析工具调用"""
        tool_calls: List[Dict[str, Any]] = []
        for tool_name in tools.keys():
            if tool_name.lower() in prompt.lower():
                tool_calls.append(
                    {
                    "tool_name": tool_name,
                        "parameters": self._extract_parameters(prompt, tool_name),
                    }
                )
        return tool_calls

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        if not tool_calls:
            return []

        results: List[ToolResult] = []
        for call in tool_calls:
            results.append(await self._execute_tool_call(call))
        return results

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """执行工具调用"""
        tool_name = tool_call["tool_name"]
        parameters = tool_call.get("parameters", {})

        # 优先使用外部 ToolSystem
        if self._tool_executor:
            try:
                return await self._tool_executor(tool_name, parameters)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(tool_name=tool_name, success=False, error=str(exc))

        registry = self.tools.get(tool_name)
        if not registry:
            return ToolResult(tool_name=tool_name, success=False, error=f"Tool not found: {tool_name}")

        try:
            result = await self._execute_tool_function(registry["function"], parameters)
            return ToolResult(tool_name=tool_name, success=True, result=result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool_name=tool_name, success=False, error=str(exc))

    async def _execute_tool_function(self, func: Callable[..., Any], parameters: Dict[str, Any]) -> Any:
        """执行工具函数"""
        if asyncio.iscoroutinefunction(func):
            return await func(**parameters)
            return func(**parameters)

    def _build_prompt_with_results(
        self,
        original_prompt: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[ToolResult],
    ) -> str:
        """构建包含工具结果的提示"""
        if not tool_results:
            return original_prompt

        prompt = original_prompt + "\n\n工具执行结果:\n"
        for result in tool_results:
            prompt += f"{result.tool_name}: "
            if result.success:
                prompt += f"{result.result}\n"
            else:
                prompt += f"Error: {result.error}\n"

        return prompt + "\n\n基于以上信息，请回答原始问题。"

    def _extract_tool_params(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """提取工具参数"""
        return {}

    def _extract_parameters(self, prompt: str, tool_name: str) -> Dict[str, Any]:
        """从提示中提取参数"""
        return {}


class RateLimitedLLM(BaseLLM):
    """限流的LLM"""

    def __init__(self, base_llm: BaseLLM, max_requests_per_minute: int = 60):
        self.base_llm = base_llm
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str, config: Optional[LLMConfig] = None) -> str:
        """带限流的生成"""
        await self._check_rate_limit()
        return await self.base_llm.generate(prompt, config)

    async def generate_stream(self, prompt: str, config: Optional[LLMConfig] = None) -> AsyncGenerator[str, None]:
        """带限流的流式生成"""
        await self._check_rate_limit()
        async for chunk in self.base_llm.generate_stream(prompt, config):
            yield chunk

    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """带限流的嵌入"""
        await self._check_rate_limit()
        return await self.base_llm.embed(texts)

    async def _check_rate_limit(self):
        """检查限流"""
        async with self._lock:
            now = time.time()
            # 清理1分钟前的请求记录
            cutoff_time = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]

            # 检查是否超过限制
            if len(self.request_times) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                raise LLMError(f"Rate limit exceeded. Please wait {sleep_time:.1f} seconds.")

            # 记录当前请求
            self.request_times.append(now)

    async def initialize(self) -> bool:
        """初始化"""
        return await self.base_llm.initialize()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_llm.get_model_info()
        info["rate_limit"] = self.max_requests_per_minute
        return info
