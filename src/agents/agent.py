"""Agent 核心组件"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.exceptions import AgentError, LLMError, MemoryError, RAGError, ToolError
from ..core.interfaces import BaseAgent
from ..core.types import (
    AgentConfig,
    AgentMessage,
    AgentResponse,
    AgentState,
    ContextInfo,
    LLMConfig,
    MemoryConfig,
    MemoryItem,
    MessageType,
    PerformanceMetrics,
    RAGConfig,
    ToolResult,
)
from ..memory.memory import MemorySystem
from ..rag.rag import RAGSystem
from ..llm.base import BaseLLM, LLMWithTools
from ..llm.mock_llm import MockLLM
from ..utils.context import ContextManager
from ..utils.tools import ToolSystem

logger = logging.getLogger(__name__)


class IntelligentAgent(BaseAgent):
    """默认的智能 Agent 实现，整合记忆、RAG 与工具能力。"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        llm: Optional[BaseLLM] = None,
        llm_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        memory_system: Optional[MemorySystem] = None,
        rag_system: Optional[RAGSystem] = None,
        tools: Optional[ToolSystem] = None,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.session_id = str(uuid.uuid4())
        self.state = AgentState.INITIALIZING
        self.created_at = time.time()

        self._memory = memory_system or MemorySystem(memory_config)
        self._rag = rag_system or RAGSystem(rag_config)
        self._context = context_manager or ContextManager()
        self._tools = tools or ToolSystem()
        llm_cfg = llm_config or self.config.llm_config
        self._llm: BaseLLM = llm or MockLLM(llm_cfg)
        self._llm_ready = False
        self._llm_metrics: List[PerformanceMetrics] = []

        self._history: List[AgentMessage] = []
        self._latest_context: Optional[ContextInfo] = None

        self._llm.set_metrics_callback(self._on_llm_metrics)
        if isinstance(self._llm, LLMWithTools):
            self._llm.set_tool_executor(self._tools.call_tool_async)

        if tools is None:
            self._register_default_tools()

        self.state = AgentState.IDLE
        logger.debug("Agent 初始化完成，session=%s", self.session_id)

    async def process_message(
        self,
        message: AgentMessage | str,
        config: Optional[AgentConfig] = None,
    ) -> AgentResponse:
        if isinstance(message, str):
            user_message = AgentMessage(role="user", content=message.strip())
        else:
            user_message = message

        self._validate_message(user_message)
        user_message.metadata.setdefault("conversation_id", self.session_id)
        user_message.metadata.setdefault("timestamp", time.time())
        self._history.append(user_message)

        start_time = time.time()
        self.state = AgentState.PROCESSING
        self._llm_metrics = []

        try:
            context = await self._build_context()
            memories, documents, tool_results = await asyncio.gather(
                self._retrieve_memories(user_message, context),
                self._retrieve_documents(user_message, context),
                self._maybe_call_tools(user_message),
            )

            model_reply, prompt_metadata = await self._generate_model_reply(
                user_message,
                context,
                memories,
                documents,
                tool_results,
            )

            response_text = self._compose_response(
                user_message,
                context,
                memories,
                documents,
                tool_results,
                model_reply,
            )

            response = AgentResponse(
                content=response_text,
                reasoning=self._build_reasoning(memories, documents, tool_results),
                confidence=self._estimate_confidence(memories, documents, tool_results),
                sources=[doc.title for doc in documents],
                tool_calls=[result.__dict__ for result in tool_results],
                metadata={
                    "session_id": self.session_id,
                    "llm_prompt": prompt_metadata,
                    "llm_metrics": [metric.__dict__ for metric in self._llm_metrics[-3:]],
                },
            )

            response.processing_time = round(time.time() - start_time, 4)
            response.token_usage["prompt"] = len(user_message.content)
            response.token_usage["completion"] = len(response.content)

            assistant_msg = AgentMessage(
                role="assistant",
                content=response.content,
                message_type=MessageType.TEXT,
                metadata={"confidence": response.confidence},
            )
            self._history.append(assistant_msg)

            self.state = AgentState.IDLE
            return response
        except Exception as exc:  # noqa: BLE001
            self.state = AgentState.ERROR
            logger.exception("消息处理失败: %s", exc)
            raise AgentError(str(exc)) from exc

    async def initialize(self) -> bool:
        try:
            if not self._llm_ready:
                await self._llm.initialize()
                self._llm_ready = True
        except LLMError as exc:
            logger.warning("LLM 初始化失败，使用 fallback 策略: %s", exc)

        self.state = AgentState.IDLE
        return True

    async def shutdown(self) -> bool:
        await self._memory.cleanup()
        await self._rag.cleanup()
        await self._context.cleanup_contexts()
        shutdown = getattr(self._llm, "shutdown", None)
        if callable(shutdown):
            try:
                await shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.debug("关闭 LLM 时出现异常: %s", exc)
        self.state = AgentState.SHUTDOWN
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "history_length": len(self._history),
            "created_at": self.created_at,
        }

    def _validate_message(self, message: AgentMessage) -> None:
        if not message.content or not message.content.strip():
            raise AgentError("消息内容不能为空")
        if message.role not in {"user", "assistant", "system", "tool"}:
            raise AgentError(f"无效的角色: {message.role}")

    def _register_default_tools(self) -> None:
        async def register() -> None:
            await self._tools.register_callable(
                "calculator",
                self._calculator_tool,
                description="执行简单算术计算",
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
                                    "description": "算术表达式，例如 1+2*3",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                },
            )

        try:
            asyncio.run(register())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(register())
            loop.close()

    async def _build_context(self) -> ContextInfo:
        context = await self._context.build_context(
            self._history,
            self.config.max_context_length,
        )
        self._latest_context = context
        return context

    async def _retrieve_memories(
        self,
        message: AgentMessage,
        context: ContextInfo,
    ) -> List[MemoryItem]:
        if not self.config.memory_enabled:
            return []

        try:
            await self._memory.add_memory(
                content=message.content,
                importance=1.0,
                metadata={
                    "role": message.role,
                    "conversation_id": context.conversation_id,
                    "session_id": context.session_id,
                },
            )
            memories = await self._memory.retrieve(
                message.content,
                limit=self.config.max_concurrent_requests,
            )
            if not memories and message.role:
                metadata_hits = await self._memory.retrieve_by_metadata(
                    "role",
                    message.role,
                    limit=self.config.max_concurrent_requests,
                )
                return metadata_hits
            return memories
        except MemoryError as exc:
            logger.warning("记忆系统执行失败: %s", exc)
            return []

    async def _retrieve_documents(
        self,
        message: AgentMessage,
        context: ContextInfo,
    ):
        if not self.config.rag_enabled:
            return []
        try:
            filters = {}
            if context.key_entities:
                filters["tag"] = context.key_entities[0]
            if message.metadata.get("source"):
                filters["source"] = message.metadata["source"]
            return await self._rag.retrieve(
                message.content,
                limit=self.config.max_concurrent_requests,
                filters=filters or None,
            )
        except RAGError as exc:
            logger.warning("RAG 检索失败: %s", exc)
            return []

    async def _maybe_call_tools(self, message: AgentMessage) -> List[ToolResult]:
        if not self.config.tools_enabled:
            return []

        tool_plans = self._plan_tool_usage(message.content)
        if not tool_plans:
            return []

        try:
            results_raw = await self._tools.call_tools_parallel(tool_plans, return_exceptions=True)
        except ToolError as exc:
            logger.warning("工具批量调用失败: %s", exc)
            return []

        results: List[ToolResult] = []
        for item in results_raw:
            if isinstance(item, ToolResult):
                results.append(item)
            else:
                logger.warning("工具执行出现异常: %s", item)
        return results

    def _plan_tool_usage(self, content: str) -> List[Dict[str, Any]]:
        plans: List[Dict[str, Any]] = []
        lower = content.lower()
        if any(keyword in lower for keyword in ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"]):
            expression = content.replace("计算", "").strip()
            plans.append({"name": "calculator", "parameters": {"expression": expression or "0"}})
        return plans

    def _compose_response(
        self,
        message: AgentMessage,
        context: ContextInfo,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
        model_reply: str,
    ) -> str:
        parts = [model_reply.strip() or "抱歉，目前无法生成有效回复。"]

        if context.summary:
            parts.append(f"\n\n[上下文摘要]\n{context.summary}")
        if memories:
            samples = "；".join(mem.content[:60] for mem in memories[:3])
            parts.append(f"\n[相关记忆] {samples}")
        if documents:
            titles = "、".join(doc.title for doc in documents[:3])
            parts.append(f"\n[参考文档] {titles}")
        if tool_results:
            success_tools = [r.tool_name for r in tool_results if r.success]
            if success_tools:
                parts.append(f"\n[已调用工具] {', '.join(success_tools)}")

        return "".join(parts)

    def _generate_suggestion(
        self,
        message: AgentMessage,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> str:
        content = message.content.lower()
        if "天气" in content:
            return "请提供具体城市，我可以为您查询天气信息。"
        if "谢谢" in content:
            return "感谢您的反馈，很高兴能提供帮助。"
        if tool_results:
            result = tool_results[0]
            if result.success:
                return f"计算结果为：{result.result}"
        if documents:
            return f"建议阅读参考资料：{documents[0].title}。"
        if memories:
            return f"基于历史记录，建议关注：{memories[0].content[:50]}。"
        return "我已记录您的需求，请提供更多细节以便继续协助。"

    async def _generate_model_reply(
        self,
        message: AgentMessage,
        context: ContextInfo,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> Tuple[str, Dict[str, Any]]:
        prompt = self._build_prompt(message, context, memories, documents, tool_results)
        prompt_metadata = {
            "length": len(prompt),
            "context_tokens": len(context.summary),
            "memory_hits": len(memories),
            "document_hits": len(documents),
            "tool_calls": [result.tool_name for result in tool_results if result.success],
        }

        if not self._llm_ready:
            try:
                await self._llm.initialize()
                self._llm_ready = True
            except LLMError as exc:
                logger.warning("LLM 初始化失败，使用 fallback：%s", exc)
                return self._fallback_reply(message, memories, documents, tool_results), prompt_metadata

        try:
            reply = await self._llm.generate(prompt)
            return reply, prompt_metadata
        except LLMError as exc:
            logger.warning("LLM 生成失败，使用 fallback：%s", exc)
            return self._fallback_reply(message, memories, documents, tool_results), prompt_metadata

    def _build_prompt(
        self,
        message: AgentMessage,
        context: ContextInfo,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> str:
        sections = [
            "你是一名智能助手，请根据以下信息回答用户问题。",
            f"[用户消息]\n{message.content}",
        ]

        if context.summary:
            sections.append(f"[上下文摘要]\n{context.summary}")
        if memories:
            mem_lines = "\n".join(f"- {mem.content}" for mem in memories[:5])
            sections.append(f"[相关记忆]\n{mem_lines}")
        if documents:
            doc_lines = "\n".join(
                f"- {doc.title}: {getattr(doc, 'content', '')[:120]}"
                for doc in documents[:3]
            )
            sections.append(f"[知识库文档]\n{doc_lines}")
        if tool_results:
            result_lines = "\n".join(
                f"- {result.tool_name}: {result.result if result.success else result.error}"
                for result in tool_results
            )
            sections.append(f"[外部工具结果]\n{result_lines}")

        sections.append("请给出清晰、结构化的回答。")
        return "\n\n".join(sections)

    def _fallback_reply(
        self,
        message: AgentMessage,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> str:
        suggestion = self._generate_suggestion(message, memories, documents, tool_results)
        return f"（LLM 暂不可用，已使用规则回复）{suggestion}"

    def _build_reasoning(
        self,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> str:
        reasoning = []
        if memories:
            reasoning.append(f"引用记忆 {len(memories)} 条")
        if documents:
            reasoning.append(f"参考文档 {len(documents)} 篇")
        if tool_results:
            success = sum(1 for result in tool_results if result.success)
            reasoning.append(f"执行工具 {success}/{len(tool_results)} 次成功")
        return "; ".join(reasoning) if reasoning else "依据当前消息直接回复"

    def _estimate_confidence(
        self,
        memories: Sequence[MemoryItem],
        documents: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> float:
        confidence = 0.4
        confidence += min(len(memories), 3) * 0.05
        confidence += min(len(documents), 3) * 0.07
        confidence += sum(1 for result in tool_results if result.success) * 0.1
        return round(min(confidence, 0.95), 3)

    @staticmethod
    async def _calculator_tool(*, expression: str) -> float:
        safe_chars = "0123456789+-*/(). "
        if not set(expression) <= set(safe_chars):
            raise ToolError("表达式包含非法字符")
        return eval(expression, {"__builtins__": {}})  # noqa: S307

    def _on_llm_metrics(self, event: str, payload: Dict[str, Any]) -> None:
        metric = PerformanceMetrics(
            response_time=payload.get("duration", 0.0),
            token_usage={
                "prompt": payload.get("prompt_length", 0),
                "max_tokens": payload.get("max_tokens", 0),
            },
        )
        self._llm_metrics.append(metric)


class Agent:
    """向后兼容的同步包装器"""

    def __init__(
        self,
        name: str,
        *,
        session_id: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        config = AgentConfig(name=name)
        self._agent = IntelligentAgent(config=config, llm=llm, llm_config=llm_config)
        self.session_id = session_id or self._agent.session_id

    @property
    def name(self) -> str:
        return self._agent.config.name

    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in self._agent._history
        ]

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async def run() -> AgentResponse:
            agent_message = AgentMessage(role="user", content=message, metadata=context or {})
            return await self._agent.process_message(agent_message)

        response = asyncio.run(run())
        return {
            "response": response.content,
            "context": self._agent._latest_context.summary if self._agent._latest_context else "",
            "memories_used": [],
            "docs_used": response.sources,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "metadata": response.metadata,
        }

    def get_state(self) -> Dict[str, Any]:
        return self._agent.get_status()


__all__ = ["IntelligentAgent", "Agent"]
