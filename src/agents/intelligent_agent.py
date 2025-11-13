"""
智能Agent实现

这是系统的核心Agent类，集成了记忆系统、RAG检索、工具调用等完整功能。
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..core.interfaces import BaseAgent, BaseMemory, BaseRAG, BaseLLM
from ..core.types import (
    AgentResponse, Message, ToolResult, MemoryItem, RAGDocument,
    AgentConfig, LLMConfig, MemoryConfig, RAGConfig, ToolConfig
)
from ..core.exceptions import (
    AgentError, ConfigurationError, ToolExecutionError,
    MemoryError, RAGError, LLMError
)
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector


class AgentState(Enum):
    """Agent状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ConversationContext:
    """对话上下文"""
    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    context_variables: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentAgent(BaseAgent):
    """
    智能Agent类

    集成了完整AI能力的Agent，包括：
    - 多层次记忆系统
    - RAG检索能力
    - 灵活工具调用
    - 异步处理
    - 性能监控
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger(f"agent.{config.name}")
        self.metrics = MetricsCollector(f"agent.{config.name}")

        # 状态管理
        self._state = AgentState.UNINITIALIZED
        self._contexts: Dict[str, ConversationContext] = {}
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, dict] = {}

        # 核心组件（延迟初始化）
        self._llm: Optional[BaseLLM] = None
        self._memory: Optional[BaseMemory] = None
        self._rag: Optional[BaseRAG] = None

        # 性能统计
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "tool_calls": 0,
            "memory_retrievals": 0,
            "rag_retrievals": 0
        }

    @property
    def state(self) -> AgentState:
        """获取Agent状态"""
        return self._state

    @property
    def is_ready(self) -> bool:
        """检查Agent是否准备就绪"""
        return self._state == AgentState.READY

    async def initialize(self) -> None:
        """
        初始化Agent

        按顺序初始化所有组件：
        1. LLM配置验证
        2. 记忆系统
        3. RAG系统
        4. 工具系统
        """
        try:
            self._state = AgentState.INITIALIZING
            self.logger.info(f"开始初始化Agent: {self.config.name}")

            # 初始化LLM
            await self._initialize_llm()

            # 初始化记忆系统
            if self.config.memory_enabled:
                await self._initialize_memory()

            # 初始化RAG系统
            if self.config.rag_enabled:
                await self._initialize_rag()

            # 初始化工具系统
            if self.config.tools_enabled:
                await self._initialize_tools()

            self._state = AgentState.READY
            self.logger.info("Agent初始化完成")

        except Exception as e:
            self._state = AgentState.ERROR
            self.logger.error(f"Agent初始化失败: {e}")
            raise ConfigurationError(f"Agent初始化失败: {e}")

    async def _initialize_llm(self) -> None:
        """初始化LLM组件"""
        from ..llm import LLMManager

        llm_manager = LLMManager()
        self._llm = await llm_manager.create_llm(self.config.llm_config)
        await self._llm.initialize()
        self.logger.info(f"LLM初始化完成: {self.config.llm_config.model_name}")

    async def _initialize_memory(self) -> None:
        """初始化记忆系统"""
        from ..memory import MemorySystem

        self._memory = MemorySystem(self.config.memory_config)
        await self._memory.initialize()
        self.logger.info("记忆系统初始化完成")

    async def _initialize_rag(self) -> None:
        """初始化RAG系统"""
        from ..rag import RAGSystem

        self._rag = RAGSystem(self.config.rag_config)
        await self._rag.initialize()
        self.logger.info("RAG系统初始化完成")

    async def _initialize_tools(self) -> None:
        """初始化工具系统"""
        # 注册默认工具
        await self.register_default_tools()
        self.logger.info(f"工具系统初始化完成，已注册 {len(self._tools)} 个工具")

    async def register_default_tools(self) -> None:
        """注册默认工具"""
        from ..utils.tools import (
            CalculatorTool, WebSearchTool, FileTool,
            DateTimeTool, WeatherTool, MemoryTool
        )

        default_tools = [
            CalculatorTool(),
            WebSearchTool(),
            FileTool(),
            DateTimeTool(),
            WeatherTool(),
            MemoryTool(self._memory) if self._memory else None
        ]

        for tool in default_tools:
            if tool is not None:
                await self.register_tool(tool.name, tool)

    async def process_message(
        self,
        message: Union[str, Message],
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """
        处理用户消息

        Args:
            message: 用户消息（字符串或Message对象）
            conversation_id: 对话ID
            **kwargs: 额外参数

        Returns:
            AgentResponse: Agent响应结果
        """
        start_time = time.time()

        try:
            # 状态检查
            if not self.is_ready:
                raise AgentError("Agent未准备就绪")

            self._state = AgentState.PROCESSING
            self._stats["total_requests"] += 1

            # 消息标准化
            if isinstance(message, str):
                user_message = Message(
                    content=message,
                    role="user",
                    timestamp=time.time()
                )
            else:
                user_message = message

            # 获取或创建对话上下文
            context = await self._get_or_create_context(
                conversation_id, user_message, **kwargs
            )

            # 记录用户消息
            context.messages.append(user_message)
            context.last_activity = time.time()

            # 构建增强提示
            enhanced_prompt = await self._build_enhanced_prompt(
                user_message, context
            )

            # LLM推理
            llm_response = await self._llm.generate(enhanced_prompt)

            # 处理工具调用
            if llm_response.tool_calls:
                tool_results = await self._execute_tool_calls(
                    llm_response.tool_calls
                )
                # 可能需要基于工具结果进行第二次LLM调用
                final_response = await self._process_tool_results(
                    enhanced_prompt, tool_results
                )
            else:
                final_response = llm_response

            # 存储对话和记忆
            await self._store_conversation_memory(
                user_message, final_response, context
            )

            # 构建响应对象
            response = AgentResponse(
                content=final_response.content,
                conversation_id=context.conversation_id,
                message_id=self._generate_message_id(),
                reasoning=final_response.reasoning,
                confidence=final_response.confidence,
                metadata={
                    "processing_time": time.time() - start_time,
                    "tools_used": final_response.tool_calls,
                    "memory_retrieved": bool(context.context_variables.get("memory_items")),
                    "rag_retrieved": bool(context.context_variables.get("rag_documents"))
                }
            )

            # 更新统计
            self._update_stats(response, start_time)

            self._state = AgentState.READY
            self._stats["successful_requests"] += 1

            self.logger.info(f"消息处理完成，耗时: {response.processing_time:.2f}s")
            return response

        except Exception as e:
            self._state = AgentState.ERROR
            self._stats["failed_requests"] += 1
            self.logger.error(f"消息处理失败: {e}")

            # 返回错误响应
            return AgentResponse(
                content=f"抱歉，处理您的请求时遇到了错误: {str(e)}",
                conversation_id=conversation_id or "error",
                message_id=self._generate_message_id(),
                confidence=0.0,
                error=str(e),
                metadata={"processing_time": time.time() - start_time}
            )

    async def _build_enhanced_prompt(
        self,
        message: Message,
        context: ConversationContext
    ) -> str:
        """构建增强的提示词"""
        prompt_parts = [f"用户消息: {message.content}"]

        # 添加记忆信息
        if self._memory and self.config.memory_enabled:
            try:
                memory_items = await self._memory.retrieve(
                    message.content, limit=self.config.memory_config.retrieval_limit
                )
                if memory_items:
                    context.context_variables["memory_items"] = memory_items
                    memory_text = "\n".join([item.content for item in memory_items])
                    prompt_parts.append(f"相关记忆:\n{memory_text}")
                    self._stats["memory_retrievals"] += 1
            except Exception as e:
                self.logger.warning(f"记忆检索失败: {e}")

        # 添加RAG信息
        if self._rag and self.config.rag_enabled:
            try:
                rag_docs = await self._rag.retrieve(
                    message.content, limit=self.config.rag_config.retrieval_limit
                )
                if rag_docs:
                    context.context_variables["rag_documents"] = rag_docs
                    rag_text = "\n".join([doc.content for doc in rag_docs])
                    prompt_parts.append(f"相关文档:\n{rag_text}")
                    self._stats["rag_retrievals"] += 1
            except Exception as e:
                self.logger.warning(f"RAG检索失败: {e}")

        # 添加对话历史
        if len(context.messages) > 1:
            recent_messages = context.messages[-5:]  # 最近5条消息
            history = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages[:-1]
            ])
            prompt_parts.append(f"对话历史:\n{history}")

        # 添加工具描述
        if self._tools:
            tools_desc = "\n".join([
                f"- {name}: {self._tool_schemas.get(name, {}).get('description', '')}"
                for name in self._tools.keys()
            ])
            prompt_parts.append(f"可用工具:\n{tools_desc}")

        return "\n\n".join(prompt_parts)

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCallResult]:
        """执行工具调用"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            try:
                if tool_name not in self._tools:
                    raise ToolExecutionError(f"工具 '{tool_name}' 未注册")

                tool_func = self._tools[tool_name]

                # 同步工具包装为异步
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(tool_args)
                else:
                    result = tool_func(tool_args)

                results.append(ToolCallResult(
                    tool_name=tool_name,
                    success=True,
                    result=result
                ))

                self._stats["tool_calls"] += 1

            except Exception as e:
                results.append(ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e)
                ))

                self.logger.error(f"工具 '{tool_name}' 执行失败: {e}")

        return results

    async def _process_tool_results(
        self,
        original_prompt: str,
        tool_results: List[ToolCallResult]
    ) -> Any:
        """处理工具结果，可能需要二次LLM调用"""
        # 如果没有工具调用，直接返回
        if not tool_results:
            return await self._llm.generate(original_prompt)

        # 构建包含工具结果的提示
        tool_results_text = "\n".join([
            f"工具 '{result.tool_name}' 执行{'成功' if result.success else '失败'}: "
            f"{result.result if result.success else result.error}"
            for result in tool_results
        ])

        enhanced_prompt = f"""
原始提示:
{original_prompt}

工具执行结果:
{tool_results_text}

请基于以上信息和工具执行结果，给出最终的回答。
"""

        return await self._llm.generate(enhanced_prompt)

    async def _store_conversation_memory(
        self,
        user_message: Message,
        response: Any,
        context: ConversationContext
    ) -> None:
        """存储对话记忆"""
        if not self._memory or not self.config.memory_enabled:
            return

        try:
            # 存储用户消息
            await self._memory.add_memory(
                content=f"用户: {user_message.content}",
                memory_type="episodic",
                importance=0.7,
                tags=["user_message", context.conversation_id]
            )

            # 存储Agent响应
            if hasattr(response, 'content'):
                await self._memory.add_memory(
                    content=f"助手: {response.content}",
                    memory_type="episodic",
                    importance=0.7,
                    tags=["agent_response", context.conversation_id]
                )

        except Exception as e:
            self.logger.warning(f"存储对话记忆失败: {e}")

    async def _get_or_create_context(
        self,
        conversation_id: Optional[str],
        message: Message,
        **kwargs
    ) -> ConversationContext:
        """获取或创建对话上下文"""
        if conversation_id and conversation_id in self._contexts:
            context = self._contexts[conversation_id]
            context.last_activity = time.time()
            return context

        # 创建新上下文
        new_id = conversation_id or self._generate_conversation_id()
        context = ConversationContext(
            conversation_id=new_id,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            context_variables=kwargs
        )

        self._contexts[new_id] = context
        return context

    def _generate_conversation_id(self) -> str:
        """生成对话ID"""
        import uuid
        return str(uuid.uuid4())

    def _generate_message_id(self) -> str:
        """生成消息ID"""
        import uuid
        return str(uuid.uuid4())

    def _update_stats(self, response: AgentResponse, start_time: float) -> None:
        """更新性能统计"""
        processing_time = time.time() - start_time

        # 更新平均响应时间
        total_requests = self._stats["total_requests"]
        current_avg = self._stats["average_response_time"]
        new_avg = (current_avg * (total_requests - 1) + processing_time) / total_requests
        self._stats["average_response_time"] = new_avg

        # 记录指标
        self.metrics.record("response_time", processing_time)
        self.metrics.record("confidence", response.confidence)
        self.metrics.increment("total_requests")

    async def register_tool(self, name: str, tool: Callable) -> None:
        """注册工具"""
        self._tools[name] = tool

        # 获取工具模式描述
        if hasattr(tool, 'get_schema'):
            self._tool_schemas[name] = tool.get_schema()
        else:
            self._tool_schemas[name] = {
                "name": name,
                "description": getattr(tool, '__doc__', f"工具: {name}")
            }

    async def unregister_tool(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
        if name in self._tool_schemas:
            del self._tool_schemas[name]

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            **self._stats,
            "state": self._state.value,
            "registered_tools": len(self._tools),
            "active_contexts": len(self._contexts),
            "memory_enabled": self._memory is not None,
            "rag_enabled": self._rag is not None
        }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            self._state = AgentState.SHUTTING_DOWN

            # 清理对话上下文
            self._contexts.clear()

            # 清理核心组件
            if self._memory:
                await self._memory.cleanup()
            if self._rag:
                await self._rag.cleanup()
            if self._llm:
                await self._llm.cleanup()

            self._state = AgentState.UNINITIALIZED
            self.logger.info("Agent资源清理完成")

        except Exception as e:
            self.logger.error(f"Agent资源清理失败: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "agent": {
                "status": self._state.value,
                "healthy": self.is_ready,
                "uptime": time.time() - (self._stats.get("start_time", time.time()))
            },
            "components": {}
        }

        # 检查各组件健康状态
        if self._llm:
            try:
                llm_health = await self._llm.health_check()
                health_status["components"]["llm"] = llm_health
            except Exception as e:
                health_status["components"]["llm"] = {"status": "error", "error": str(e)}

        if self._memory:
            try:
                memory_health = await self._memory.health_check()
                health_status["components"]["memory"] = memory_health
            except Exception as e:
                health_status["components"]["memory"] = {"status": "error", "error": str(e)}

        if self._rag:
            try:
                rag_health = await self._rag.health_check()
                health_status["components"]["rag"] = rag_health
            except Exception as e:
                health_status["components"]["rag"] = {"status": "error", "error": str(e)}

        return health_status
