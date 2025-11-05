"""
Agent核心组件 - 增强版实现
"""
from typing import Dict, List, Optional, Any, Union, Callable
import uuid
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .memory import MemorySystem
from .rag import RAGSystem
from .tools import ToolSystem
from .context import ContextManager
from .metrics import AgentMetrics

# 设置日志
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent状态枚举"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    THINKING = "thinking"
    RESPONDING = "responding"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentMessage:
    """消息数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # user, assistant, system, tool
    content: str = ""
    message_type: str = "text"  # text, image, audio, file
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    token_count: int = 0
    processing_time: float = 0.0

@dataclass
class AgentResponse:
    """Agent响应结构"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 检索和记忆信息
    retrieved_memories: List[Dict] = field(default_factory=list)
    retrieved_documents: List[Dict] = field(default_factory=list)
    used_tools: List[Dict] = field(default_factory=list)

    # 上下文信息
    context_summary: str = ""
    conversation_turn: int = 0

    # 性能指标
    token_count: int = 0
    step_count: int = 0

class AdvancedAgent:
    """高级Agent类 - 完整实现"""

    def __init__(
        self,
        name: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化Agent"""
        self.name = self._validate_name(name)
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.config = config or {}

        # 初始化状态
        self.state = AgentState.INITIALIZING
        self.conversation_history: List[AgentMessage] = []
        self.conversation_turn = 0

        # 核心组件初始化
        self._init_components()

        # 初始化指标系统
        self.metrics = AgentMetrics(self.session_id)

        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            "on_message_received": [],
            "on_processing_start": [],
            "on_processing_complete": [],
            "on_error": []
        }

        # 状态变为初始化完成
        self.state = AgentState.IDLE
        logger.info(f"Agent '{self.name}' 初始化完成，session_id: {self.session_id}")

    def _init_components(self):
        """初始化核心组件"""
        try:
            # 记忆系统
            memory_config = self.config.get("memory", {})
            self.memory = MemorySystem(memory_config)

            # RAG系统
            rag_config = self.config.get("rag", {})
            self.rag = RAGSystem(rag_config)

            # 工具系统
            tools_config = self.config.get("tools", {})
            self.tools = ToolSystem(tools_config)

            # 上下文管理
            context_config = self.config.get("context", {})
            self.context = ContextManager(context_config)

            logger.info("所有核心组件初始化成功")

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"组件初始化失败: {str(e)}")
            raise RuntimeError(f"Agent组件初始化失败: {str(e)}")

    def _validate_name(self, name: str) -> str:
        """验证Agent名称"""
        if not name or not name.strip():
            raise ValueError("Agent名称不能为空")

        name = name.strip()

        if len(name) > 100:
            raise ValueError("Agent名称长度不能超过100字符")

        # 检查特殊字符
        import re
        if not re.match(r'^[a-zA-Z0-9\u4e00-\u9fa5_\-\s]+$', name):
            raise ValueError("Agent名称只能包含中英文、数字、下划线和连字符")

        return name

    async def process_message(
        self,
        message: Union[str, AgentMessage],
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """处理用户消息 - 异步版本"""
        if self.state == AgentState.SHUTDOWN:
            raise RuntimeError("Agent已关闭")

        if self.state == AgentState.ERROR:
            raise RuntimeError("Agent处于错误状态")

        # 创建消息对象
        if isinstance(message, str):
            user_message = AgentMessage(
                role="user",
                content=message,
                metadata=context or {}
            )
        else:
            user_message = message

        # 验证消息
        self._validate_message(user_message)

        # 触发消息接收回调
        await self._trigger_callbacks("on_message_received", user_message)

        # 添加到对话历史
        self.conversation_history.append(user_message)
        self.conversation_turn += 1

        try:
            self.state = AgentState.PROCESSING
            processing_start = datetime.now()

            # 触发处理开始回调
            await self._trigger_callbacks("on_processing_start", user_message)

            # 执行完整的消息处理流程
            response = await self._process_message_internal(user_message, context, options)

            # 添加处理时间
            processing_time = (datetime.now() - processing_start).total_seconds()
            response.processing_time = processing_time
            response.conversation_turn = self.conversation_turn

            # 计算token数量
            response.token_count = self._count_tokens(response.content)

            # 添加到对话历史
            assistant_message = AgentMessage(
                role="assistant",
                content=response.content,
                message_type="text",
                metadata={
                    "processing_time": processing_time,
                    "confidence": response.confidence,
                    "used_memories": len(response.retrieved_memories),
                    "used_documents": len(response.retrieved_documents),
                    "used_tools": len(response.used_tools)
                }
            )
            self.conversation_history.append(assistant_message)

            # 更新指标
            await self.metrics.record_processing_event(
                user_message, response, processing_time
            )

            # 触发处理完成回调
            await self._trigger_callbacks("on_processing_complete", response)

            self.state = AgentState.IDLE
            logger.info(f"消息处理完成，耗时: {processing_time:.2f}s")

            return response

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"消息处理失败: {str(e)}")

            # 触发错误回调
            await self._trigger_callbacks("on_error", e)

            # 创建错误响应
            return AgentResponse(
                content=f"抱歉，处理您的消息时出现错误: {str(e)}",
                reasoning="系统内部错误",
                confidence=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )

    async def _process_message_internal(
        self,
        message: AgentMessage,
        context: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> AgentResponse:
        """内部消息处理流程"""
        response = AgentResponse()

        # 1. 上下文构建
        self.state = AgentState.THINKING
        context_info = await self._build_context(message, context)
        response.context_summary = context_info.get("summary", "")

        # 2. 并行检索
        retrieval_tasks = [
            self.memory.retrieve(message.content),
            self.rag.retrieve(message.content)
        ]

        retrieved_memories, retrieved_documents = await asyncio.gather(*retrieval_tasks)
        response.retrieved_memories = retrieved_memories
        response.retrieved_documents = retrieved_documents

        # 3. 工具调用规划
        tools_to_call = await self._plan_tools(message, context_info, options)

        # 4. 工具执行
        if tools_to_call:
            tool_results = await self._execute_tools(tools_to_call)
            response.used_tools = tool_results
            response.reasoning += f"执行了{len(tool_results)}个工具调用; "

        # 5. 生成响应
        self.state = AgentState.RESPONDING
        response.content = await self._generate_response(
            message, context_info, response
        )

        # 6. 后处理
        response.content = await self._post_process_response(response.content, options)
        response.confidence = await self._calculate_confidence(response, message)

        return response

    async def _build_context(
        self,
        message: AgentMessage,
        external_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建上下文信息"""
        try:
            # 获取最近的历史消息
            recent_history = self.conversation_history[-10:]  # 最近10条

            # 构建上下文
            context_info = await self.context.build_context(
                recent_history, external_context, self.session_id
            )

            # 添加元数据
            context_info.update({
                "current_message": message.content,
                "message_type": message.message_type,
                "session_id": self.session_id,
                "agent_name": self.name,
                "conversation_turn": self.conversation_turn
            })

            return context_info

        except Exception as e:
            logger.error(f"上下文构建失败: {str(e)}")
            return {
                "summary": f"处理消息: {message.content}",
                "context": {"error": str(e)}
            }

    async def _plan_tools(
        self,
        message: AgentMessage,
        context_info: Dict[str, Any],
        options: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """规划需要调用的工具"""
        try:
            # 分析消息中的工具调用意图
            tools_needed = []

            # 检查是否需要计算器
            if any(keyword in message.content.lower()
                  for keyword in ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"]):
                tools_needed.append({
                    "name": "calculator",
                    "priority": "high",
                    "reason": "检测到数学计算需求"
                })

            # 检查是否需要时间信息
            if any(keyword in message.content.lower()
                  for keyword in ["时间", "日期", "今天", "明天"]):
                tools_needed.append({
                    "name": "time_query",
                    "priority": "medium",
                    "reason": "检测到时间查询需求"
                })

            # 检查是否需要天气信息
            if any(keyword in message.content.lower()
                  for keyword in ["天气", "气温", "下雨", "晴天"]):
                tools_needed.append({
                    "name": "weather_query",
                    "priority": "medium",
                    "reason": "检测到天气查询需求"
                })

            # 排序优先级
            tools_needed.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)

            return tools_needed

        except Exception as e:
            logger.error(f"工具规划失败: {str(e)}")
            return []

    async def _execute_tools(self, tools_to_call: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用"""
        results = []

        for tool in tools_to_call:
            try:
                result = await self.tools.call_tool_async(
                    tool["name"],
                    {"query": tool.get("query", "")}
                )

                results.append({
                    "tool_name": tool["name"],
                    "success": True,
                    "result": result,
                    "reason": tool.get("reason", ""),
                    "execution_time": result.get("execution_time", 0.0)
                })

            except Exception as e:
                logger.error(f"工具 {tool['name']} 调用失败: {str(e)}")
                results.append({
                    "tool_name": tool["name"],
                    "success": False,
                    "error": str(e),
                    "reason": tool.get("reason", ""),
                    "execution_time": 0.0
                })

        return results

    async def _generate_response(
        self,
        message: AgentMessage,
        context_info: Dict[str, Any],
        response: AgentResponse
    ) -> str:
        """生成响应内容"""
        try:
            # 收集所有信息用于生成响应
            generation_context = {
                "user_message": message.content,
                "context_summary": response.context_summary,
                "retrieved_memories": response.retrieved_memories,
                "retrieved_documents": response.retrieved_documents,
                "tool_results": response.used_tools,
                "conversation_turn": response.conversation_turn,
                "agent_name": self.name
            }

            # 构建响应模板
            response_parts = []

            # 基础问候或确认
            if response.conversation_turn == 1:
                response_parts.append(f"您好！我是{self.name}，很高兴为您服务。")
            else:
                response_parts.append("我理解了您的问题。")

            # 基于检索的信息
            if response.retrieved_memories:
                response_parts.append(f"我找到了{len(response.retrieved_memories)}条相关的记忆信息。")

            if response.retrieved_documents:
                response_parts.append(f"我参考了{len(response.retrieved_documents)}篇相关文档。")

            # 基于工具结果
            if response.used_tools:
                for tool_result in response.used_tools:
                    if tool_result["success"]:
                        response_parts.append(f"通过{tool_result['tool_name']}获得了相关信息。")
                    else:
                        response_parts.append(f"调用{tool_result['tool_name']}时遇到了问题。")

            # 生成核心响应
            core_response = await self._generate_core_response(generation_context)
            response_parts.append(core_response)

            # 添加结束语
            response_parts.append("如果您还有其他问题，请随时告诉我。")

            return " ".join(response_parts)

        except Exception as e:
            logger.error(f"响应生成失败: {str(e)}")
            return f"抱歉，生成响应时遇到问题: {str(e)}"

    async def _generate_core_response(self, context: Dict[str, Any]) -> str:
        """生成核心响应内容"""
        # 简化的响应生成逻辑
        user_message = context["user_message"]

        # 基于用户消息生成响应
        if "你好" in user_message.lower() or "hi" in user_message.lower():
            return "很高兴与您交流，请问有什么可以帮助您的吗？"
        elif "谢谢" in user_message.lower() or "感谢" in user_message.lower():
            return "不客气！这是我应该做的。"
        elif "再见" in user_message.lower() or "拜拜" in user_message.lower():
            return "再见！期待下次与您交流。"
        elif "?" in user_message or "？" in user_message or "什么" in user_message or "how" in user_message.lower():
            return "这是一个很好的问题。让我来帮您分析和解答。"
        else:
            return f"关于您提到的'{user_message}'，我会尽力为您提供帮助。"

    async def _post_process_response(
        self,
        content: str,
        options: Optional[Dict[str, Any]]
    ) -> str:
        """后处理响应内容"""
        if not options:
            return content

        # 格式化处理
        if options.get("format") == "markdown":
            content = f"```\n{content}\n```"
        elif options.get("format") == "html":
            content = f"<p>{content}</p>"

        # 语言处理
        if options.get("language") == "en":
            # 简单的英文化处理（这里应该是更复杂的逻辑）
            content = content.replace("您好", "Hello").replace("谢谢", "Thank you")
        elif options.get("language") == "zh":
            # 简单的中文化处理
            content = content.replace("Hello", "您好").replace("Thank you", "谢谢")

        return content

    async def _calculate_confidence(
        self,
        response: AgentResponse,
        message: AgentMessage
    ) -> float:
        """计算响应置信度"""
        confidence = 0.5  # 基础置信度

        # 基于检索结果增加置信度
        if response.retrieved_memories:
            confidence += 0.1 * min(len(response.retrieved_memories), 3) / 3

        if response.retrieved_documents:
            confidence += 0.15 * min(len(response.retrieved_documents), 5) / 5

        # 基于工具结果增加置信度
        successful_tools = sum(1 for t in response.used_tools if t["success"])
        if response.used_tools:
            confidence += 0.2 * (successful_tools / len(response.used_tools))

        # 基于上下文匹配度
        if response.context_summary and len(response.context_summary) > 10:
            confidence += 0.1

        # 基于响应长度
        response_length = len(response.content)
        if 50 <= response_length <= 500:
            confidence += 0.05

        return min(confidence, 1.0)

    def _validate_message(self, message: AgentMessage):
        """验证消息"""
        if not message or not message.content or not message.content.strip():
            raise ValueError("消息内容不能为空")

        if len(message.content) > 10000:
            raise ValueError("消息内容长度不能超过10000字符")

        # 验证角色
        valid_roles = ["user", "assistant", "system", "tool"]
        if message.role not in valid_roles:
            raise ValueError(f"无效的消息角色: {message.role}")

    def _count_tokens(self, text: str) -> int:
        """计算token数量（简化实现）"""
        # 简化的token计算，实际应该使用tokenizer
        return len(text.split()) + len(text) // 4

    async def _trigger_callbacks(self, event_name: str, data: Any):
        """触发回调函数"""
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {str(e)}")

    def add_callback(self, event_name: str, callback: Callable):
        """添加回调函数"""
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []

        self.callbacks[event_name].append(callback)

    def remove_callback(self, event_name: str, callback: Callable):
        """移除回调函数"""
        if event_name in self.callbacks:
            try:
                self.callbacks[event_name].remove(callback)
            except ValueError:
                pass

    def get_state(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "name": self.name,
            "session_id": self.session_id,
            "state": self.state.value,
            "conversation_turn": self.conversation_turn,
            "conversation_count": len(self.conversation_history),
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }

    def get_conversation_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        history = self.conversation_history

        if limit:
            history = history[-limit:]

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "message_type": msg.message_type,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
                "token_count": msg.token_count,
                "processing_time": msg.processing_time
            }
            for msg in history
        ]

    async def shutdown(self):
        """关闭Agent"""
        self.state = AgentState.SHUTDOWN

        # 清理资源
        if hasattr(self, 'memory'):
            await self.memory.cleanup()

        if hasattr(self, 'rag'):
            await self.rag.cleanup()

        if hasattr(self, 'tools'):
            await self.tools.cleanup()

        if hasattr(self, 'context'):
            await self.context.cleanup()

        logger.info(f"Agent '{self.name}' 已关闭")

    # 同步版本的方法
    def process_message_sync(
        self,
        message: Union[str, AgentMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """同步处理消息"""
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_message(message, context)
            )
        finally:
            loop.close()

class AgentFactory:
    """Agent工厂类"""

    @staticmethod
    def create_agent(
        agent_type: str,
        name: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AdvancedAgent:
        """创建不同类型的Agent"""

        if agent_type == "basic":
            # 基础配置
            default_config = {
                "memory": {"max_memories": 1000},
                "rag": {"max_documents": 10000},
                "tools": {"enable_all": True},
                "context": {"max_history": 10}
            }
            default_config.update(config or {})
            return AdvancedAgent(name, session_id, default_config)

        elif agent_type == "conversational":
            # 对话Agent配置
            default_config = {
                "memory": {"max_memories": 2000, "weight_decay": True},
                "rag": {"similarity_threshold": 0.7},
                "tools": {"enable_calculator": True, "enable_time_query": True},
                "context": {"context_window": 2048, "include_summaries": True}
            }
            default_config.update(config or {})
            return AdvancedAgent(name, session_id, default_config)

        elif agent_type == "task_agent":
            # 任务Agent配置
            default_config = {
                "memory": {"task_focused": True, "priority_weighting": True},
                "rag": {"task_relevant_only": True},
                "tools": {"enable_all": True, "task_execution": True},
                "context": {"goal_tracking": True, "progress_monitoring": True}
            }
            default_config.update(config or {})
            return AdvancedAgent(name, session_id, default_config)

        else:
            raise ValueError(f"不支持的Agent类型: {agent_type}")

# 向后兼容的Agent类
class Agent(AdvancedAgent):
    """向后兼容的Agent类"""

    def __init__(self, name: str, session_id: Optional[str] = None):
        # 使用基础配置
        basic_config = {
            "memory": {"max_memories": 1000},
            "rag": {"max_documents": 10000},
            "tools": {"enable_all": True},
            "context": {"max_history": 10}
        }
        super().__init__(name, session_id, basic_config)

    def process_message(self, message: str, context: Optional[Dict] = None) -> Dict:
        """同步处理消息 - 向后兼容接口"""
        response = self.process_message_sync(message, context)

        return {
            "response": response.content,
            "context": response.context_summary,
            "memories_used": response.retrieved_memories,
            "docs_used": response.retrieved_documents,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "metadata": response.metadata
        }
