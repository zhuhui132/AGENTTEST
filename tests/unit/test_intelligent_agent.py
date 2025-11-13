"""
智能Agent单元测试

测试IntelligentAgent类的所有功能和边界条件。
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.agents.intelligent_agent import (
    IntelligentAgent, AgentState, ConversationContext
)
from src.core.types import (
    AgentConfig, LLMConfig, MemoryConfig, RAGConfig,
    AgentResponse, Message, ToolResult
)
from src.core.exceptions import (
    AgentError, ConfigurationError, ToolExecutionError
)
from src.core.interfaces import BaseLLM, BaseMemory, BaseRAG


class TestIntelligentAgent:
    """智能Agent测试类"""

    @pytest.fixture
    def basic_config(self):
        """基础配置"""
        return AgentConfig(
            name="test_agent",
            llm_config=LLMConfig(
                model_name="gpt-3.5-turbo",
                api_key="test_key",
                temperature=0.7,
                max_tokens=2048
            ),
            memory_enabled=True,
            rag_enabled=True,
            tools_enabled=True,
            memory_config=MemoryConfig(
                max_memories=1000,
                retrieval_limit=5
            ),
            rag_config=RAGConfig(
                max_documents=5000,
                retrieval_limit=5,
                similarity_threshold=0.7
            )
        )

    @pytest.fixture
    def minimal_config(self):
        """最小配置"""
        return AgentConfig(
            name="minimal_agent",
            llm_config=LLMConfig(
                model_name="mock",
                api_key="test_key"
            ),
            memory_enabled=False,
            rag_enabled=False,
            tools_enabled=False
        )

    @pytest.fixture
    async def mock_agent(self, minimal_config):
        """模拟Agent实例"""
        agent = IntelligentAgent(minimal_config)

        # 模拟LLM
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_llm.generate.return_value = Mock(
            content="测试响应",
            confidence=0.9,
            reasoning="测试推理",
            tool_calls=None
        )
        agent._llm = mock_llm

        # 设置为就绪状态
        agent._state = AgentState.READY

        yield agent

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, basic_config):
        """测试Agent初始化"""
        agent = IntelligentAgent(basic_config)

        # 检查初始状态
        assert agent.state == AgentState.UNINITIALIZED
        assert not agent.is_ready
        assert agent.config.name == "test_agent"

        # 检查统计信息
        stats = agent.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_agent_full_initialization(self, basic_config):
        """测试Agent完整初始化"""
        with patch('src.agents.intelligent_agent.LLMManager') as mock_llm_manager, \
             patch('src.agents.intelligent_agent.MemorySystem') as mock_memory_system, \
             patch('src.agents.intelligent_agent.RAGSystem') as mock_rag_system:

            # 模拟组件
            mock_llm_instance = AsyncMock()
            mock_memory_instance = AsyncMock()
            mock_rag_instance = AsyncMock()

            mock_llm_manager.return_value.create_llm.return_value = mock_llm_instance
            mock_memory_system.return_value = mock_memory_instance
            mock_rag_system.return_value = mock_rag_instance

            agent = IntelligentAgent(basic_config)
            await agent.initialize()

            # 验证状态
            assert agent.is_ready
            assert agent.state == AgentState.READY
            assert agent._llm is not None
            assert agent._memory is not None
            assert agent._rag is not None

    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, basic_config):
        """测试Agent初始化失败"""
        with patch('src.agents.intelligent_agent.LLMManager') as mock_llm_manager:
            # 模拟LLM初始化失败
            mock_llm_manager.return_value.create_llm.side_effect = Exception("LLM初始化失败")

            agent = IntelligentAgent(basic_config)

            with pytest.raises(ConfigurationError):
                await agent.initialize()

            assert agent.state == AgentState.ERROR
            assert not agent.is_ready

    @pytest.mark.asyncio
    async def test_process_text_message(self, mock_agent):
        """测试处理文本消息"""
        message = "你好，请介绍一下你的功能"

        response = await mock_agent.process_message(message)

        # 验证响应
        assert isinstance(response, AgentResponse)
        assert response.content == "测试响应"
        assert response.confidence == 0.9
        assert response.reasoning == "测试推理"
        assert response.conversation_id is not None
        assert response.message_id is not None
        assert "processing_time" in response.metadata

    @pytest.mark.asyncio
    async def test_process_message_object(self, mock_agent):
        """测试处理Message对象"""
        message = Message(
            content="测试消息",
            role="user",
            timestamp=time.time()
        )

        response = await mock_agent.process_message(message)

        assert isinstance(response, AgentResponse)
        assert response.content == "测试响应"

    @pytest.mark.asyncio
    async def test_process_message_with_conversation_id(self, mock_agent):
        """测试带对话ID的消息处理"""
        conversation_id = "test_conversation_123"
        message = "继续我们的对话"

        response = await mock_agent.process_message(
            message,
            conversation_id=conversation_id
        )

        assert response.conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_process_message_unready_state(self, basic_config):
        """测试未就绪状态下处理消息"""
        agent = IntelligentAgent(basic_config)
        # 不初始化，保持未就绪状态

        with pytest.raises(AgentError, match="Agent未准备就绪"):
            await agent.process_message("测试消息")

    @pytest.mark.asyncio
    async def test_process_message_with_tool_calls(self, mock_agent):
        """测试包含工具调用的消息处理"""
        # 模拟工具调用
        mock_agent._llm.generate.return_value = Mock(
            content="计算结果为16",
            confidence=0.9,
            reasoning="使用计算器工具",
            tool_calls=[
                {
                    "name": "calculator",
                    "arguments": {"expression": "4 * 4"}
                }
            ]
        )

        # 模拟工具
        mock_calculator = AsyncMock(return_value=16)
        await mock_agent.register_tool("calculator", mock_calculator)

        response = await mock_agent.process_message("计算4乘以4")

        assert response.content == "计算结果为16"
        assert "calculator" in str(response.metadata)

    @pytest.mark.asyncio
    async def test_process_message_tool_execution_error(self, mock_agent):
        """测试工具执行错误处理"""
        # 模拟工具调用
        mock_agent._llm.generate.return_value = Mock(
            content="计算失败",
            confidence=0.5,
            reasoning="工具执行失败",
            tool_calls=[
                {
                    "name": "nonexistent_tool",
                    "arguments": {}
                }
            ]
        )

        response = await mock_agent.process_message("执行不存在的工具")

        # 应该返回响应，但包含错误信息
        assert isinstance(response, AgentResponse)
        assert response.confidence < 1.0

    @pytest.mark.asyncio
    async def test_conversation_context_management(self, mock_agent):
        """测试对话上下文管理"""
        conversation_id = "test_context"

        # 第一条消息
        response1 = await mock_agent.process_message(
            "我是张三",
            conversation_id=conversation_id
        )

        # 第二条消息（应该保持上下文）
        response2 = await mock_agent.process_message(
            "我的名字是什么？",
            conversation_id=conversation_id
        )

        # 验证对话ID一致性
        assert response1.conversation_id == response2.conversation_id == conversation_id

        # 验证上下文存在
        assert conversation_id in mock_agent._contexts
        context = mock_agent._contexts[conversation_id]
        assert len(context.messages) >= 2

    @pytest.mark.asyncio
    async def test_memory_integration(self, basic_config):
        """测试记忆系统集成"""
        with patch('src.agents.intelligent_agent.MemorySystem') as mock_memory_class:
            # 模拟记忆系统
            mock_memory = AsyncMock()
            mock_memory.retrieve.return_value = [
                Mock(
                    content="用户之前询问过类似问题",
                    memory_type="episodic",
                    importance=0.8,
                    timestamp=time.time()
                )
            ]
            mock_memory_class.return_value = mock_memory

            # 模拟LLM
            with patch('src.agents.intelligent_agent.LLMManager') as mock_llm_manager:
                mock_llm = AsyncMock()
                mock_llm.generate.return_value = Mock(
                    content="基于记忆的回答",
                    confidence=0.9,
                    reasoning="使用了记忆信息",
                    tool_calls=None
                )
                mock_llm_manager.return_value.create_llm.return_value = mock_llm

                agent = IntelligentAgent(basic_config)
                await agent.initialize()

                response = await agent.process_message("我之前问过什么？")

                # 验证记忆被检索
                mock_memory.retrieve.assert_called_once()
                assert "memory_retrieved" in response.metadata

    @pytest.mark.asyncio
    async def test_rag_integration(self, basic_config):
        """测试RAG系统集成"""
        with patch('src.agents.intelligent_agent.RAGSystem') as mock_rag_class:
            # 模拟RAG系统
            mock_rag = AsyncMock()
            mock_rag.retrieve.return_value = [
                Mock(
                    content="相关文档内容",
                    title="测试文档",
                    score=0.85
                )
            ]
            mock_rag_class.return_value = mock_rag

            # 模拟LLM
            with patch('src.agents.intelligent_agent.LLMManager') as mock_llm_manager:
                mock_llm = AsyncMock()
                mock_llm.generate.return_value = Mock(
                    content="基于RAG的回答",
                    confidence=0.9,
                    reasoning="使用了检索到的文档",
                    tool_calls=None
                )
                mock_llm_manager.return_value.create_llm.return_value = mock_llm

                agent = IntelligentAgent(basic_config)
                await agent.initialize()

                response = await agent.process_message("相关主题是什么？")

                # 验证RAG被检索
                mock_rag.retrieve.assert_called_once()
                assert "rag_retrieved" in response.metadata

    @pytest.mark.asyncio
    async def test_tool_registration(self, mock_agent):
        """测试工具注册"""
        def dummy_tool(args):
            return "工具结果"

        await mock_agent.register_tool("dummy", dummy_tool)

        assert "dummy" in mock_agent._tools
        assert mock_agent._tools["dummy"] == dummy_tool
        assert "dummy" in mock_agent._tool_schemas

    @pytest.mark.asyncio
    async def test_tool_unregistration(self, mock_agent):
        """测试工具注销"""
        # 先注册工具
        def dummy_tool(args):
            return "工具结果"

        await mock_agent.register_tool("dummy", dummy_tool)
        assert "dummy" in mock_agent._tools

        # 注销工具
        await mock_agent.unregister_tool("dummy")
        assert "dummy" not in mock_agent._tools
        assert "dummy" not in mock_agent._tool_schemas

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_agent):
        """测试统计信息跟踪"""
        # 初始统计
        stats = mock_agent.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0

        # 处理消息
        await mock_agent.process_message("测试消息")

        # 更新统计
        stats = mock_agent.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0
        assert stats["average_response_time"] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agent):
        """测试错误处理"""
        # 模拟LLM异常
        mock_agent._llm.generate.side_effect = Exception("LLM错误")

        response = await mock_agent.process_message("触发错误的消息")

        # 验证错误响应
        assert isinstance(response, AgentResponse)
        assert "error" in response.metadata
        assert response.confidence == 0.0

        # 验证统计更新
        stats = mock_agent.get_stats()
        assert stats["failed_requests"] == 1

    @pytest.mark.asyncio
    async def test_health_check(self, mock_agent):
        """测试健康检查"""
        health = await mock_agent.health_check()

        assert "agent" in health
        assert "components" in health
        assert health["agent"]["status"] == AgentState.READY.value
        assert health["agent"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_agent):
        """测试资源清理"""
        # 模拟组件清理
        mock_agent._memory = AsyncMock()
        mock_agent._rag = AsyncMock()
        mock_agent._llm = AsyncMock()

        await mock_agent.cleanup()

        # 验证清理调用
        mock_agent._memory.cleanup.assert_called_once()
        mock_agent._rag.cleanup.assert_called_once()
        mock_agent._llm.cleanup.assert_called_once()

        # 验证状态重置
        assert mock_agent.state == AgentState.UNINITIALIZED
        assert len(mock_agent._contexts) == 0

    def test_conversation_context_creation(self):
        """测试对话上下文创建"""
        context = ConversationContext(
            conversation_id="test_123",
            user_id="user_456",
            session_id="session_789"
        )

        assert context.conversation_id == "test_123"
        assert context.user_id == "user_456"
        assert context.session_id == "session_789"
        assert len(context.messages) == 0
        assert len(context.context_variables) == 0
        assert context.start_time > 0
        assert context.last_activity > 0

    def test_agent_state_enum(self):
        """测试Agent状态枚举"""
        assert AgentState.UNINITIALIZED.value == "uninitialized"
        assert AgentState.INITIALIZING.value == "initializing"
        assert AgentState.READY.value == "ready"
        assert AgentState.PROCESSING.value == "processing"
        assert AgentState.ERROR.value == "error"
        assert AgentState.SHUTTING_DOWN.value == "shutting_down"

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, mock_agent):
        """测试并发消息处理"""
        messages = [
            f"消息 {i}" for i in range(5)
        ]

        # 并发处理
        tasks = [
            mock_agent.process_message(msg)
            for msg in messages
        ]

        responses = await asyncio.gather(*tasks)

        # 验证所有响应
        assert len(responses) == len(messages)
        for response in responses:
            assert isinstance(response, AgentResponse)
            assert response.content == "测试响应"

    @pytest.mark.asyncio
    async def test_message_with_extra_parameters(self, mock_agent):
        """测试带额外参数的消息处理"""
        message = "测试参数"
        user_id = "user_123"
        session_id = "session_456"

        response = await mock_agent.process_message(
            message,
            user_id=user_id,
            session_id=session_id
        )

        assert isinstance(response, AgentResponse)

        # 验证上下文包含额外参数
        context_id = response.conversation_id
        if context_id in mock_agent._contexts:
            context = mock_agent._contexts[context_id]
            assert context.user_id == user_id
            assert context.session_id == session_id

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, mock_agent):
        """测试空消息处理"""
        response = await mock_agent.process_message("")

        assert isinstance(response, AgentResponse)
        # 应该能正常处理空消息

    @pytest.mark.asyncio
    async def test_very_long_message_handling(self, mock_agent):
        """测试超长消息处理"""
        long_message = "测试" * 10000  # 4万字符

        response = await mock_agent.process_message(long_message)

        assert isinstance(response, AgentResponse)
        # 应该能正常处理长消息

    def test_id_generation(self, mock_agent):
        """测试ID生成"""
        conv_id1 = mock_agent._generate_conversation_id()
        conv_id2 = mock_agent._generate_conversation_id()
        msg_id1 = mock_agent._generate_message_id()
        msg_id2 = mock_agent._generate_message_id()

        # ID应该唯一
        assert conv_id1 != conv_id2
        assert msg_id1 != msg_id2
        assert conv_id1 != msg_id1
        assert conv_id2 != msg_id2

        # ID应该是字符串
        assert isinstance(conv_id1, str)
        assert isinstance(msg_id1, str)

    @pytest.mark.asyncio
    async def test_context_activity_update(self, mock_agent):
        """测试上下文活动时间更新"""
        conversation_id = "test_activity"

        # 第一条消息
        start_time = time.time()
        await mock_agent.process_message(
            "第一条消息",
            conversation_id=conversation_id
        )

        # 等待一小段时间
        await asyncio.sleep(0.1)

        # 第二条消息
        await mock_agent.process_message(
            "第二条消息",
            conversation_id=conversation_id
        )

        # 验证活动时间更新
        context = mock_agent._contexts[conversation_id]
        assert context.last_activity > start_time

    @pytest.mark.asyncio
    async def test_memory_storage_error_handling(self, mock_agent):
        """测试记忆存储错误处理"""
        # 模拟记忆系统
        mock_memory = AsyncMock()
        mock_memory.add_memory.side_effect = Exception("存储失败")
        mock_agent._memory = mock_memory

        # 处理消息（应该忽略存储错误）
        response = await mock_agent.process_message("测试消息")

        # 应该正常响应
        assert isinstance(response, AgentResponse)
        assert response.content == "测试响应"

    @pytest.mark.asyncio
    async def test_rag_retrieval_error_handling(self, mock_agent):
        """测试RAG检索错误处理"""
        # 模拟RAG系统
        mock_rag = AsyncMock()
        mock_rag.retrieve.side_effect = Exception("检索失败")
        mock_agent._rag = mock_rag

        # 处理消息（应该忽略检索错误）
        response = await mock_agent.process_message("测试消息")

        # 应该正常响应
        assert isinstance(response, AgentResponse)
        assert response.content == "测试响应"
