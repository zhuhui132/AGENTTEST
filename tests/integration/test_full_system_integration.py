"""
完整系统集成测试

测试整个系统的端到端功能和组件协作。
"""

import asyncio
import pytest
import time
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.agents.intelligent_agent import IntelligentAgent
from src.agents.multi_modal_agent import MultiModalAgent, MultimodalMessage, ModalityType
from src.llm.manager import LLMManager
from src.memory.memory import MemorySystem
from src.rag.rag import RAGSystem
from src.utils.tools import CalculatorTool, FileTool, MemoryTool
from src.core.types import (
    AgentConfig, LLMConfig, MemoryConfig, RAGConfig,
    AgentResponse, MultimodalContent
)
from src.core.exceptions import AgentError, ConfigurationError


class TestFullSystemIntegration:
    """完整系统集成测试类"""

    @pytest.fixture
    def full_config(self):
        """完整系统配置"""
        return AgentConfig(
            name="full_integration_agent",
            llm_config=LLMConfig(
                model_name="mock",
                api_key="test_key",
                temperature=0.7,
                max_tokens=2048,
                timeout=30.0
            ),
            memory_enabled=True,
            memory_config=MemoryConfig(
                max_memories=1000,
                retrieval_limit=5,
                importance_decay_rate=0.99
            ),
            rag_enabled=True,
            rag_config=RAGConfig(
                max_documents=5000,
                retrieval_limit=5,
                similarity_threshold=0.7,
                embedding_model="test-embedding"
            ),
            tools_enabled=True,
            vision_enabled=False,
            audio_enabled=False,
            fusion_enabled=True
        )

    @pytest.fixture
    async def integrated_agent(self, full_config):
        """集成Agent实例"""
        with patch('src.agents.intelligent_agent.LLMManager') as mock_llm_manager, \
             patch('src.agents.intelligent_agent.MemorySystem') as mock_memory_class, \
             patch('src.agents.intelligent_agent.RAGSystem') as mock_rag_class:

            # 模拟LLM
            mock_llm = AsyncMock()
            mock_llm.generate.return_value = Mock(
                content="系统集成测试响应",
                confidence=0.9,
                reasoning="通过集成系统生成",
                tool_calls=None
            )

            # 模拟记忆系统
            mock_memory = AsyncMock()
            mock_memory.retrieve.return_value = []
            mock_memory.add_memory.return_value = "memory_id"

            # 模拟RAG系统
            mock_rag = AsyncMock()
            mock_rag.retrieve.return_value = []

            mock_llm_manager.return_value.create_llm.return_value = mock_llm
            mock_memory_class.return_value = mock_memory
            mock_rag_class.return_value = mock_rag

            agent = IntelligentAgent(full_config)
            await agent.initialize()

            # 注入模拟组件
            agent._llm = mock_llm
            agent._memory = mock_memory
            agent._rag = mock_rag

            yield agent

            await agent.cleanup()

    @pytest.fixture
    async def multimodal_agent(self, full_config):
        """多模态Agent实例"""
        with patch('src.agents.multi_modal_agent.LLMManager') as mock_llm_manager, \
             patch('src.agents.multi_modal_agent.MemorySystem') as mock_memory_class, \
             patch('src.agents.multi_modal_agent.RAGSystem') as mock_rag_class, \
             patch('src.agents.multi_modal_agent.VisionModelFactory') as mock_vision_factory, \
             patch('src.agents.multi_modal_agent.AudioModelFactory') as mock_audio_factory:

            # 模拟组件
            mock_llm = AsyncMock()
            mock_llm.generate.return_value = Mock(
                content="多模态集成测试响应",
                confidence=0.85,
                reasoning="通过多模态集成系统生成",
                tool_calls=None
            )

            mock_memory = AsyncMock()
            mock_memory.retrieve.return_value = []

            mock_rag = AsyncMock()
            mock_rag.retrieve.return_value = []

            mock_vision = AsyncMock()
            mock_vision.analyze_image.return_value = "图像分析结果"

            mock_audio = AsyncMock()
            mock_audio.transcribe.return_value = "音频转录结果"

            mock_llm_manager.return_value.create_llm.return_value = mock_llm
            mock_memory_class.return_value = mock_memory
            mock_rag_class.return_value = mock_rag
            mock_vision_factory.create_model.return_value = mock_vision
            mock_audio_factory.create_model.return_value = mock_audio

            # 启用多模态功能
            config = AgentConfig(
                **full_config.__dict__,
                vision_enabled=True,
                audio_enabled=True
            )

            agent = MultiModalAgent(config)
            await agent.initialize()

            # 注入模拟组件
            agent._llm = mock_llm
            agent._memory = mock_memory
            agent._rag = mock_rag
            agent._vision_model = mock_vision
            agent._audio_model = mock_audio

            yield agent

            await agent.cleanup()

    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, integrated_agent):
        """测试完整对话流程"""
        conversation_id = "integration_test_conversation"

        # 第一轮对话
        response1 = await integrated_agent.process_message(
            "你好，我想了解人工智能",
            conversation_id=conversation_id
        )

        assert isinstance(response1, AgentResponse)
        assert response1.content == "系统集成测试响应"
        assert response1.conversation_id == conversation_id

        # 第二轮对话（应该保持上下文）
        response2 = await integrated_agent.process_message(
            "具体是机器学习的应用有哪些？",
            conversation_id=conversation_id
        )

        assert isinstance(response2, AgentResponse)
        assert response2.conversation_id == conversation_id

        # 验证对话上下文
        context = integrated_agent._contexts.get(conversation_id)
        assert context is not None
        assert len(context.messages) >= 2

    @pytest.mark.asyncio
    async def test_memory_integration(self, integrated_agent):
        """测试记忆系统集成"""
        # 模拟记忆检索
        integrated_agent._memory.retrieve.return_value = [
            Mock(
                content="用户之前询问过AI相关话题",
                memory_type="episodic",
                importance=0.8,
                tags=["ai", "history"],
                timestamp=time.time()
            )
        ]

        response = await integrated_agent.process_message(
            "继续我们上次关于AI的讨论"
        )

        # 验证记忆被使用
        integrated_agent._memory.retrieve.assert_called()
        assert "memory_retrieved" in response.metadata

    @pytest.mark.asyncio
    async def test_rag_integration(self, integrated_agent):
        """测试RAG系统集成"""
        # 模拟RAG检索
        integrated_agent._rag.retrieve.return_value = [
            Mock(
                content="AI是计算机科学的一个分支",
                title="人工智能基础",
                score=0.9,
                source="knowledge_base.pdf"
            )
        ]

        response = await integrated_agent.process_message(
            "请解释人工智能的定义"
        )

        # 验证RAG被使用
        integrated_agent._rag.retrieve.assert_called()
        assert "rag_retrieved" in response.metadata

    @pytest.mark.asyncio
    async def test_tool_integration(self, integrated_agent):
        """测试工具系统集成"""
        # 注册计算器工具
        calculator = CalculatorTool()
        await integrated_agent.register_tool("calculator", calculator.execute)

        # 模拟工具调用
        integrated_agent._llm.generate.return_value = Mock(
            content="计算结果是25",
            confidence=0.95,
            reasoning="使用计算器计算5*5",
            tool_calls=[
                {
                    "name": "calculator",
                    "arguments": {"expression": "5 * 5"}
                }
            ]
        )

        response = await integrated_agent.process_message("计算5乘以5")

        assert response.content == "计算结果是25"
        assert "tools_used" in response.metadata

    @pytest.mark.asyncio
    async def test_multimodal_text_processing(self, multimodal_agent):
        """测试多模态文本处理"""
        message = MultimodalMessage(
            content=[
                MultimodalContent(
                    type=ModalityType.TEXT,
                    data="请分析这段文本"
                )
            ],
            role="user",
            timestamp=time.time()
        )

        response = await multimodal_agent.process_multimodal_message(message)

        assert isinstance(response, AgentResponse)
        assert response.content == "多模态集成测试响应"
        assert response.metadata["primary_modality"] == ModalityType.TEXT.value

    @pytest.mark.asyncio
    async def test_multimodal_image_processing(self, multimodal_agent):
        """测试多模态图像处理"""
        message = MultimodalMessage(
            content=[
                MultimodalContent(
                    type=ModalityType.IMAGE,
                    data=b"fake_image_data"
                ),
                MultimodalContent(
                    type=ModalityType.TEXT,
                    data="请分析这张图片"
                )
            ],
            role="user",
            timestamp=time.time()
        )

        response = await multimodal_agent.process_multimodal_message(message)

        assert isinstance(response, AgentResponse)
        assert response.metadata["primary_modality"] == ModalityType.IMAGE.value
        assert "image" in response.metadata["processed_modalities"]

    @pytest.mark.asyncio
    async def test_multimodal_audio_processing(self, multimodal_agent):
        """测试多模态音频处理"""
        message = MultimodalMessage(
            content=[
                MultimodalContent(
                    type=ModalityType.AUDIO,
                    data=b"fake_audio_data"
                )
            ],
            role="user",
            timestamp=time.time()
        )

        response = await multimodal_agent.process_multimodal_message(message)

        assert isinstance(response, AgentResponse)
        assert response.metadata["primary_modality"] == ModalityType.AUDIO.value
        assert "audio" in response.metadata["processed_modalities"]

    @pytest.mark.asyncio
    async def test_multimodal_fusion_processing(self, multimodal_agent):
        """测试多模态融合处理"""
        message = MultimodalMessage(
            content=[
                MultimodalContent(type=ModalityType.TEXT, data="结合图片内容"),
                MultimodalContent(type=ModalityType.IMAGE, data=b"image_data"),
                MultimodalContent(type=ModalityType.AUDIO, data=b"audio_data")
            ],
            role="user",
            timestamp=time.time()
        )

        response = await multimodal_agent.process_multimodal_message(message)

        assert isinstance(response, AgentResponse)
        assert response.metadata["processing_mode"] == "multi_modal"
        processed_modalities = response.metadata["processed_modalities"]
        assert "text" in processed_modalities
        assert "image" in processed_modalities
        assert "audio" in processed_modalities

    @pytest.mark.asyncio
    async def test_concurrent_conversation_handling(self, integrated_agent):
        """测试并发对话处理"""
        conversations = [
            f"conversation_{i}" for i in range(5)
        ]

        tasks = [
            integrated_agent.process_message(
                f"消息 {i} 对话 {conv}",
                conversation_id=conv
            )
            for i, conv in enumerate(conversations)
        ]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == len(conversations)

        # 验证每个对话都有独立的上下文
        for i, conv in enumerate(conversations):
            assert responses[i].conversation_id == conv
            assert conv in integrated_agent._contexts

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, integrated_agent):
        """测试系统健康监控"""
        health = await integrated_agent.health_check()

        assert "agent" in health
        assert "components" in health

        # 验证Agent状态
        assert health["agent"]["status"] == "ready"
        assert health["agent"]["healthy"] is True

        # 验证组件健康状态
        assert "llm" in health["components"]
        assert "memory" in health["components"]
        assert "rag" in health["components"]

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integrated_agent):
        """测试错误处理集成"""
        # 模拟LLM错误
        integrated_agent._llm.generate.side_effect = Exception("LLM集成错误")

        response = await integrated_agent.process_message("触发错误的消息")

        # 验证错误处理
        assert isinstance(response, AgentResponse)
        assert "error" in response.metadata
        assert response.confidence == 0.0

        # 验证统计更新
        stats = integrated_agent.get_stats()
        assert stats["failed_requests"] > 0

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integrated_agent):
        """测试性能监控集成"""
        # 执行多个请求
        start_time = time.time()

        for i in range(3):
            await integrated_agent.process_message(f"性能测试消息 {i}")

        total_time = time.time() - start_time

        # 验证性能统计
        stats = integrated_agent.get_stats()
        assert stats["total_requests"] >= 3
        assert stats["successful_requests"] >= 3
        assert stats["average_response_time"] > 0

        # 验证每个请求的处理时间
        assert total_time > 0

    @pytest.mark.asyncio
    async def test_tool_execution_integration(self, integrated_agent):
        """测试工具执行集成"""
        # 注册文件工具
        with tempfile.TemporaryDirectory() as temp_dir:
            file_tool = FileTool(base_path=temp_dir)
            await integrated_agent.register_tool("file_tool", file_tool.execute)

            # 模拟工具调用序列
            integrated_agent._llm.generate.side_effect = [
                Mock(  # 第一次调用：创建文件
                    content="文件创建成功",
                    confidence=0.9,
                    tool_calls=[
                        {
                            "name": "file_tool",
                            "arguments": {
                                "operation": "write",
                                "path": "test.txt",
                                "content": "测试内容"
                            }
                        }
                    ]
                ),
                Mock(  # 第二次调用：基于工具结果的响应
                    content="基于文件创建的回答",
                    confidence=0.9,
                    tool_calls=None
                )
            ]

            response = await integrated_agent.process_message(
                "请创建一个测试文件"
            )

            assert "文件创建成功" in response.content or "基于文件创建的回答" in response.content

    @pytest.mark.asyncio
    async def test_memory_storage_integration(self, integrated_agent):
        """测试记忆存储集成"""
        response = await integrated_agent.process_message(
            "请记住我喜欢编程",
            user_id="test_user"
        )

        # 验证记忆存储被调用
        integrated_agent._memory.add_memory.assert_called()

        # 验证对话上下文包含用户信息
        contexts = [
            ctx for ctx in integrated_agent._contexts.values()
            if ctx.user_id == "test_user"
        ]
        assert len(contexts) > 0

    @pytest.mark.asyncio
    async def test_context_persistence_integration(self, integrated_agent):
        """测试上下文持久化集成"""
        conversation_id = "context_persistence_test"

        # 多轮对话
        messages = [
            "我是张三",
            "我从事软件开发",
            "我最喜欢的编程语言是Python",
            "我的专长是什么？"
        ]

        responses = []
        for msg in messages:
            response = await integrated_agent.process_message(
                msg,
                conversation_id=conversation_id
            )
            responses.append(response)

        # 验证上下文持久化
        context = integrated_agent._contexts.get(conversation_id)
        assert context is not None
        assert len(context.messages) == len(messages)

        # 验证所有响应都有相同的对话ID
        for response in responses:
            assert response.conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_resource_cleanup_integration(self, integrated_agent):
        """测试资源清理集成"""
        # 执行一些操作
        await integrated_agent.process_message("测试消息")

        # 验证组件存在
        assert integrated_agent._llm is not None
        assert integrated_agent._memory is not None
        assert integrated_agent._rag is not None

        # 清理资源
        await integrated_agent.cleanup()

        # 验证状态重置
        assert integrated_agent.state.value == "uninitialized"

    @pytest.mark.asyncio
    async def test_configuration_integration(self, integrated_agent):
        """测试配置集成"""
        # 验证配置正确应用
        assert integrated_agent.config.name == "full_integration_agent"
        assert integrated_agent.config.memory_enabled is True
        assert integrated_agent.config.rag_enabled is True
        assert integrated_agent.config.tools_enabled is True

        # 验证组件配置
        assert integrated_agent.config.memory_config.max_memories == 1000
        assert integrated_agent.config.rag_config.max_documents == 5000
        assert integrated_agent.config.llm_config.temperature == 0.7

    @pytest.mark.asyncio
    async def test_advanced_multimodal_scenarios(self, multimodal_agent):
        """测试高级多模态场景"""
        # 复杂的多模态消息
        message = MultimodalMessage(
            content=[
                MultimodalContent(
                    type=ModalityType.TEXT,
                    data="请分析这个场景"
                ),
                MultimodalContent(
                    type=ModalityType.IMAGE,
                    data=b"complex_image_data"
                ),
                MultimodalContent(
                    type=ModalityType.AUDIO,
                    data=b"background_audio"
                )
            ],
            role="user",
            timestamp=time.time()
        )

        response = await multimodal_agent.process_multimodal_message(
            message,
            processing_mode=multimodal_agent.ProcessingMode.MULTI_MODAL
        )

        # 验证多模态处理
        assert isinstance(response, AgentResponse)
        assert len(response.metadata["processed_modalities"]) == 3

        # 验证统计更新
        stats = multimodal_agent.get_multimodal_stats()
        assert stats["multimodal"]["image_processed"] > 0
        assert stats["multimodal"]["audio_processed"] > 0

    @pytest.mark.asyncio
    async def test_system_scalability(self, integrated_agent):
        """测试系统可扩展性"""
        # 大量并发请求
        num_requests = 20
        tasks = [
            integrated_agent.process_message(f"并发消息 {i}")
            for i in range(num_requests)
        ]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        # 验证所有请求都成功处理
        assert len(responses) == num_requests
        for response in responses:
            assert isinstance(response, AgentResponse)

        # 验证性能
        total_time = end_time - start_time
        avg_time = total_time / num_requests

        stats = integrated_agent.get_stats()
        assert stats["total_requests"] >= num_requests
        assert stats["successful_requests"] >= num_requests

        # 平均响应时间应该合理
        assert avg_time < 5.0  # 每个请求平均不超过5秒

    @pytest.mark.asyncio
    async def test_system_resilience(self, integrated_agent):
        """测试系统韧性"""
        # 模拟部分组件失败
        original_retrieve = integrated_agent._memory.retrieve

        async def failing_retrieve(*args, **kwargs):
            if integrated_agent._memory.retrieve.call_count % 3 == 0:
                raise Exception("模拟记忆检索失败")
            return await original_retrieve(*args, **kwargs)

        integrated_agent._memory.retrieve.side_effect = failing_retrieve

        # 执行多个请求，部分应该失败
        success_count = 0
        total_requests = 10

        for i in range(total_requests):
            try:
                response = await integrated_agent.process_message(f"韧性测试 {i}")
                if response.confidence > 0:
                    success_count += 1
            except Exception:
                pass

        # 验证系统仍能处理部分请求
        assert success_count > total_requests // 2  # 至少一半成功

        # 验证错误统计
        stats = integrated_agent.get_stats()
        assert stats["failed_requests"] > 0

    def test_integration_test_configuration(self):
        """测试集成测试配置验证"""
        config = AgentConfig(
            name="integration_test",
            llm_config=LLMConfig(model_name="test", api_key="test"),
            memory_enabled=True,
            rag_enabled=True,
            tools_enabled=True
        )

        # 验证配置完整性
        assert config.name is not None
        assert config.llm_config is not None
        assert config.memory_config is not None
        assert config.rag_config is not None

        # 验证布尔配置
        assert isinstance(config.memory_enabled, bool)
        assert isinstance(config.rag_enabled, bool)
        assert isinstance(config.tools_enabled, bool)
