"""
Agent基础功能测试
测试Agent核心功能的基础用例
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.agents.agent import IntelligentAgent
from src.core.exceptions import AgentError
from src.core.types import AgentConfig


class TestAgentBasic:
    """Agent基础功能测试类"""

    @pytest.fixture
    def agent_config(self):
        """Agent配置fixture"""
        return AgentConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7
        )

    @pytest.fixture
    def mock_llm(self):
        """模拟LLM fixture"""
        llm = Mock()
        llm.generate = AsyncMock(return_value="测试响应")
        return llm

    @pytest.fixture
    def agent(self, agent_config, mock_llm):
        """Agent实例fixture"""
        return IntelligentAgent(config=agent_config, llm=mock_llm)

    def test_agent_initialization(self, agent_config, mock_llm):
        """测试Agent初始化"""
        agent = IntelligentAgent(config=agent_config, llm=mock_llm)

        assert agent.config == agent_config
        assert agent.llm == mock_llm
        assert agent.memory is not None
        assert agent.tools is not None

    @pytest.mark.asyncio
    async def test_basic_message_processing(self, agent):
        """测试基本消息处理"""
        message = "你好，请介绍一下自己"

        response = await agent.process_message(message)

        assert response.content == "测试响应"
        assert response.role == "assistant"
        assert response.finish_reason is not None

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, agent):
        """测试空消息处理"""
        message = ""

        with pytest.raises(AgentError):
            await agent.process_message(message)

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, agent):
        """测试并发消息处理"""
        messages = ["消息1", "消息2", "消息3"]

        tasks = [agent.process_message(msg) for msg in messages]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for response in responses:
            assert response.content == "测试响应"

    def test_agent_state_management(self, agent):
        """测试Agent状态管理"""
        # 初始状态
        assert agent.state == "idle"

        # 更新状态
        agent._update_state("processing")
        assert agent.state == "processing"

        # 重置状态
        agent._reset_state()
        assert agent.state == "idle"

    @pytest.mark.asyncio
    async def test_memory_integration(self, agent):
        """测试记忆系统集成"""
        message1 = "我的名字是张三"
        message2 = "我的名字是什么？"

        # 发送第一条消息
        response1 = await agent.process_message(message1)

        # 发送询问消息
        response2 = await agent.process_message(message2)

        assert "张三" in response2.content or "张三" in str(agent.memory.get_recent_messages())

    def test_tool_system_integration(self, agent):
        """测试工具系统集成"""
        tools = agent.get_available_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 0

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_llm):
        """测试错误处理"""
        # 模拟LLM错误
        mock_llm.generate.side_effect = Exception("LLM错误")

        with pytest.raises(AgentError):
            await agent.process_message("测试消息")

    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = AgentConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7
        )
        agent = IntelligentAgent(config=valid_config)
        assert agent.config.model_name == "test-model"

        # 无效配置
        with pytest.raises(AgentError):
            IntelligentAgent(config=None)

    @pytest.mark.asyncio
    async def test_message_history_tracking(self, agent):
        """测试消息历史跟踪"""
        messages = ["消息1", "消息2", "消息3"]

        for msg in messages:
            await agent.process_message(msg)

        history = agent.get_message_history()
        assert len(history) == len(messages) * 2  # 每条消息包含用户和助手响应

    def test_agent_reset(self, agent):
        """测试Agent重置"""
        # 添加一些状态
        agent._update_state("processing")
        agent.memory.add_message("测试消息", "测试响应")

        # 重置Agent
        agent.reset()

        # 验证重置
        assert agent.state == "idle"
        assert len(agent.memory.get_recent_messages()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
