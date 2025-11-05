"""Agent类综合单元测试"""
import pytest
import asyncio
from src.agent import AdvancedAgent, AgentFactory, AgentState, AgentMessage, AgentResponse

class TestAdvancedAgent:
    @pytest.mark.unit
    async def test_basic_functionality(self):
        agent = AdvancedAgent("TestAgent")
        response = await agent.process_message("你好")
        assert isinstance(response, AgentResponse)
        assert response.content is not None

if __name__ == "__main__":
    pytest.main([__file__])
