"""Agent端到端测试"""
import pytest
import asyncio
from src.agent import AdvancedAgent

class TestAgentE2E:
    @pytest.mark.e2e
    async def test_complete_conversation_flow(self):
        agent = AdvancedAgent("E2ETestAgent")
        
        # 模拟完整对话流程
        messages = [
            "你好", 
            "帮我计算 2+3", 
            "今天天气怎么样？", 
            "再见" 
        ]
        
        for msg in messages:
            response = await agent.process_message(msg)
            assert response is not None
            assert response.content is not None
        
        # 验证对话历史
        history = agent.get_conversation_history()
        assert len(history) == 8  # 4 user + 4 assistant
        print("✅ 端到端测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
