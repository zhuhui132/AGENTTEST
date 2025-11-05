"""详细集成测试"""
import pytest
import asyncio
from src.agent import AdvancedAgent
from src.memory import MemorySystem
from src.rag import RAGSystem
from src.tools import ToolSystem

class TestAgentIntegration:
    """Agent组件集成测试"""

    @pytest.mark.integration
    async def test_complete_agent_workflow(self):
        """测试完整Agent工作流"""
        config = {
            "memory": {"max_memories": 100},
            "rag": {"max_documents": 1000},
            "tools": {"enable_all": True}
        }
        agent = AdvancedAgent("IntegrationTestAgent", config=config)
        
        response = await agent.process_message("测试完整工作流")
        assert response is not None
        print("✅ 完整Agent工作流测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
