"""Agent组件集成测试"""
import pytest
import asyncio
from src.agent import AdvancedAgent
from src.memory import MemorySystem
from src.rag import RAGSystem
from src.tools import ToolSystem
from src.context import ContextManager

class TestAgentFullIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_components_integration(self):
        config = {
            "memory": {"max_memories": 100},
            "rag": {"max_documents": 1000},
            "tools": {"enable_all": True},
            "context": {"max_history": 10}
        }
        agent = AdvancedAgent("IntegrationTestAgent", config=config)

        # 测试组件集成
        assert agent.memory is not None
        assert agent.rag is not None
        assert agent.tools is not None
        assert agent.context is not None

        # 测试消息处理
        response = await agent.process_message("测试组件集成")
        assert response is not None
        print("✅ 集成测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
