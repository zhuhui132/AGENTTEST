"""Agent性能测试"""
import pytest
import asyncio
import time
import psutil
from src.agent import AdvancedAgent

class TestAgentPerformance:
    @pytest.mark.performance
    async def test_response_time(self):
        agent = AdvancedAgent("PerfTestAgent")
        
        start_time = time.time()
        response = await agent.process_message("性能测试消息")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # 响应时间应小于5秒
        print(f"✅ 响应时间: {response_time:.2f}s")

if __name__ == "__main__":
    pytest.main([__file__])
