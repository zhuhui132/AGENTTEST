"""记忆系统详细单元测试"""
import pytest
import asyncio
from datetime import datetime, timedelta
from src.memory import MemorySystem, MemoryType

class TestMemorySystem:
    """记忆系统测试类"""

    @pytest.fixture
    def memory_system(self):
        return MemorySystem()

    @pytest.mark.unit
    def test_add_memory_basic(self, memory_system):
        """测试基本添加记忆"""
        content = "这是一个测试记忆"
        memory_id = memory_system.add_memory(content)
        assert memory_id is not None
        assert len(memory_system.memories) == 1
        print("✅ 基本记忆添加测试通过")

    @pytest.mark.unit
    def test_retrieve_memory(self, memory_system):
        """测试记忆检索"""
        content = "Python编程"
        memory_id = memory_system.add_memory(content)
        results = memory_system.retrieve("Python")
        assert len(results) >= 1
        assert results[0]["memory_id"] == memory_id
        print("✅ 记忆检索测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
