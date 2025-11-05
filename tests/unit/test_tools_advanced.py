"""工具系统详细单元测试"""
import pytest
import asyncio
from src.tools import ToolSystem, ToolType

class TestToolSystem:
    """工具系统测试类"""

    @pytest.fixture
    def tool_system(self):
        return ToolSystem()

    @pytest.mark.unit
    def test_register_tool(self, tool_system):
        """测试注册工具"""
        def sample_function(x):
            return x * 2

        tool_id = tool_system.register_tool("test_tool", sample_function)
        assert tool_id is not None
        assert "test_tool" in tool_system.tools
        print("✅ 工具注册测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
