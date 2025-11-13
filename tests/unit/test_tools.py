"""
工具系统单元测试

测试各种工具的功能和边界条件。
"""

import asyncio
import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch, AsyncMock

from src.utils.tools import (
    CalculatorTool, WebSearchTool, FileTool, DateTimeTool,
    WeatherTool, MemoryTool, ToolCategory, ToolMetadata
)
from src.core.types import ToolResult
from src.core.exceptions import ToolExecutionError


class TestCalculatorTool:
    """计算器工具测试"""

    @pytest.fixture
    def calculator(self):
        """计算器工具实例"""
        return CalculatorTool()

    def test_tool_metadata(self, calculator):
        """测试工具元数据"""
        assert calculator.metadata.name == "calculator"
        assert calculator.metadata.category == ToolCategory.CALCULATION
        assert "math" in calculator.metadata.tags
        assert calculator.metadata.async_capable is True

    def test_get_schema(self, calculator):
        """测试工具模式"""
        schema = calculator.get_schema()
        assert schema.name == "calculator"
        assert "expression" in schema.parameters["properties"]
        assert "expression" in schema.parameters["required"]

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator):
        """测试基础算术运算"""
        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5),
            ("2 ** 3", 8),
            ("10 % 3", 1)
        ]

        for expression, expected in test_cases:
            result = await calculator.execute({"expression": expression})

            assert result.success is True
            assert result.result == expected
            assert result.tool_name == "calculator"
            assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_math_functions(self, calculator):
        """测试数学函数"""
        test_cases = [
            ("sqrt(16)", 4.0),
            ("abs(-5)", 5),
            ("round(3.14159, 2)", 3.14),
            ("min(1, 5, 3)", 1),
            ("max(1, 5, 3)", 5),
            ("pow(2, 3)", 8),
            ("sin(0)", 0),
            ("cos(0)", 1),
            ("log(1)", 0),
            ("exp(0)", 1),
            ("pi", 3.141592653589793),
            ("e", 2.718281828459045)
        ]

        for expression, expected in test_cases:
            result = await calculator.execute({"expression": expression})

            assert result.success is True
            assert abs(result.result - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_empty_expression(self, calculator):
        """测试空表达式"""
        result = await calculator.execute({"expression": ""})

        assert result.success is False
        assert "表达式不能为空" in result.error
        assert result.tool_name == "calculator"

    @pytest.mark.asyncio
    async def test_unsafe_expression(self, calculator):
        """测试不安全表达式"""
        unsafe_expressions = [
            "import os",
            "__import__('os')",
            "eval('1+1')",
            "exec('print(1)')",
            "open('test.txt')",
            "file('test.txt')",
            "while True: pass",
            "for i in range(10): pass",
            "def test(): pass",
            "class Test: pass",
            "lambda x: x",
            "yield 1",
            "return 1",
            "break",
            "continue"
        ]

        for expression in unsafe_expressions:
            result = await calculator.execute({"expression": expression})

            assert result.success is False
            assert "不安全的内容" in result.error

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        """测试复杂表达式"""
        expression = "sqrt(16) + log(10) * sin(pi/2) - abs(-5)"
        result = await calculator.execute({"expression": expression})

        assert result.success is True
        assert isinstance(result.result, (int, float))

    @pytest.mark.asyncio
    async def test_syntax_error(self, calculator):
        """测试语法错误"""
        result = await calculator.execute({"expression": "2 + + 3"})

        assert result.success is False
        assert "计算错误" in result.error


class TestWebSearchTool:
    """网页搜索工具测试"""

    @pytest.fixture
    def web_search(self):
        """搜索工具实例"""
        return WebSearchTool()

    def test_tool_metadata(self, web_search):
        """测试工具元数据"""
        assert web_search.metadata.name == "web_search"
        assert web_search.metadata.category == ToolCategory.SEARCH
        assert "search" in web_search.metadata.tags

    def test_get_schema(self, web_search):
        """测试工具模式"""
        schema = web_search.get_schema()
        assert schema.name == "web_search"
        assert "query" in schema.parameters["required"]
        assert "limit" in schema.parameters["properties"]

    @pytest.mark.asyncio
    async def test_basic_search(self, web_search):
        """测试基础搜索"""
        params = {
            "query": "人工智能",
            "limit": 5
        }

        result = await web_search.execute(params)

        assert result.success is True
        assert isinstance(result.result, list)
        assert len(result.result) <= 5
        assert result.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_empty_query(self, web_search):
        """测试空查询"""
        result = await web_search.execute({"query": ""})

        assert result.success is False
        assert "搜索查询不能为空" in result.error

    @pytest.mark.asyncio
    async def test_search_with_source(self, web_search):
        """测试指定搜索源"""
        params = {
            "query": "机器学习",
            "source": "academic",
            "limit": 3
        }

        result = await web_search.execute(params)

        assert result.success is True
        assert result.metadata["source"] == "academic"
        assert result.metadata["limit"] == 3

    @pytest.mark.asyncio
    async def test_search_result_structure(self, web_search):
        """测试搜索结果结构"""
        params = {"query": "Python"}
        result = await web_search.execute(params)

        if result.success and result.result:
            first_result = result.result[0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result
            assert "relevance_score" in first_result


class TestFileTool:
    """文件工具测试"""

    @pytest.fixture
    def file_tool(self):
        """文件工具实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield FileTool(base_path=temp_dir)

    def test_tool_metadata(self, file_tool):
        """测试工具元数据"""
        assert file_tool.metadata.name == "file_tool"
        assert file_tool.metadata.category == ToolCategory.FILE
        assert "file" in file_tool.metadata.tags

    def test_get_schema(self, file_tool):
        """测试工具模式"""
        schema = file_tool.get_schema()
        assert schema.name == "file_tool"
        assert "operation" in schema.parameters["required"]
        assert "path" in schema.parameters["required"]

    @pytest.mark.asyncio
    async def test_write_and_read_file(self, file_tool):
        """测试写入和读取文件"""
        content = "测试文件内容"

        # 写入文件
        write_result = await file_tool.execute({
            "operation": "write",
            "path": "test.txt",
            "content": content
        })

        assert write_result.success is True
        assert "文件写入成功" in write_result.result

        # 读取文件
        read_result = await file_tool.execute({
            "operation": "read",
            "path": "test.txt"
        })

        assert read_result.success is True
        assert read_result.result == content

    @pytest.mark.asyncio
    async def test_append_file(self, file_tool):
        """测试追加文件"""
        initial_content = "初始内容"
        append_content = "\n追加内容"

        # 写入初始内容
        await file_tool.execute({
            "operation": "write",
            "path": "append_test.txt",
            "content": initial_content
        })

        # 追加内容
        append_result = await file_tool.execute({
            "operation": "append",
            "path": "append_test.txt",
            "content": append_content
        })

        assert append_result.success is True

        # 验证完整内容
        read_result = await file_tool.execute({
            "operation": "read",
            "path": "append_test.txt"
        })

        assert read_result.result == initial_content + append_content

    @pytest.mark.asyncio
    async def test_delete_file(self, file_tool):
        """测试删除文件"""
        # 创建文件
        await file_tool.execute({
            "operation": "write",
            "path": "delete_test.txt",
            "content": "待删除内容"
        })

        # 删除文件
        delete_result = await file_tool.execute({
            "operation": "delete",
            "path": "delete_test.txt"
        })

        assert delete_result.success is True

        # 验证文件不存在
        info_result = await file_tool.execute({
            "operation": "info",
            "path": "delete_test.txt"
        })

        assert info_result.result["exists"] is False

    @pytest.mark.asyncio
    async def test_list_directory(self, file_tool):
        """测试列出目录"""
        # 创建几个文件
        for i in range(3):
            await file_tool.execute({
                "operation": "write",
                "path": f"file_{i}.txt",
                "content": f"内容 {i}"
            })

        # 列出目录
        list_result = await file_tool.execute({
            "operation": "list",
            "path": ""
        })

        assert list_result.success is True
        assert isinstance(list_result.result, list)
        assert len(list_result.result) >= 3

    @pytest.mark.asyncio
    async def test_file_info(self, file_tool):
        """测试获取文件信息"""
        content = "测试内容" * 100  # 较长的内容

        # 创建文件
        await file_tool.execute({
            "operation": "write",
            "path": "info_test.txt",
            "content": content
        })

        # 获取文件信息
        info_result = await file_tool.execute({
            "operation": "info",
            "path": "info_test.txt"
        })

        assert info_result.success is True
        info = info_result.result
        assert info["exists"] is True
        assert info["is_file"] is True
        assert info["is_directory"] is False
        assert info["size"] > 0

    @pytest.mark.asyncio
    async def test_unsafe_path(self, file_tool):
        """测试不安全路径"""
        unsafe_paths = [
            "../../../etc/passwd",
            "/etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "\\windows\\system32\\config\\sam"
        ]

        for path in unsafe_paths:
            result = await file_tool.execute({
                "operation": "read",
                "path": path
            })

            assert result.success is False
            assert "文件路径不安全" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, file_tool):
        """测试不支持的操作"""
        result = await file_tool.execute({
            "operation": "unsupported",
            "path": "test.txt"
        })

        assert result.success is False
        assert "不支持的操作" in result.error


class TestDateTimeTool:
    """日期时间工具测试"""

    @pytest.fixture
    def datetime_tool(self):
        """日期时间工具实例"""
        return DateTimeTool()

    def test_tool_metadata(self, datetime_tool):
        """测试工具元数据"""
        assert datetime_tool.metadata.name == "datetime_tool"
        assert datetime_tool.metadata.category == ToolCategory.TIME
        assert "time" in datetime_tool.metadata.tags

    def test_get_schema(self, datetime_tool):
        """测试工具模式"""
        schema = datetime_tool.get_schema()
        assert schema.name == "datetime_tool"
        assert "operation" in schema.parameters["required"]

    @pytest.mark.asyncio
    async def test_get_current_time(self, datetime_tool):
        """测试获取当前时间"""
        result = await datetime_tool.execute({
            "operation": "current"
        })

        assert result.success is True
        assert "timestamp" in result.result
        assert "formatted" in result.result
        assert "components" in result.result

        components = result.result["components"]
        assert "year" in components
        assert "month" in components
        assert "day" in components
        assert "hour" in components
        assert "minute" in components
        assert "second" in components

    @pytest.mark.asyncio
    async def test_format_datetime(self, datetime_tool):
        """测试格式化日期时间"""
        dt_str = "2023-12-25T15:30:00"
        format_str = "%Y年%m月%d日 %H:%M:%S"

        result = await datetime_tool.execute({
            "operation": "format",
            "datetime": dt_str,
            "format": format_str
        })

        assert result.success is True
        assert result.result == "2023年12月25日 15:30:00"

    @pytest.mark.asyncio
    async def test_parse_datetime(self, datetime_tool):
        """测试解析日期时间"""
        dt_str = "2023-12-25 15:30:00"
        format_str = "%Y-%m-%d %H:%M:%S"

        result = await datetime_tool.execute({
            "operation": "parse",
            "datetime": dt_str,
            "format": format_str
        })

        assert result.success is True
        assert "timestamp" in result.result
        assert "iso_format" in result.result
        assert "components" in result.result

        components = result.result["components"]
        assert components["year"] == 2023
        assert components["month"] == 12
        assert components["day"] == 25
        assert components["hour"] == 15
        assert components["minute"] == 30
        assert components["second"] == 0

    @pytest.mark.asyncio
    async def test_calculate_datetime_add(self, datetime_tool):
        """测试日期时间计算 - 加法"""
        base_dt = "2023-12-25T00:00:00"

        # 加天数
        result = await datetime_tool.execute({
            "operation": "calculate",
            "base_datetime": base_dt,
            "calc_operation": "add",
            "value": 7,
            "unit": "days"
        })

        assert result.success is True
        assert result.result["original"] == base_dt
        assert "2024-01-01" in result.result["result"]  # 7天后应该是下一年
        assert result.result["operation"] == "add"
        assert result.result["value"] == 7
        assert result.result["unit"] == "days"

    @pytest.mark.asyncio
    async def test_calculate_datetime_subtract(self, datetime_tool):
        """测试日期时间计算 - 减法"""
        base_dt = "2023-12-25T00:00:00"

        # 减小时
        result = await datetime_tool.execute({
            "operation": "calculate",
            "base_datetime": base_dt,
            "calc_operation": "subtract",
            "value": 2,
            "unit": "hours"
        })

        assert result.success is True
        assert result.result["original"] == base_dt
        assert "2023-12-24T22" in result.result["result"]  # 2小时前应该是前一天22点
        assert result.result["operation"] == "subtract"

    @pytest.mark.asyncio
    async def test_timezone_time(self, datetime_tool):
        """测试时区时间"""
        result = await datetime_tool.execute({
            "operation": "timezone",
            "timezone": "UTC"
        })

        assert result.success is True
        assert "timezone" in result.result
        assert "time" in result.result
        assert "formatted" in result.result


class TestWeatherTool:
    """天气工具测试"""

    @pytest.fixture
    def weather_tool(self):
        """天气工具实例"""
        return WeatherTool()

    def test_tool_metadata(self, weather_tool):
        """测试工具元数据"""
        assert weather_tool.metadata.name == "weather_tool"
        assert weather_tool.metadata.category == ToolCategory.WEATHER
        assert "weather" in weather_tool.metadata.tags

    def test_get_schema(self, weather_tool):
        """测试工具模式"""
        schema = weather_tool.get_schema()
        assert schema.name == "weather_tool"
        assert "location" in schema.parameters["required"]

    @pytest.mark.asyncio
    async def test_current_weather(self, weather_tool):
        """测试当前天气"""
        result = await weather_tool.execute({
            "location": "北京",
            "type": "current"
        })

        assert result.success is True
        assert "current" in result.result
        assert "location" in result.result
        assert result.result["location"] == "北京"

        current = result.result["current"]
        assert "temperature" in current
        assert "humidity" in current
        assert "description" in current

    @pytest.mark.asyncio
    async def test_weather_forecast(self, weather_tool):
        """测试天气预报"""
        result = await weather_tool.execute({
            "location": "上海",
            "type": "forecast",
            "days": 5
        })

        assert result.success is True
        assert "forecast" in result.result
        assert result.result["location"] == "上海"
        assert result.result["days"] == 5
        assert isinstance(result.result["forecast"], list)

        if result.result["forecast"]:
            first_day = result.result["forecast"][0]
            assert "date" in first_day
            assert "temperature" in first_day
            assert "description" in first_day

    @pytest.mark.asyncio
    async def test_historical_weather(self, weather_tool):
        """测试历史天气"""
        result = await weather_tool.execute({
            "location": "广州",
            "type": "history",
            "date": "2023-12-25"
        })

        assert result.success is True
        assert "historical" in result.result
        assert result.result["location"] == "广州"

        historical = result.result["historical"]
        assert "temperature" in historical
        assert "description" in historical

    @pytest.mark.asyncio
    async def test_empty_location(self, weather_tool):
        """测试空位置"""
        result = await weather_tool.execute({
            "location": "",
            "type": "current"
        })

        assert result.success is False
        assert "位置信息不能为空" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_query_type(self, weather_tool):
        """测试不支持的查询类型"""
        result = await weather_tool.execute({
            "location": "北京",
            "type": "unsupported"
        })

        assert result.success is False
        assert "不支持的查询类型" in result.error


class TestMemoryTool:
    """记忆工具测试"""

    @pytest.fixture
    def mock_memory_system(self):
        """模拟记忆系统"""
        memory = AsyncMock()
        memory.add_memory.return_value = "memory_123"
        memory.retrieve.return_value = [
            Mock(
                id="mem1",
                content="测试记忆内容",
                memory_type="episodic",
                importance=0.8,
                tags=["test"],
                timestamp=time.time()
            )
        ]
        memory.update_memory.return_value = True
        memory.delete_memory.return_value = True
        memory.clear_memory.return_value = 10
        memory.search.return_value = []
        memory.get_stats.return_value = {"total_memories": 100}
        return memory

    @pytest.fixture
    def memory_tool(self, mock_memory_system):
        """记忆工具实例"""
        return MemoryTool(mock_memory_system)

    def test_tool_metadata(self, memory_tool):
        """测试工具元数据"""
        assert memory_tool.metadata.name == "memory_tool"
        assert memory_tool.metadata.category == ToolCategory.MEMORY
        assert "memory" in memory_tool.metadata.tags

    def test_get_schema(self, memory_tool):
        """测试工具模式"""
        schema = memory_tool.get_schema()
        assert schema.name == "memory_tool"
        assert "operation" in schema.parameters["required"]

    @pytest.mark.asyncio
    async def test_add_memory(self, memory_tool, mock_memory_system):
        """测试添加记忆"""
        result = await memory_tool.execute({
            "operation": "add",
            "content": "新的记忆内容",
            "type": "episodic",
            "importance": 0.9,
            "tags": ["important", "test"]
        })

        assert result.success is True
        assert "memory_123" in result.result
        mock_memory_system.add_memory.assert_called_once_with(
            content="新的记忆内容",
            memory_type="episodic",
            importance=0.9,
            tags=["important", "test"]
        )

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, memory_tool, mock_memory_system):
        """测试检索记忆"""
        result = await memory_tool.execute({
            "operation": "retrieve",
            "query": "测试查询",
            "limit": 5
        })

        assert result.success is True
        assert isinstance(result.result, list)
        assert len(result.result) == 1

        first_memory = result.result[0]
        assert "id" in first_memory
        assert "content" in first_memory
        assert "type" in first_memory
        assert first_memory["content"] == "测试记忆内容"

    @pytest.mark.asyncio
    async def test_update_memory(self, memory_tool, mock_memory_system):
        """测试更新记忆"""
        result = await memory_tool.execute({
            "operation": "update",
            "id": "mem1",
            "updates": {"importance": 0.9}
        })

        assert result.success is True
        assert "成功" in result.result
        mock_memory_system.update_memory.assert_called_once_with(
            "mem1", {"importance": 0.9}
        )

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_tool, mock_memory_system):
        """测试删除记忆"""
        result = await memory_tool.execute({
            "operation": "delete",
            "id": "mem1"
        })

        assert result.success is True
        assert "成功" in result.result
        mock_memory_system.delete_memory.assert_called_once_with("mem1")

    @pytest.mark.asyncio
    async def test_clear_memory(self, memory_tool, mock_memory_system):
        """测试清空记忆"""
        result = await memory_tool.execute({
            "operation": "clear",
            "type": "episodic"
        })

        assert result.success is True
        assert "10 条记忆" in result.result
        mock_memory_system.clear_memory.assert_called_once_with("episodic")

    @pytest.mark.asyncio
    async def test_search_memory(self, memory_tool, mock_memory_system):
        """测试搜索记忆"""
        result = await memory_tool.execute({
            "operation": "search",
            "keyword": "测试",
            "tags": ["important"],
            "limit": 10
        })

        assert result.success is True
        assert isinstance(result.result, list)
        mock_memory_system.search.assert_called_once_with(
            keyword="测试",
            tags=["important"],
            limit=10
        )

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_tool, mock_memory_system):
        """测试获取记忆统计"""
        result = await memory_tool.execute({
            "operation": "stats"
        })

        assert result.success is True
        assert result.result["total_memories"] == 100
        mock_memory_system.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_memory_system(self):
        """测试没有记忆系统的情况"""
        tool = MemoryTool(None)

        result = await tool.execute({
            "operation": "add",
            "content": "测试内容"
        })

        assert result.success is False
        assert "记忆系统未初始化" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, memory_tool):
        """测试不支持的操作"""
        result = await memory_tool.execute({
            "operation": "unsupported"
        })

        assert result.success is False
        assert "不支持的操作" in result.error


class TestToolMetadata:
    """工具元数据测试"""

    def test_tool_metadata_creation(self):
        """测试工具元数据创建"""
        metadata = ToolMetadata(
            name="test_tool",
            description="测试工具",
            category=ToolCategory.CALCULATION,
            version="2.0.0",
            author="Test Author",
            tags=["test", "example"],
            dependencies=["numpy", "scipy"],
            timeout=60.0,
            async_capable=True
        )

        assert metadata.name == "test_tool"
        assert metadata.description == "测试工具"
        assert metadata.category == ToolCategory.CALCULATION
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert "test" in metadata.tags
        assert "example" in metadata.tags
        assert "numpy" in metadata.dependencies
        assert "scipy" in metadata.dependencies
        assert metadata.timeout == 60.0
        assert metadata.async_capable is True

    def test_tool_category_enum(self):
        """测试工具分类枚举"""
        assert ToolCategory.CALCULATION.value == "calculation"
        assert ToolCategory.SEARCH.value == "search"
        assert ToolCategory.FILE.value == "file"
        assert ToolCategory.SYSTEM.value == "system"
        assert ToolCategory.COMMUNICATION.value == "communication"
        assert ToolCategory.DATA_PROCESSING.value == "data_processing"
        assert ToolCategory.TIME.value == "time"
        assert ToolCategory.WEATHER.value == "weather"
        assert ToolCategory.MEMORY.value == "memory"
