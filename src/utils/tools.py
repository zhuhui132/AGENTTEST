"""
工具系统实现

提供各种实用工具的统一接口和实现。
"""

import asyncio
import datetime
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..core.interfaces import BaseTool
from ..core.types import ToolResult, ToolSchema
from ..core.exceptions import ToolExecutionError


class ToolCategory(Enum):
    """工具分类"""
    CALCULATION = "calculation"
    SEARCH = "search"
    FILE = "file"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    TIME = "time"
    WEATHER = "weather"
    MEMORY = "memory"


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "System"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    async_capable: bool = True


class CalculatorTool(BaseTool):
    """计算器工具"""

    def __init__(self):
        self.metadata = ToolMetadata(
            name="calculator",
            description="执行数学表达式计算",
            category=ToolCategory.CALCULATION,
            tags=["math", "calculation", "expression"]
        )

    async def execute(self, parameters: dict) -> ToolResult:
        """执行计算"""
        try:
            expression = parameters.get("expression", "")
            if not expression:
                return ToolResult(
                    success=False,
                    error="表达式不能为空",
                    tool_name=self.metadata.name
                )

            start_time = time.time()

            # 安全表达式解析
            allowed_names = {
                '__builtins__': {},
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
            }

            # 检查表达式安全性
            if not self._is_safe_expression(expression):
                return ToolResult(
                    success=False,
                    error="表达式包含不安全的内容",
                    tool_name=self.metadata.name
                )

            result = eval(expression, allowed_names, {})
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={
                    "expression": expression,
                    "result_type": type(result).__name__
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"计算错误: {str(e)}",
                tool_name=self.metadata.name
            )

    def _is_safe_expression(self, expression: str) -> bool:
        """检查表达式是否安全"""
        # 检查危险字符
        dangerous_patterns = [
            r'import\s+', r'__import__', r'eval\s*\(', r'exec\s*\(',
            r'open\s*\(', r'file\s*\(', r'input\s*\(', r'raw_input\s*\(',
            r'\.strip\s*\(', r'\.split\s*\(', r'\.replace\s*\(',
            r'while\s+', r'for\s+', r'def\s+', r'class\s+',
            r'lambda\s+', r'yield\s+', r'return\s+', r'break\s+', r'continue\s+'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False

        return True

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式",
                        "examples": [
                            "2 + 3 * 4",
                            "sin(0.5) * 2",
                            "sqrt(16) + log(10)",
                            "pow(2, 8)"
                        ]
                    }
                },
                "required": ["expression"]
            }
        )


class WebSearchTool(BaseTool):
    """网页搜索工具"""

    def __init__(self, api_key: Optional[str] = None):
        self.metadata = ToolMetadata(
            name="web_search",
            description="在网络上搜索信息",
            category=ToolCategory.SEARCH,
            tags=["search", "web", "internet"],
            dependencies=["aiohttp"]
        )
        self.api_key = api_key

    async def execute(self, parameters: dict) -> ToolResult:
        """执行网页搜索"""
        try:
            query = parameters.get("query", "")
            limit = parameters.get("limit", 10)
            source = parameters.get("source", "general")

            if not query:
                return ToolResult(
                    success=False,
                    error="搜索查询不能为空",
                    tool_name=self.metadata.name
                )

            start_time = time.time()

            # 模拟搜索结果（实际应用中调用真实搜索API）
            results = await self._perform_search(query, limit, source)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=results,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={
                    "query": query,
                    "limit": limit,
                    "source": source,
                    "result_count": len(results)
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"搜索失败: {str(e)}",
                tool_name=self.metadata.name
            )

    async def _perform_search(
        self,
        query: str,
        limit: int,
        source: str
    ) -> List[Dict[str, Any]]:
        """执行搜索（模拟实现）"""
        # 这里应该调用真实的搜索API
        # 目前返回模拟结果
        mock_results = [
            {
                "title": f"搜索结果 {i+1}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"关于'{query}'的相关信息 {i+1}",
                "relevance_score": 0.9 - (i * 0.1)
            }
            for i in range(min(limit, 5))
        ]

        return mock_results

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "source": {
                        "type": "string",
                        "description": "搜索来源",
                        "enum": ["general", "news", "academic", "images"],
                        "default": "general"
                    }
                },
                "required": ["query"]
            }
        )


class FileTool(BaseTool):
    """文件操作工具"""

    def __init__(self, base_path: str = "/tmp"):
        self.metadata = ToolMetadata(
            name="file_tool",
            description="文件读写和操作工具",
            category=ToolCategory.FILE,
            tags=["file", "io", "storage"]
        )
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    async def execute(self, parameters: dict) -> ToolResult:
        """执行文件操作"""
        try:
            operation = parameters.get("operation", "read")
            file_path = parameters.get("path", "")
            content = parameters.get("content", "")
            encoding = parameters.get("encoding", "utf-8")

            if not file_path:
                return ToolResult(
                    success=False,
                    error="文件路径不能为空",
                    tool_name=self.metadata.name
                )

            start_time = time.time()

            # 安全检查
            full_path = self._get_safe_path(file_path)
            if not full_path:
                return ToolResult(
                    success=False,
                    error="文件路径不安全",
                    tool_name=self.metadata.name
                )

            result = None
            if operation == "read":
                result = await self._read_file(full_path, encoding)
            elif operation == "write":
                result = await self._write_file(full_path, content, encoding)
            elif operation == "append":
                result = await self._append_file(full_path, content, encoding)
            elif operation == "delete":
                result = await self._delete_file(full_path)
            elif operation == "list":
                result = await self._list_directory(full_path)
            elif operation == "info":
                result = await self._get_file_info(full_path)
            else:
                return ToolResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    tool_name=self.metadata.name
                )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={
                    "operation": operation,
                    "path": file_path,
                    "full_path": full_path
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"文件操作失败: {str(e)}",
                tool_name=self.metadata.name
            )

    def _get_safe_path(self, file_path: str) -> Optional[str]:
        """获取安全的文件路径"""
        try:
            # 规范化路径
            normalized = os.path.normpath(file_path)

            # 检查是否试图访问父目录
            if '..' in normalized or normalized.startswith('/'):
                return None

            # 构建完整路径
            full_path = os.path.join(self.base_path, normalized)

            # 确保路径在允许的范围内
            if not full_path.startswith(self.base_path):
                return None

            return full_path

        except Exception:
            return None

    async def _read_file(self, path: str, encoding: str) -> str:
        """读取文件"""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()

    async def _write_file(self, path: str, content: str, encoding: str) -> str:
        """写入文件"""
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return f"文件写入成功: {path}"

    async def _append_file(self, path: str, content: str, encoding: str) -> str:
        """追加文件"""
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)
        return f"内容追加成功: {path}"

    async def _delete_file(self, path: str) -> str:
        """删除文件"""
        if os.path.exists(path):
            os.remove(path)
            return f"文件删除成功: {path}"
        else:
            return f"文件不存在: {path}"

    async def _list_directory(self, path: str) -> List[str]:
        """列出目录"""
        if os.path.isdir(path):
            return os.listdir(path)
        else:
            return []

    async def _get_file_info(self, path: str) -> Dict[str, Any]:
        """获取文件信息"""
        if not os.path.exists(path):
            return {"exists": False}

        stat = os.stat(path)
        return {
            "exists": True,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "is_directory": os.path.isdir(path),
            "is_file": os.path.isfile(path)
        }

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "文件操作类型",
                        "enum": ["read", "write", "append", "delete", "list", "info"],
                        "default": "read"
                    },
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容（用于write和append操作）"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "文件编码",
                        "default": "utf-8"
                    }
                },
                "required": ["operation", "path"]
            }
        )


class DateTimeTool(BaseTool):
    """日期时间工具"""

    def __init__(self):
        self.metadata = ToolMetadata(
            name="datetime_tool",
            description="日期时间处理工具",
            category=ToolCategory.TIME,
            tags=["time", "date", "datetime"]
        )

    async def execute(self, parameters: dict) -> ToolResult:
        """执行日期时间操作"""
        try:
            operation = parameters.get("operation", "current")

            start_time = time.time()
            result = None

            if operation == "current":
                result = await self._get_current_time(parameters)
            elif operation == "format":
                result = await self._format_datetime(parameters)
            elif operation == "parse":
                result = await self._parse_datetime(parameters)
            elif operation == "calculate":
                result = await self._calculate_datetime(parameters)
            elif operation == "timezone":
                result = await self._get_timezone_time(parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    tool_name=self.metadata.name
                )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={"operation": operation}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"日期时间操作失败: {str(e)}",
                tool_name=self.metadata.name
            )

    async def _get_current_time(self, params: dict) -> Dict[str, Any]:
        """获取当前时间"""
        timezone = params.get("timezone", "UTC")
        format_str = params.get("format", "%Y-%m-%d %H:%M:%S")

        now = datetime.datetime.now()
        if timezone != "UTC":
            # 简化的时区处理
            pass

        return {
            "timestamp": now.timestamp(),
            "formatted": now.strftime(format_str),
            "timezone": timezone,
            "components": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "weekday": now.weekday()
            }
        }

    async def _format_datetime(self, params: dict) -> str:
        """格式化日期时间"""
        dt_str = params.get("datetime", "")
        format_str = params.get("format", "%Y-%m-%d %H:%M:%S")

        dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime(format_str)

    async def _parse_datetime(self, params: dict) -> Dict[str, Any]:
        """解析日期时间"""
        dt_str = params.get("datetime", "")
        format_str = params.get("format", "%Y-%m-%d %H:%M:%S")

        dt = datetime.datetime.strptime(dt_str, format_str)
        return {
            "timestamp": dt.timestamp(),
            "iso_format": dt.isoformat(),
            "components": {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "minute": dt.minute,
                "second": dt.second
            }
        }

    async def _calculate_datetime(self, params: dict) -> Dict[str, Any]:
        """日期时间计算"""
        base_dt = params.get("base_datetime", "")
        operation = params.get("calc_operation", "add")
        value = params.get("value", 0)
        unit = params.get("unit", "days")

        dt = datetime.datetime.fromisoformat(base_dt.replace('Z', '+00:00'))

        if operation == "add":
            if unit == "days":
                result_dt = dt + datetime.timedelta(days=value)
            elif unit == "hours":
                result_dt = dt + datetime.timedelta(hours=value)
            elif unit == "minutes":
                result_dt = dt + datetime.timedelta(minutes=value)
            elif unit == "seconds":
                result_dt = dt + datetime.timedelta(seconds=value)
            else:
                raise ValueError(f"不支持的时间单位: {unit}")
        elif operation == "subtract":
            if unit == "days":
                result_dt = dt - datetime.timedelta(days=value)
            elif unit == "hours":
                result_dt = dt - datetime.timedelta(hours=value)
            elif unit == "minutes":
                result_dt = dt - datetime.timedelta(minutes=value)
            elif unit == "seconds":
                result_dt = dt - datetime.timedelta(seconds=value)
            else:
                raise ValueError(f"不支持的时间单位: {unit}")
        else:
            raise ValueError(f"不支持的计算操作: {operation}")

        return {
            "original": dt.isoformat(),
            "result": result_dt.isoformat(),
            "operation": operation,
            "value": value,
            "unit": unit,
            "difference_seconds": (result_dt - dt).total_seconds()
        }

    async def _get_timezone_time(self, params: dict) -> Dict[str, Any]:
        """获取时区时间（简化实现）"""
        timezone = params.get("timezone", "UTC")

        # 这里应该使用真实的时区库
        current_time = datetime.datetime.now()

        return {
            "timezone": timezone,
            "time": current_time.isoformat(),
            "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "日期时间操作类型",
                        "enum": ["current", "format", "parse", "calculate", "timezone"],
                        "default": "current"
                    },
                    "datetime": {
                        "type": "string",
                        "description": "日期时间字符串"
                    },
                    "format": {
                        "type": "string",
                        "description": "日期时间格式",
                        "default": "%Y-%m-%d %H:%M:%S"
                    },
                    "timezone": {
                        "type": "string",
                        "description": "时区",
                        "default": "UTC"
                    },
                    "calc_operation": {
                        "type": "string",
                        "description": "计算操作类型",
                        "enum": ["add", "subtract"]
                    },
                    "value": {
                        "type": "number",
                        "description": "计算数值"
                    },
                    "unit": {
                        "type": "string",
                        "description": "时间单位",
                        "enum": ["days", "hours", "minutes", "seconds"],
                        "default": "days"
                    }
                },
                "required": ["operation"]
            }
        )


class WeatherTool(BaseTool):
    """天气工具"""

    def __init__(self, api_key: Optional[str] = None):
        self.metadata = ToolMetadata(
            name="weather_tool",
            description="天气信息查询工具",
            category=ToolCategory.WEATHER,
            tags=["weather", "climate", "forecast"],
            dependencies=["aiohttp"]
        )
        self.api_key = api_key

    async def execute(self, parameters: dict) -> ToolResult:
        """执行天气查询"""
        try:
            location = parameters.get("location", "")
            query_type = parameters.get("type", "current")
            days = parameters.get("days", 7)

            if not location:
                return ToolResult(
                    success=False,
                    error="位置信息不能为空",
                    tool_name=self.metadata.name
                )

            start_time = time.time()

            if query_type == "current":
                result = await self._get_current_weather(location)
            elif query_type == "forecast":
                result = await self._get_weather_forecast(location, days)
            elif query_type == "history":
                result = await self._get_historical_weather(location, parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"不支持的查询类型: {query_type}",
                    tool_name=self.metadata.name
                )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={
                    "location": location,
                    "type": query_type
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"天气查询失败: {str(e)}",
                tool_name=self.metadata.name
            )

    async def _get_current_weather(self, location: str) -> Dict[str, Any]:
        """获取当前天气（模拟实现）"""
        return {
            "location": location,
            "current": {
                "temperature": 22.5,
                "humidity": 65,
                "pressure": 1013.25,
                "wind_speed": 10.5,
                "wind_direction": "NE",
                "description": "多云",
                "visibility": 10.0,
                "uv_index": 3,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }

    async def _get_weather_forecast(self, location: str, days: int) -> Dict[str, Any]:
        """获取天气预报（模拟实现）"""
        forecast = []
        for i in range(min(days, 7)):
            date = datetime.datetime.now() + datetime.timedelta(days=i)
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "temperature": {
                    "min": 18 + i,
                    "max": 25 + i
                },
                "humidity": 60 + i * 2,
                "description": "晴转多云" if i % 2 == 0 else "小雨",
                "precipitation": 0.0 if i % 2 == 0 else 5.2
            })

        return {
            "location": location,
            "forecast": forecast,
            "days": len(forecast)
        }

    async def _get_historical_weather(self, location: str, params: dict) -> Dict[str, Any]:
        """获取历史天气（模拟实现）"""
        date_str = params.get("date", "")
        return {
            "location": location,
            "date": date_str,
            "historical": {
                "temperature": {
                    "min": 15.2,
                    "max": 28.7,
                    "avg": 21.9
                },
                "humidity": 68.5,
                "precipitation": 2.1,
                "description": "晴"
            }
        }

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "查询位置（城市名、坐标等）"
                    },
                    "type": {
                        "type": "string",
                        "description": "查询类型",
                        "enum": ["current", "forecast", "history"],
                        "default": "current"
                    },
                    "days": {
                        "type": "integer",
                        "description": "预报天数",
                        "default": 7,
                        "minimum": 1,
                        "maximum": 14
                    },
                    "date": {
                        "type": "string",
                        "description": "历史天气查询日期（YYYY-MM-DD）"
                    }
                },
                "required": ["location"]
            }
        )


class MemoryTool(BaseTool):
    """记忆工具"""

    def __init__(self, memory_system=None):
        self.metadata = ToolMetadata(
            name="memory_tool",
            description="记忆系统操作工具",
            category=ToolCategory.MEMORY,
            tags=["memory", "storage", "retrieval"]
        )
        self.memory_system = memory_system

    async def execute(self, parameters: dict) -> ToolResult:
        """执行记忆操作"""
        try:
            operation = parameters.get("operation", "retrieve")

            if not self.memory_system:
                return ToolResult(
                    success=False,
                    error="记忆系统未初始化",
                    tool_name=self.metadata.name
                )

            start_time = time.time()
            result = None

            if operation == "add":
                result = await self._add_memory(parameters)
            elif operation == "retrieve":
                result = await self._retrieve_memory(parameters)
            elif operation == "update":
                result = await self._update_memory(parameters)
            elif operation == "delete":
                result = await self._delete_memory(parameters)
            elif operation == "clear":
                result = await self._clear_memory(parameters)
            elif operation == "search":
                result = await self._search_memory(parameters)
            elif operation == "stats":
                result = await self._get_memory_stats()
            else:
                return ToolResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    tool_name=self.metadata.name
                )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={"operation": operation}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"记忆操作失败: {str(e)}",
                tool_name=self.metadata.name
            )

    async def _add_memory(self, params: dict) -> str:
        """添加记忆"""
        content = params.get("content", "")
        memory_type = params.get("type", "episodic")
        importance = params.get("importance", 0.5)
        tags = params.get("tags", [])

        memory_id = await self.memory_system.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags
        )
        return f"记忆添加成功，ID: {memory_id}"

    async def _retrieve_memory(self, params: dict) -> List[Dict[str, Any]]:
        """检索记忆"""
        query = params.get("query", "")
        limit = params.get("limit", 10)
        memory_type = params.get("type", None)

        memories = await self.memory_system.retrieve(
            query=query,
            limit=limit,
            memory_type=memory_type
        )

        return [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "tags": memory.tags,
                "timestamp": memory.timestamp.isoformat()
            }
            for memory in memories
        ]

    async def _update_memory(self, params: dict) -> str:
        """更新记忆"""
        memory_id = params.get("id", "")
        updates = params.get("updates", {})

        success = await self.memory_system.update_memory(memory_id, updates)
        return f"记忆更新{'成功' if success else '失败'}"

    async def _delete_memory(self, params: dict) -> str:
        """删除记忆"""
        memory_id = params.get("id", "")

        success = await self.memory_system.delete_memory(memory_id)
        return f"记忆删除{'成功' if success else '失败'}"

    async def _clear_memory(self, params: dict) -> str:
        """清空记忆"""
        memory_type = params.get("type", None)

            count = await self.memory_system.clear_memory(memory_type)
        return f"已清空 {count} 条记忆"

    async def _search_memory(self, params: dict) -> List[Dict[str, Any]]:
        """搜索记忆"""
        keyword = params.get("keyword", "")
        tags = params.get("tags", [])
        limit = params.get("limit", 20)

        memories = await self.memory_system.search(
            keyword=keyword,
            tags=tags,
            limit=limit
        )

        return [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "tags": memory.tags,
                "timestamp": memory.timestamp.isoformat()
            }
            for memory in memories
        ]

    async def _get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        stats = await self.memory_system.get_stats()
        return stats

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "记忆操作类型",
                        "enum": ["add", "retrieve", "update", "delete", "clear", "search", "stats"],
                        "default": "retrieve"
                    },
                    "id": {
                        "type": "string",
                        "description": "记忆ID"
                    },
                    "content": {
                        "type": "string",
                        "description": "记忆内容"
                    },
                    "type": {
                        "type": "string",
                        "description": "记忆类型",
                        "enum": ["episodic", "semantic", "working", "procedural"]
                    },
                    "query": {
                        "type": "string",
                        "description": "检索查询"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "记忆标签"
                    },
                    "importance": {
                        "type": "number",
                        "description": "记忆重要性（0-1）",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "结果数量限制",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "updates": {
                        "type": "object",
                        "description": "更新内容"
                    }
                },
                "required": ["operation"]
            }
        )
