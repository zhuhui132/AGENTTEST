# 工具系统API文档

## 概览

工具系统提供了丰富的预构建工具和自定义工具开发接口，支持计算、搜索、文件操作、日期时间处理、天气查询、记忆管理等多种功能。

## 核心接口

### BaseTool

所有工具的基类，定义了工具的标准接口。

```python
from abc import ABC, abstractmethod
from src.core.types import ToolResult, ToolSchema

class BaseTool(ABC):
    """工具基类"""

    @abstractmethod
    async def execute(self, parameters: dict) -> ToolResult:
        """执行工具"""
        pass

    def get_schema(self) -> ToolSchema:
        """获取工具模式描述"""
        pass

    @property
    def name(self) -> str:
        """工具名称"""
        return self.get_schema().name
```

## 预构建工具

### 1. CalculatorTool - 计算器工具

数学表达式计算工具，支持基础运算和高级数学函数。

#### 构造函数

```python
CalculatorTool() -> CalculatorTool
```

#### 使用示例

```python
from src.utils.tools import CalculatorTool

# 创建计算器实例
calculator = CalculatorTool()

# 基础算术运算
result = await calculator.execute({
    "expression": "2 + 3 * 4"
})
print(f"结果: {result.result}")  # 输出: 14

# 数学函数
result = await calculator.execute({
    "expression": "sqrt(16) + sin(pi/2)"
})
print(f"结果: {result.result}")  # 输出: 5.0

# 复杂表达式
result = await calculator.execute({
    "expression": "pow(2, 3) * log(10) - abs(-5)"
})
print(f"结果: {result.result}")
```

#### 支持的运算符和函数

**算术运算符：**
- `+` 加法
- `-` 减法
- `*` 乘法
- `/` 除法
- `**` 幂运算
- `%` 取模

**数学函数：**
- `abs(x)` 绝对值
- `round(x, n)` 四舍五入
- `min(...)` 最小值
- `max(...)` 最大值
- `sum(...)` 求和
- `len(x)` 长度
- `pow(x, y)` 幂运算
- `sqrt(x)` 平方根
- `sin(x)` 正弦
- `cos(x)` 余弦
- `tan(x)` 正切
- `log(x)` 自然对数
- `exp(x)` 指数

**常数：**
- `pi` 圆周率
- `e` 自然常数

#### 安全特性

计算器工具具有严格的安全检查：

```python
# 不安全表达式会被拒绝
unsafe_expressions = [
    "import os",
    "__import__('os')",
    "eval('1+1')",
    "open('file.txt')",
    "while True: pass"
]

for expr in unsafe_expressions:
    result = await calculator.execute({"expression": expr})
    assert result.success is False
    assert "不安全的内容" in result.error
```

### 2. WebSearchTool - 网页搜索工具

网络信息搜索工具，支持多种搜索源和结果过滤。

#### 构造函数

```python
WebSearchTool(api_key: Optional[str] = None) -> WebSearchTool
```

**参数：**
- `api_key` (str, 可选): 搜索API密钥

#### 使用示例

```python
from src.utils.tools import WebSearchTool

# 创建搜索工具
search_tool = WebSearchTool()

# 基础搜索
result = await search_tool.execute({
    "query": "Python编程最佳实践",
    "limit": 10
})

if result.success:
    for item in result.result:
        print(f"标题: {item['title']}")
        print(f"链接: {item['url']}")
        print(f"摘要: {item['snippet']}")
        print(f"相关性: {item['relevance_score']}")
        print("-" * 50)

# 指定搜索源
result = await search_tool.execute({
    "query": "机器学习最新研究",
    "source": "academic",  # 学术搜索
    "limit": 5
})

# 图像搜索
result = await search_tool.execute({
    "query": "深度学习架构图",
    "source": "images",
    "limit": 20
})
```

#### 搜索源类型

- `general`: 通用网页搜索
- `news`: 新闻搜索
- `academic`: 学术搜索
- `images`: 图像搜索

#### 结果格式

```python
{
    "title": "搜索结果标题",
    "url": "https://example.com/page",
    "snippet": "搜索结果摘要...",
    "relevance_score": 0.85,
    "publish_date": "2023-12-25",
    "author": "作者名",
    "additional_info": {
        "word_count": 1500,
        "language": "zh-CN"
    }
}
```

### 3. FileTool - 文件操作工具

安全的文件读写和管理工具，支持多种文件操作。

#### 构造函数

```python
FileTool(base_path: str = "/tmp") -> FileTool
```

**参数：**
- `base_path` (str): 基础路径，限制文件访问范围

#### 使用示例

```python
import tempfile
import os

# 创建临时目录作为基础路径
with tempfile.TemporaryDirectory() as temp_dir:
    file_tool = FileTool(base_path=temp_dir)

    # 写入文件
    result = await file_tool.execute({
        "operation": "write",
        "path": "test.txt",
        "content": "Hello, World!\n这是测试内容。"
    })
    print(result.result)  # 文件写入成功

    # 读取文件
    result = await file_tool.execute({
        "operation": "read",
        "path": "test.txt"
    })
    print(result.result)  # 文件内容

    # 追加内容
    result = await file_tool.execute({
        "operation": "append",
        "path": "test.txt",
        "content": "\n追加的新行"
    })

    # 获取文件信息
    result = await file_tool.execute({
        "operation": "info",
        "path": "test.txt"
    })
    print(f"文件大小: {result.result['size']} 字节")
    print(f"最后修改: {result.result['modified']}")

    # 列出目录
    result = await file_tool.execute({
        "operation": "list",
        "path": ""
    })
    print("目录内容:", result.result)

    # 删除文件
    result = await file_tool.execute({
        "operation": "delete",
        "path": "test.txt"
    })
```

#### 支持的操作

| 操作 | 描述 | 参数 |
|------|------|------|
| `read` | 读取文件 | `path`, `encoding` |
| `write` | 写入文件 | `path`, `content`, `encoding` |
| `append` | 追加内容 | `path`, `content`, `encoding` |
| `delete` | 删除文件 | `path` |
| `list` | 列出目录 | `path` |
| `info` | 获取文件信息 | `path` |

#### 安全特性

文件工具具有严格的安全限制：

```python
# 这些路径会被拒绝
unsafe_paths = [
    "../../../etc/passwd",
    "/etc/passwd",
    "..\\..\\windows\\system32\\config\\sam"
]

for path in unsafe_paths:
    result = await file_tool.execute({
        "operation": "read",
        "path": path
    })
    assert result.success is False
    assert "文件路径不安全" in result.error
```

### 4. DateTimeTool - 日期时间工具

日期时间处理和计算工具，支持格式化、解析、计算等操作。

#### 构造函数

```python
DateTimeTool() -> DateTimeTool
```

#### 使用示例

```python
from src.utils.tools import DateTimeTool

datetime_tool = DateTimeTool()

# 获取当前时间
result = await datetime_tool.execute({
    "operation": "current",
    "timezone": "Asia/Shanghai",
    "format": "%Y-%m-%d %H:%M:%S"
})

current = result.result
print(f"当前时间: {current['formatted']}")
print(f"时间戳: {current['timestamp']}")
print(f"组件: {current['components']}")

# 格式化日期时间
result = await datetime_tool.execute({
    "operation": "format",
    "datetime": "2023-12-25T15:30:00",
    "format": "%Y年%m月%d日 %H时%M分"
})
print(f"格式化结果: {result.result}")

# 解析日期时间
result = await datetime_tool.execute({
    "operation": "parse",
    "datetime": "2023-12-25 15:30:00",
    "format": "%Y-%m-%d %H:%M:%S"
})

parsed = result.result
print(f"时间戳: {parsed['timestamp']}")
print(f"ISO格式: {parsed['iso_format']}")
print(f"组件: {parsed['components']}")

# 日期时间计算
# 加法运算
result = await datetime_tool.execute({
    "operation": "calculate",
    "base_datetime": "2023-12-25T00:00:00",
    "calc_operation": "add",
    "value": 7,
    "unit": "days"
})
print(f"7天后: {result.result['result']}")

# 减法运算
result = await datetime_tool.execute({
    "operation": "calculate",
    "base_datetime": "2023-12-25T12:00:00",
    "calc_operation": "subtract",
    "value": 2,
    "unit": "hours"
})
print(f"2小时前: {result.result['result']}")

# 获取时区时间
result = await datetime_tool.execute({
    "operation": "timezone",
    "timezone": "UTC"
})
print(f"UTC时间: {result.result['time']}")
```

#### 支持的时间单位

- `days`: 天
- `hours`: 小时
- `minutes`: 分钟
- `seconds`: 秒

#### 时区支持

工具支持常见时区：
- `UTC`: 协调世界时
- `Asia/Shanghai`: 上海时间
- `America/New_York`: 纽约时间
- `Europe/London`: 伦敦时间

### 5. WeatherTool - 天气工具

天气信息查询工具，支持当前天气、天气预报、历史天气查询。

#### 构造函数

```python
WeatherTool(api_key: Optional[str] = None) -> WeatherTool
```

**参数：**
- `api_key` (str, 可选): 天气API密钥

#### 使用示例

```python
from src.utils.tools import WeatherTool

weather_tool = WeatherTool()

# 查询当前天气
result = await weather_tool.execute({
    "location": "北京",
    "type": "current"
})

if result.success:
    current = result.result["current"]
    print(f"温度: {current['temperature']}°C")
    print(f"湿度: {current['humidity']}%")
    print(f"气压: {current['pressure']} hPa")
    print(f"风速: {current['wind_speed']} m/s")
    print(f"天气描述: {current['description']}")

# 查询天气预报
result = await weather_tool.execute({
    "location": "上海",
    "type": "forecast",
    "days": 5
})

if result.success:
    forecast = result.result["forecast"]
    for day in forecast:
        print(f"日期: {day['date']}")
        print(f"温度: {day['temperature']['min']}°C - {day['temperature']['max']}°C")
        print(f"天气: {day['description']}")
        print(f"湿度: {day['humidity']}%")
        print("-" * 30)

# 查询历史天气
result = await weather_tool.execute({
    "location": "广州",
    "type": "history",
    "date": "2023-12-25"
})

if result.success:
    historical = result.result["historical"]
    print(f"历史温度: {historical['temperature']['avg']}°C")
    print(f"湿度: {historical['humidity']}%")
    print(f"天气: {historical['description']}")
```

#### 天气数据格式

**当前天气：**
```python
{
    "temperature": 22.5,        # 温度 (°C)
    "humidity": 65,            # 湿度 (%)
    "pressure": 1013.25,       # 气压 (hPa)
    "wind_speed": 10.5,        # 风速 (m/s)
    "wind_direction": "NE",    # 风向
    "description": "多云",      # 天气描述
    "visibility": 10.0,        # 能见度 (km)
    "uv_index": 3,            # 紫外线指数
    "timestamp": "2023-12-25T15:30:00Z"
}
```

**天气预报：**
```python
{
    "date": "2023-12-26",
    "temperature": {
        "min": 18,
        "max": 25
    },
    "humidity": 60,
    "description": "晴转多云",
    "precipitation": 0.0,      # 降水量 (mm)
    "wind_speed": 8.0,
    "uv_index": 4
}
```

### 6. MemoryTool - 记忆工具

记忆系统操作工具，用于管理Agent的记忆存储和检索。

#### 构造函数

```python
MemoryTool(memory_system: Optional[BaseMemory] = None) -> MemoryTool
```

**参数：**
- `memory_system` (BaseMemory, 可选): 记忆系统实例

#### 使用示例

```python
from src.utils.tools import MemoryTool
from src.memory.memory import MemorySystem

# 创建记忆系统
memory_system = MemorySystem()
await memory_system.initialize()

# 创建记忆工具
memory_tool = MemoryTool(memory_system)

# 添加记忆
result = await memory_tool.execute({
    "operation": "add",
    "content": "用户喜欢Python编程，有3年开发经验",
    "type": "semantic",
    "importance": 0.8,
    "tags": ["用户信息", "技能", "Python"]
})
print(f"记忆添加: {result.result}")

# 检索记忆
result = await memory_tool.execute({
    "operation": "retrieve",
    "query": "用户的编程技能",
    "limit": 5
})

for memory in result.result:
    print(f"ID: {memory['id']}")
    print(f"内容: {memory['content']}")
    print(f"类型: {memory['type']}")
    print(f"重要性: {memory['importance']}")
    print(f"标签: {memory['tags']}")
    print("-" * 30)

# 更新记忆
result = await memory_tool.execute({
    "operation": "update",
    "id": "memory_123",
    "updates": {
        "importance": 0.9,
        "tags": ["用户信息", "技能", "Python", "高级"]
    }
})

# 搜索记忆
result = await memory_tool.execute({
    "operation": "search",
    "keyword": "编程",
    "tags": ["技能"],
    "limit": 10
})

# 获取统计信息
result = await memory_tool.execute({
    "operation": "stats"
})

stats = result.result
print(f"总记忆数: {stats['total_memories']}")
print(f"按类型分布: {stats['type_distribution']}")

# 清空记忆
result = await memory_tool.execute({
    "operation": "clear",
    "type": "episodic"
})
print(f"清空结果: {result.result}")
```

#### 记忆类型

- `episodic`: 情景记忆（事件、对话）
- `semantic`: 语义记忆（知识、概念）
- `working`: 工作记忆（临时信息）
- `procedural`: 程序性记忆（技能、流程）

## 自定义工具开发

### 创建自定义工具

```python
from src.utils.tools import BaseTool, ToolResult, ToolSchema, ToolCategory
from src.core.types import ToolMetadata

class CustomTool(BaseTool):
    """自定义工具示例"""

    def __init__(self):
        self.metadata = ToolMetadata(
            name="custom_tool",
            description="自定义工具描述",
            category=ToolCategory.DATA_PROCESSING,
            tags=["custom", "example"],
            version="1.0.0"
        )

    async def execute(self, parameters: dict) -> ToolResult:
        """执行工具逻辑"""
        try:
            # 获取参数
            input_data = parameters.get("input", "")
            options = parameters.get("options", {})

            start_time = time.time()

            # 执行自定义逻辑
            result = await self.process_data(input_data, options)

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                execution_time=execution_time,
                metadata={
                    "input_length": len(input_data),
                    "options": options
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"工具执行失败: {str(e)}",
                tool_name=self.metadata.name
            )

    async def process_data(self, data: str, options: dict) -> str:
        """处理数据的核心逻辑"""
        # 实现具体的处理逻辑
        processed = data.upper() if options.get("uppercase", False) else data.lower()
        return f"处理结果: {processed}"

    def get_schema(self) -> ToolSchema:
        """返回工具模式"""
        return ToolSchema(
            name=self.metadata.name,
            description=self.metadata.description,
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "输入数据"
                    },
                    "options": {
                        "type": "object",
                        "description": "处理选项",
                        "properties": {
                            "uppercase": {
                                "type": "boolean",
                                "description": "是否转换为大写",
                                "default": False
                            }
                        }
                    }
                },
                "required": ["input"]
            }
        )
```

### 注册自定义工具

```python
# 在Agent中注册自定义工具
from src.agents.intelligent_agent import IntelligentAgent

agent = IntelligentAgent(config)
await agent.initialize()

# 注册工具
custom_tool = CustomTool()
await agent.register_tool("custom", custom_tool.execute)

# 使用工具
response = await agent.process_message(
    "请使用自定义工具处理这段文字"
)
```

### 工具开发最佳实践

#### 1. 错误处理

```python
async def execute(self, parameters: dict) -> ToolResult:
    try:
        # 参数验证
        self._validate_parameters(parameters)

        # 执行逻辑
        result = await self._perform_operation(parameters)

        return ToolResult(
            success=True,
            result=result,
            tool_name=self.name
        )

    except ValueError as e:
        return ToolResult(
            success=False,
            error=f"参数错误: {str(e)}",
            tool_name=self.name
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=f"执行错误: {str(e)}",
            tool_name=self.name
        )

def _validate_parameters(self, parameters: dict):
    """验证参数"""
    required_params = self.get_schema().parameters.get("required", [])
    for param in required_params:
        if param not in parameters:
            raise ValueError(f"缺少必需参数: {param}")
```

#### 2. 性能监控

```python
import time
from typing import Dict, Any

class MonitoredTool(BaseTool):
    """带性能监控的工具基类"""

    async def execute(self, parameters: dict) -> ToolResult:
        start_time = time.time()

        try:
            result = await self._execute_core(parameters)

            execution_time = time.time() - start_time
            self._log_metrics(execution_time, True, parameters)

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._log_metrics(execution_time, False, parameters)

            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time
            )

    async def _execute_core(self, parameters: dict) -> Any:
        """核心执行逻辑，子类实现"""
        raise NotImplementedError

    def _log_metrics(self, execution_time: float, success: bool, parameters: dict):
        """记录性能指标"""
        metrics = {
            "tool": self.name,
            "execution_time": execution_time,
            "success": success,
            "parameter_count": len(parameters),
            "timestamp": time.time()
        }

        # 发送到监控系统
        self._send_metrics(metrics)
```

#### 3. 缓存支持

```python
from functools import lru_cache
import hashlib

class CachedTool(BaseTool):
    """带缓存功能的工具基类"""

    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self._cache = {}

    async def execute(self, parameters: dict) -> ToolResult:
        # 生成缓存键
        cache_key = self._generate_cache_key(parameters)

        # 检查缓存
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            return cached_result

        # 执行并缓存结果
        result = await self._execute_without_cache(parameters)

        # 更新缓存
        if len(self._cache) >= self.cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result
        return result

    def _generate_cache_key(self, parameters: dict) -> str:
        """生成缓存键"""
        param_str = str(sorted(parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
```

## 工具管理系统

### 工具注册表

```python
from typing import Dict, Type, List
from src.utils.tools import BaseTool, ToolCategory

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {}

    def register_tool(self, name: str, tool: BaseTool):
        """注册工具"""
        self._tools[name] = tool

        # 按分类索引
        category = tool.metadata.category
        if category not in self._categories:
            self._categories[category] = []

        if name not in self._categories[category]:
            self._categories[category].append(name)

    def get_tool(self, name: str) -> BaseTool:
        """获取工具"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册")
        return self._tools[name]

    def list_tools(self, category: ToolCategory = None) -> List[str]:
        """列出工具"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def get_tools_by_category(self) -> Dict[ToolCategory, List[str]]:
        """按分类获取工具"""
        return self._categories.copy()

# 全局工具注册表
tool_registry = ToolRegistry()

# 注册默认工具
from src.utils.tools import (
    CalculatorTool, WebSearchTool, FileTool,
    DateTimeTool, WeatherTool, MemoryTool
)

default_tools = {
    "calculator": CalculatorTool(),
    "web_search": WebSearchTool(),
    "file_tool": FileTool(),
    "datetime_tool": DateTimeTool(),
    "weather_tool": WeatherTool(),
    "memory_tool": MemoryTool()
}

for name, tool in default_tools.items():
    tool_registry.register_tool(name, tool)
```

### 工具调用器

```python
class ToolInvoker:
    """工具调用器"""

    def __init__(self, registry: ToolRegistry = tool_registry):
        self.registry = registry
        self.call_history: List[Dict[str, Any]] = []

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: dict,
        context: dict = None
    ) -> ToolResult:
        """调用工具"""
        tool = self.registry.get_tool(tool_name)

        call_info = {
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": time.time(),
            "context": context or {}
        }

        try:
            result = await tool.execute(parameters)

            call_info.update({
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error
            })

            self.call_history.append(call_info)
            return result

        except Exception as e:
            call_info.update({
                "success": False,
                "error": str(e),
                "execution_time": 0
            })

            self.call_history.append(call_info)

            return ToolResult(
                success=False,
                error=f"工具调用失败: {str(e)}",
                tool_name=tool_name
            )

    def get_call_history(self, tool_name: str = None) -> List[Dict[str, Any]]:
        """获取调用历史"""
        if tool_name:
            return [call for call in self.call_history if call["tool"] == tool_name]
        return self.call_history.copy()

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for call in self.call_history if call["success"])

        tool_counts = {}
        for call in self.call_history:
            tool = call["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        avg_execution_time = 0
        if total_calls > 0:
            total_time = sum(call.get("execution_time", 0) for call in self.call_history)
            avg_execution_time = total_time / total_calls

        return {
            "total_calls": total_calls,
            "success_rate": successful_calls / max(1, total_calls),
            "tool_usage": tool_counts,
            "average_execution_time": avg_execution_time
        }

# 创建全局工具调用器
tool_invoker = ToolInvoker()
```

## 配置和扩展

### 工具配置

```python
# config/tools_config.yaml
tools:
  calculator:
    enabled: true
    timeout: 30.0

  web_search:
    enabled: true
    api_key: "${WEB_SEARCH_API_KEY}"
    default_limit: 10
    timeout: 60.0

  file_tool:
    enabled: true
    base_path: "/var/agent/files"
    max_file_size: 10485760  # 10MB

  weather_tool:
    enabled: true
    api_key: "${WEATHER_API_KEY}"
    default_location: "北京"

  datetime_tool:
    enabled: true
    default_timezone: "Asia/Shanghai"

  memory_tool:
    enabled: true
    max_retrieval: 50
```

### 动态工具加载

```python
import importlib
import os
from typing import List

class ToolLoader:
    """动态工具加载器"""

    def __init__(self, tools_dir: str = "custom_tools"):
        self.tools_dir = tools_dir
        self.loaded_tools = {}

    def load_tools_from_directory(self) -> List[BaseTool]:
        """从目录加载工具"""
        tools = []

        if not os.path.exists(self.tools_dir):
            return tools

        for filename in os.listdir(self.tools_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]

                try:
                    tool = self._load_tool_from_module(module_name)
                    if tool:
                        tools.append(tool)
                        self.loaded_tools[tool.name] = tool
                except Exception as e:
                    print(f"加载工具 {module_name} 失败: {e}")

        return tools

    def _load_tool_from_module(self, module_name: str) -> BaseTool:
        """从模块加载工具"""
        module_path = f"{self.tools_dir}.{module_name}"
        module = importlib.import_module(module_path)

        # 查找工具类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if (isinstance(attr, type) and
                issubclass(attr, BaseTool) and
                attr != BaseTool):

                return attr()

        return None

# 使用动态加载
tool_loader = ToolLoader("custom_tools")
custom_tools = tool_loader.load_tools_from_directory()

for tool in custom_tools:
    tool_registry.register_tool(tool.name, tool)
```

这个工具系统API文档提供了完整的工具使用指南、自定义开发方法和系统管理功能，帮助开发者充分利用和扩展工具生态。
