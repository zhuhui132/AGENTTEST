# 🤖 Agent API

简要介绍 `src/agents/agent.py` 暴露的两个类：`IntelligentAgent`（异步）与 `Agent`（同步包装）。

---

## 1. IntelligentAgent

```python
from src.agents.agent import IntelligentAgent
from src.core import AgentMessage

agent = IntelligentAgent()
response = await agent.process_message(
    AgentMessage(role="user", content="你好")
)
print(response.content, response.confidence)
```

### 1.1 构造参数
```python
IntelligentAgent(
    config: AgentConfig | None = None,
    *,
    memory_config: MemoryConfig | None = None,
    rag_config: RAGConfig | None = None,
    memory_system=None,
    rag_system=None,
    tools=None,
    context_manager=None,
)
```
- 任意参数可注入自定义实现；未提供时使用默认内存版组件。
- `config` 为空时会创建默认 `AgentConfig`。

### 1.2 核心方法
| 方法 | 说明 |
| --- | --- |
| `await initialize()` | 可选的初始化钩子，默认返回 True |
| `await process_message(message, config=None)` | 主流程：构造上下文 → 检索记忆/文档 → 工具执行 → 生成回复 |
| `await shutdown()` | 触发 `cleanup`，将状态改为 `SHUTDOWN` |
| `get_status()` | 返回当前状态、创建时间、历史长度等信息 |

> 若 `process_message` 过程中出现异常，状态会切换至 `AgentState.ERROR` 并返回带错误信息的 `AgentResponse`。

### 1.3 可重写/扩展点
- `_plan_tool_usage(content)`：根据消息内容决定调用哪些工具
- `_generate_suggestion(...)`：自定义最终输出文本
- `_calculate_confidence(...)`：修改置信度评估策略

---

## 2. 同步包装器 `Agent`

```python
from agent import Agent

legacy_agent = Agent("demo")
result = legacy_agent.process_message("两位数乘法 12*8")
print(result["response"], result["confidence"])
```

特点：
- 自动创建内置 `IntelligentAgent`
- `process_message` 内部调用 `asyncio.run`，适合 CLI 或脚本
- 返回字典包含 `response`、`docs_used`、`confidence` 等简化字段

---

## 3. 典型用法

### 3.1 注册新工具
```python
from src.utils.tools import ToolSystem
from src.core.interfaces import BaseTool

class EchoTool(BaseTool):
    async def execute(self, parameters, config=None):
        return ToolResult(tool_name="echo", success=True, result=parameters.get("text", ""))

tool_system = ToolSystem()
await tool_system.register_tool("echo", EchoTool())
agent = IntelligentAgent(tools=tool_system)
```

### 3.2 更换记忆实现
```python
class DummyMemory(BaseMemory):
    async def add_memory(self, content, **kwargs):
        return "dummy"
    async def retrieve(self, query, limit=5, memory_type=None):
        return []
    async def update_memory(self, memory_id, updates):
        return True
    async def delete_memory(self, memory_id):
        return True
    async def cleanup(self):
        return 0

agent = IntelligentAgent(memory_system=DummyMemory())
```

---

## 4. 错误处理

```python
from src.core import ToolError

try:
    resp = await agent.process_message("触发错误")
    if resp.metadata.get("error"):
        print("Agent 返回错误: ", resp.metadata["error"])
except ToolError as exc:
    logger.warning("工具执行失败: %s", exc)
```

当工具/检索失败时，Agent 会在响应中附带错误信息，保持对话不中断。严重异常会抛出对应的 `AgentError` 子类。

---

更多细节请结合源码与 `docs/guides/04-技术实现指南.md` 阅读。
