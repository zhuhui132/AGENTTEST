# 🔧 核心 API

介绍 `src/core/` 提供的基础类型、配置结构与常见异常。示例均基于 Python 3.10+。

---

## 1. 主要数据结构

```python
from src.core import (
    AgentMessage, AgentResponse, AgentState,
    MemoryItem, Document, ToolResult,
    ContextInfo, MessageType
)
```

### 1.1 AgentMessage
```python
msg = AgentMessage(
    role="user",
    content="请介绍一下自己",
    message_type=MessageType.TEXT,
    metadata={"channel": "web"}
)
print(msg.id, msg.timestamp)
```
- `role`：`user / assistant / system / tool`
- `message_type`：默认 `MessageType.TEXT`
- 自动生成 `id` 和 `timestamp`

### 1.2 AgentResponse
```python
resp = AgentResponse(
    content="您好！我是 AgentTest...",
    reasoning="基于历史消息生成",
    confidence=0.82,
    sources=["docs/README.md"],
    tool_calls=[],
)
```
字段说明：
- `content`：最终回复文本
- `reasoning`：可选，记录推理摘要
- `sources`：引用的文档/知识 ID
- `tool_calls`：工具调用详情列表（来自 `ToolResult.__dict__`）

### 1.3 MemoryItem & Document
```python
memory = MemoryItem(content="总部在北京", importance=0.9)
doc = Document(title="产品手册", content="...", source="internal")
```
两者均带有 `id`, `created_at`, `metadata`, `tags` 等通用字段，便于扩展。

### 1.4 ToolResult
```python
result = ToolResult(
    tool_name="calculator",
    success=True,
    result=42,
    execution_time=0.02
)
```
- `success=False` 时可设置 `error` 字段，用于响应兜底。

### 1.5 ContextInfo
`ContextManager` 返回结构，包含 `context_window`, `summary`, `user_intent`, `key_entities` 等信息。

---

## 2. 配置结构

```python
from src.core import AgentConfig, LLMConfig, MemoryConfig, RAGConfig, ToolConfig
```

### 2.1 AgentConfig
```python
agent_cfg = AgentConfig(
    name="demo_agent",
    llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
    max_context_length=4000,
    memory_enabled=True,
    rag_enabled=True,
    tools_enabled=True,
)
```
常用属性：
- `max_concurrent_requests`: 并发上限
- `response_timeout`: 单次响应超时（秒）
- `cache_enabled` 与 `cache_ttl`

### 2.2 MemoryConfig / RAGConfig / ToolConfig
均为简单的数据类，可在注入自定义实现时直接复用。例如：
```python
memory_cfg = MemoryConfig(max_memories=5000, retrieval_limit=3)
rag_cfg = RAGConfig(similarity_threshold=0.65)
tool_cfg = ToolConfig(max_tools=50, parallel_execution=True)
```

---

## 3. 配置管理器

```python
from src.core.config import ConfigManager, get_env_config, merge_configs
```

### 3.1 加载 YAML / JSON
```python
manager = ConfigManager("config")
agent_settings = manager.get_agent_config("default")
```
目录结构默认 `config/<name>.agent.yaml`，也可通过 `save_config` 生成基础模板。

### 3.2 环境变量覆盖
```python
env_overrides = get_env_config()
final_cfg = merge_configs(agent_settings.__dict__, env_overrides)
```
`merge_configs` 会递归合并嵌套字典，适合注入运行时开关。

---

## 4. 异常体系

```python
from src.core import AgentError, MemoryError, ToolError, RAGError, LLMError
```
- `AgentError`：顶层错误，子类包含 `error_code`、`details`
- `MemoryError / ToolError / RAGError / LLMError`：对应子系统异常

示例：
```python
try:
    await agent.process_message("...")
except ToolError as exc:
    logger.warning("工具失败：%s", exc)
```

---

## 5. 性能指标（可选）

`PerformanceMetrics` 数据类可用于记录响应时间、token 使用、缓存命中率等：
```python
from src.core import PerformanceMetrics
metrics = PerformanceMetrics(response_time=0.8, cache_hit_rate=0.7)
```

---

## 6. 实用提示

1. 所有核心类型均使用 `dataclass`，可通过 `asdict()` 快速转换为字典。
2. 若需序列化为 JSON，注意 datetime 默认需 `default=str` 处理。
3. 新增公共类型或配置后，请同步更新本文件与 `src/__init__.py` 导出列表。
