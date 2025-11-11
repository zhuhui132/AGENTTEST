# 🧠 LLM API

`src/llm/base.py` 提供大语言模型相关的抽象层，便于统一接入 OpenAI、HuggingFace 或自建模型。当前仓库仅包含基础接口与增强包装器，需要开发者自行实现具体模型类。

---

## 1. BaseLLM

```python
from src.llm.base import BaseLLM
from src.core import LLMConfig
```

### 1.1 关键接口
| 方法 | 说明 |
| --- | --- |
| `await initialize()` | 模型初始化（网络连接、权重加载等），成功返回 True |
| `await generate(prompt, config=None)` | 生成完整文本字符串 |
| `await generate_stream(prompt, config=None)` | 异步迭代器，逐块返回字符串 |
| `await embed(texts)` | 返回向量（单文本 -> `List[float]`，多文本 -> `List[List[float]]`）|
| `await count_tokens(text)` | 默认返回 `len(text)`，可按需重写 |
| `get_model_info()` | 返回模型名、初始化状态和配置字典 |

### 1.2 自定义实现示例
```python
class MyLLM(BaseLLM):
    async def _initialize_model(self):
        self._client = ...

    async def _generate_impl(self, prompt, config):
        resp = await self._client.generate(prompt, **config.__dict__)
        return resp.text

    async def _generate_stream_impl(self, prompt, config):
        async for chunk in self._client.generate_stream(prompt):
            yield chunk.text

    async def _embed_impl(self, texts):
        return await self._client.embed(texts)
```

> 注意：`BaseLLM` 内部已经处理了超时 (`asyncio.wait_for`) 与错误包装，子类只需关注 `_xxx_impl` 方法。

---

## 2. LLMWithTools

用于在调用模型前自动检测并执行工具。

```python
from src.llm.base import LLMWithTools

llm = LLMWithTools(config=LLMConfig(model_name="mock"))

@llm.register_tool("echo", lambda text: text, description="原样返回")
async def _():
    pass

response = await llm.generate_with_tools("请调用 echo 输出 hello", tools=["echo"])
```

- `register_tool(name, func, description, schema=None)`：注册异步或同步函数
- `_parse_tool_calls` 默认基于关键词匹配，可重写以支持更复杂的提示解析
- `generate_with_tools(prompt, tools=None)`：执行工具 → 拼接结果 → 调用 `generate`

---

## 3. RateLimitedLLM

为任意 `BaseLLM` 子类增加简单的每分钟请求速率限制。

```python
from src.llm.base import RateLimitedLLM

limited_llm = RateLimitedLLM(base_llm=my_llm, max_requests_per_minute=60)
text = await limited_llm.generate("你好")
```

- 每次调用会检查一分钟内的请求数量；超过限制会抛出 `LLMError`
- 适合包装第三方 API，避免超出配额

---

## 4. LLMConfig

```python
from src.core import LLMConfig

config = LLMConfig(
    model_name="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    timeout=30.0,
)
```
常用字段：
- `temperature`、`top_p`：随机性设置
- `timeout`：生成超时阈值（秒）
- `stop_sequences`：停止词列表

---

## 5. 集成建议

1. 将实际模型实现放置在 `src/llm/` 目录，继承 `BaseLLM`。
2. 在 `IntelligentAgent` 中注入自定义 LLM，或在工具层调用。
3. 需要缓存/重试/监控时，可自行编写包装器或参考 `RateLimitedLLM`。
4. 请勿在 `_generate_impl` 中直接使用 `eval` 等不安全方式处理外部请求。

> 当前仓库未附带真实 API Key 或 SDK，部署前请根据业务选择合适的模型提供方并实现适配。
