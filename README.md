# 🤖 Advanced Agent System

> 一个基于大语言模型的智能对话Agent系统，具备记忆系统、RAG检索、工具调用等完整功能。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

## ✨ 特性

- 🧠 **智能Agent**: 基于大语言模型的智能对话Agent
- 🧠 **记忆系统**: 多类型记忆支持（情景、语义、工作记忆、程序性记忆）
- 🔍 **RAG检索**: 检索增强生成，支持多种文档类型和高级检索策略
- 🛠️ **工具调用**: 灵活的工具系统，支持并发执行和自定义工具
- 📊 **性能监控**: 完整的指标收集和监控系统
- 🧪 **完整测试**: 多层次测试体系，覆盖单元、集成、端到端和性能测试
- 🔧 **模块化设计**: 清晰的模块化架构，易于扩展和维护
- 🧱 **全新架构**: `IntelligentAgent`、`MemorySystem`、`RAGSystem`、`ToolSystem` 重新实现，统一异步接口

### ♻️ 重构要点

- `IntelligentAgent` 现已完全实现 `BaseAgent` 接口，支持异步上下文构建与工具动态调度
- `MemorySystem`、`RAGSystem` 改为异步实现，内置相似度评分、容量淘汰和统计方法
- `ToolSystem` 支持异步注册与健康检查，并内置默认计算器工具，便于快速扩展
- `ContextManager` 提供轻量级上下文摘要、实体与意图推断，方便自定义增强
- 顶层 `src/__init__.py` 与根目录 `agent.py` 统一导出核心 API，方便第三方引入

## 🏗️ 项目结构

```
agentTest/
├── src/                    # 源代码
│   ├── core/               # 核心模块
│   │   ├── types.py        # 核心数据类型
│   │   ├── interfaces.py  # 基础接口定义
│   │   ├── exceptions.py   # 异常类定义
│   │   └── config.py      # 配置管理
│   ├── agents/             # Agent实现
│   ├── llm/               # 大语言模型
│   ├── memory/            # 记忆系统
│   ├── rag/               # RAG检索
│   ├── utils/             # 工具模块
│   └── ml/                # 机器学习模块
├── tests/                  # 测试代码
│   ├── unit/               # 单元测试
│   ├── integration/         # 集成测试
│   ├── e2e/                # 端到端测试
│   └── performance/         # 性能测试
├── docs/                   # 文档
│   ├── guide/               # 指南文档
│   ├── tutorials/           # 教程
│   ├── reference/           # 参考文档
│   └── api/                 # API文档
└── examples/               # 示例代码
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd agentTest

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
import asyncio
from src import Agent, IntelligentAgent, AgentConfig, LLMConfig

async def main():
    # 创建配置
    llm_config = LLMConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2048
    )

    agent_config = AgentConfig(
        name="my_assistant",
        llm_config=llm_config,
        memory_enabled=True,
        rag_enabled=True,
        tools_enabled=True
    )

    # 创建Agent实例
    agent = IntelligentAgent(agent_config)

    # 初始化
    await agent.initialize()

    # 处理消息
    response = await agent.process_message("你好，请介绍一下你的功能")
    print(response.content)
    print(f"推理过程: {response.reasoning}")
    print(f"置信度: {response.confidence}")
    print(f"处理时间: {response.processing_time}s")

if __name__ == "__main__":
    asyncio.run(main())

> 需要同步接口时，可直接使用根目录下的 `Agent` 包装类：

```python
from agent import Agent

legacy_agent = Agent("legacy_helper")
result = legacy_agent.process_message("请计算 12 * 8")
print(result["response"])
```

### 自定义工具

```python
from src.core import BaseTool, ToolResult

class CalculatorTool(BaseTool):
    async def execute(self, parameters):
        try:
            expression = parameters.get("expression", "")
            result = eval(expression)
            return ToolResult(
                tool_name="calculator",
                success=True,
                result=result,
                execution_time=0.1
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                success=False,
                error=str(e)
            )

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

# 注册工具
agent.register_tool("calculator", CalculatorTool())
```

### RAG系统使用

```python
from src import RAGSystem

async def rag_example():
    rag = RAGSystem()

    # 添加文档
    await rag.add_document(
        title="Python基础",
        content="Python是一种高级编程语言...",
        source="https://docs.python.org"
    )

    # 检索相关文档
    docs = await rag.retrieve("Python编程语言", limit=3)

    for doc in docs:
        print(f"标题: {doc.title}")
        print(f"内容片段: {doc.content[:100]}...")
        print(f"相关度: {doc.get('score', 0)}")
        print("-" * 50)
```

## 🧪 测试

运行完整测试套件：

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 运行端到端测试
pytest tests/e2e/

# 运行性能测试
pytest tests/performance/

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

测试覆盖率目标：
- 单元测试覆盖率 > 85%
- 集成测试覆盖率 > 75%
- 端到端成功率 > 95%

## 📊 性能指标

| 指标 | 目标值 | 当前值 |
|------|--------|----------|
| 响应时间 | < 2s | 测试中... |
| 并发处理 | 10+ | 测试中... |
| 内存使用 | < 1GB | 测试中... |
| 缓存命中率 | > 70% | 测试中... |
| 错误率 | < 1% | 测试中... |

## 📚 文档

详细文档请查看 [docs/](docs/) 目录：

- 📖 [指南文档](docs/guide/) - 项目导航和核心概念
- 📚 [教程文档](docs/tutorials/) - 开发流程和案例分析
- 🔧 [参考文档](docs/reference/) - API文档和技术细节
- 🧠 [知识库](docs/reference/深度学习与AI知识库.md) - 完整的AI知识库

### 核心概念

- **Agent**: 智能对话代理的核心组件
- **记忆系统**: 多层次的信息存储和检索
- **RAG**: 检索增强生成，结合知识库回答
- **工具调用**: 扩展Agent能力的工具接口
- **上下文管理**: 对话历史和上下文理解

## 🔧 配置

### 环境变量

```bash
export AGENT_DEBUG=true                    # 调试模式
export AGENT_LOG_LEVEL=INFO               # 日志级别
export AGENT_LLM_MODEL=gpt-3.5-turbo      # LLM模型
export AGENT_LLM_API_KEY=your-api-key     # API密钥
export AGENT_MEMORY_ENABLED=true           # 启用记忆
export AGENT_RAG_ENABLED=true              # 启用RAG
```

### 配置文件

创建 `config/agent/default.yaml`：

```yaml
name: "production_agent"
debug_mode: false
log_level: "INFO"

llm:
  model_name: "gpt-3.5-turbo"
  max_tokens: 2048
  temperature: 0.7
  timeout: 30.0

memory:
  max_memories: 10000
  retrieval_limit: 5
  importance_decay_rate: 0.99

rag:
  max_documents: 10000
  retrieval_limit: 5
  similarity_threshold: 0.7
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

tools:
  max_tools: 100
  default_timeout: 30.0
  parallel_execution: true
```

## 🤝 贡献

我们欢迎所有形式的贡献！请查看 [贡献指南](docs/reference/C-贡献指南.md) 了解详细信息。

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit hooks
pre-commit install

# 运行代码质量检查
flake8 src/
mypy src/
black src/
isort src/

# 运行测试
pytest
```

### 提交规范

- 使用清晰的commit消息
- 遵循[Conventional Commits](https://www.conventionalcommits.org/)
- 确保所有测试通过
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的启发和支持：

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [FastAPI](https://github.com/tiangolo/fastapi)

## 📞 联系

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [maintainer-email]

## 🗺️ 路线图

- [x] v1.0.0 - 基础Agent框架
- [x] v2.0.0 - 集成记忆和RAG系统
- [x] v3.0.0 - 完整的测试体系和知识库
- [ ] v3.1.0 - 性能优化和监控
- [ ] v3.2.0 - 多模态支持
- [ ] v4.0.0 - 分布式Agent系统

---

<div align="center">
  <p>如果这个项目对您有帮助，请给我们一个 ⭐️</p>
</div>

*最后更新: 2025-11-07*
