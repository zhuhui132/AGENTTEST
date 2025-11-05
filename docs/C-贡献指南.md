# 🤝 贡献指南

## 🎯 概述

感谢您对Agent测试方法论项目的关注！我们欢迎所有形式的贡献，包括但不限于代码、文档、测试用例、错误报告和功能建议。

---

## 🚀 开始贡献

### 环境准备

#### 系统要求
- Python 3.9+
- Git
- 文本编辑器或IDE

#### 开发环境设置
```bash
# 1. Fork项目到您的GitHub账号
# 2. 克隆您的Fork
git clone https://github.com/your-username/agent-testing.git
cd agent-testing

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. 创建开发分支
git checkout -b feature/your-feature-name

# 5. 安装pre-commit钩子
pre-commit install
```

---

## 📝 贡献类型

### 1. Bug报告

#### 报告前检查
- [ ] 检查是否已有相同Issue
- [ ] 确认使用最新版本
- [ ] 尝试在本地复现
- [ ] 收集足够的错误信息

#### 报告模板
```markdown
## Bug描述
简要描述遇到的问题

## 环境信息
- 操作系统: [如 macOS, Linux, Windows]
- Python版本: [如 3.9.0]
- 项目版本: [如 v1.0.0]

## 复现步骤
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为
描述您期望发生的情况

## 实际行为
描述实际发生的情况

## 错误信息
粘贴完整的错误堆栈
```

### 2. 功能请求

#### 请求模板
```markdown
## 功能描述
简要描述您希望添加的功能

## 动机
解释为什么这个功能有用

## 解决方案
描述您的解决方案

## 替代方案
描述您考虑的其他方案
```

---

## 🔧 开发规范

### 代码风格

#### Python代码规范
```python
# 遵循PEP 8规范
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ExampleClass:
    """Example class demonstrating coding standards."""

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.created_at = datetime.now()

    def process_data(self, data: List[str]) -> Dict:
        """Process input data and return results."""
        try:
            processed_data = [item.strip() for item in data if item]
            return {
                "status": "success",
                "processed_count": len(processed_data),
                "data": processed_data
            }
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
```

### 测试规范

#### 测试编写标准
```python
import pytest
from unittest.mock import Mock, patch
from src.agent import Agent

class TestAgent:
    """Agent类测试用例"""

    @pytest.fixture
    def agent_config(self):
        """测试配置"""
        return {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 100
        }

    @pytest.fixture
    def agent_instance(self, agent_config):
        """Agent实例"""
        return Agent("test_agent", agent_config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message(self, agent_instance):
        """测试消息处理功能"""
        # Arrange
        test_message = "Hello, agent!"

        # Act
        result = await agent_instance.process_message(test_message)

        # Assert
        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0
        assert result["status"] == "success"
```

---

## 🔄 开发流程

### Pre-commit钩子配置
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
```

### Pull Request流程

#### PR检查清单
- [ ] 代码符合项目规范
- [ ] 包含适当的测试
- [ ] 所有测试通过
- [ ] 代码覆盖率达标
- [ ] 文档已更新

#### PR描述模板
```markdown
## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 重大重构
- [ ] 文档更新

## 变更描述
简要描述这个PR的目的和内容

## 测试
描述您运行了哪些测试来验证这个变更

## 检查清单
- [ ] 代码遵循项目编码规范
- [ ] 添加了必要的测试
- [ ] 测试覆盖了变更的功能
- [ ] 文档已相应更新
```

---

## 📋 社区准则

### 行为准则

#### 尊重与包容
- 尊重不同的观点和经验
- 接受建设性的批评
- 专注于对社区最有利的事情
- 对其他社区成员表示同理心

#### 专业行为
- 使用友好和包容的语言
- 尊重不同的观点和经历
- 优雅地接受建设性批评
- 专注于对社区最有利的事情
- 对其他社区成员表示同理心

### 沟通渠道

#### 讨论区
- **技术讨论**: [GitHub Discussions](https://github.com/your-org/agent-testing/discussions)
- **问题反馈**: [GitHub Issues](https://github.com/your-org/agent-testing/issues)
- **经验分享**: [Community Forum](https://forum.agent-testing.org)

#### 联系方式
- **项目维护**: maintainer@agent-testing.org
- **商务合作**: business@agent-testing.org
- **技术咨询**: support@agent-testing.org

---

## 🎉 贡献者认可

### 贡献者类型
- **核心贡献者**: 频繁贡献核心代码
- **活跃贡献者**: 定期贡献代码或文档
- **社区贡献者**: 参与讨论和问题回答
- **首次贡献者**: 第一次贡献的用户

### 认可方式
- 在README中列出贡献者
- 在发布说明中感谢贡献者
- 在GitHub中给予维护权限
- 在社区活动中表彰

---

## 📄 许可证

### 版权声明
通过贡献代码，您同意：
1. 贡献的代码符合MIT许可证
2. 您拥有所贡献代码的版权
3. 您同意代码被项目使用

### 许可证条款
本项目采用MIT许可证，详见[LICENSE](../LICENSE)文件。

---

## 🆘 获取帮助

### 技术问题
- **GitHub Issues**: https://github.com/your-org/agent-testing/issues
- **讨论区**: https://github.com/your-org/agent-testing/discussions

### 社区渠道
- **Slack**: [邀请链接]
- **微信群**: [二维码]
- **邮件列表**: dev@agent-testing.org

---

*贡献指南持续更新，欢迎反馈和建议*
*最后更新时间: 2025-11-05*
*版本: v1.0.0*
