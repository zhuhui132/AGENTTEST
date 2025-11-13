"""
自定义Agent开发示例

演示如何开发具有特定功能的自定义Agent。
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.agents.intelligent_agent import IntelligentAgent
from src.core.types import (
    AgentConfig, LLMConfig, AgentResponse, Message,
    ToolResult, ToolSchema
)
from src.core.interfaces import BaseTool
from src.utils.logger import get_logger


# ============================================================================
# 1. 专用领域Agent示例：代码助手
# ============================================================================

class CodeAssistantAgent(IntelligentAgent):
    """代码助手Agent，专门用于代码生成、分析和优化"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.logger = get_logger("code_assistant")

        # 代码相关统计
        self.code_stats = {
            "code_generated": 0,
            "bugs_found": 0,
            "optimizations_suggested": 0,
            "languages_used": {}
        }

    async def process_code_request(
        self,
        request_type: str,
        code: str = None,
        language: str = "python",
        **kwargs
    ) -> AgentResponse:
        """
        处理代码相关请求

        Args:
            request_type: 请求类型 (generate, analyze, debug, optimize, explain)
            code: 代码内容
            language: 编程语言
            **kwargs: 其他参数
        """
        try:
            # 构建专用提示
            specialized_prompt = self._build_code_prompt(
                request_type, code, language, **kwargs
            )

            # 调用基础处理方法
            response = await self.process_message(specialized_prompt)

            # 更新代码统计
            self._update_code_stats(request_type, language, response)

            # 添加代码专用元数据
            response.metadata.update({
                "request_type": request_type,
                "language": language,
                "code_length": len(code) if code else 0,
                "specialization": "code_assistant"
            })

            return response

        except Exception as e:
            self.logger.error(f"代码处理失败: {e}")
            raise

    def _build_code_prompt(
        self,
        request_type: str,
        code: str = None,
        language: str = "python",
        **kwargs
    ) -> str:
        """构建代码专用提示"""

        prompts = {
            "generate": f"""
作为专业的{language}程序员，请生成以下功能的代码：
需求: {kwargs.get('requirement', '未指定需求')}
语言: {language}
约束条件: {kwargs.get('constraints', '无')}
输出格式: 代码块形式
""",
            "analyze": f"""
请分析以下{language}代码的质量和结构：

```{language}
{code}
```

请提供：
1. 代码质量评估 (1-10分)
2. 潜在问题识别
3. 改进建议
4. 最佳实践建议
""",
            "debug": f"""
请调试以下{language}代码中的错误：

```{language}
{code}
```

错误信息: {kwargs.get('error_message', '未提供错误信息')}
期望行为: {kwargs.get('expected_behavior', '未指定期望行为')}

请提供：
1. 错误定位
2. 错误原因分析
3. 修复方案
4. 修复后的完整代码
""",
            "optimize": f"""
请优化以下{language}代码的性能：

```{language}
{code}
```

优化目标: {kwargs.get('optimization_goal', '整体性能')}
性能要求: {kwargs.get('performance_requirements', '无特殊要求')}

请提供：
1. 性能瓶颈分析
2. 优化策略
3. 优化后的代码
4. 预期性能提升
""",
            "explain": f"""
请详细解释以下{language}代码的功能和工作原理：

```{language}
{code}
```

解释深度: {kwargs.get('detail_level', '中等')}
目标受众: {kwargs.get('audience', '有基础编程知识的开发者')}

请提供：
1. 整体功能概述
2. 关键算法/逻辑解释
3. 代码结构分析
4. 设计模式识别
"""
        }

        return prompts.get(request_type, "请处理代码相关请求")

    def _update_code_stats(
        self,
        request_type: str,
        language: str,
        response: AgentResponse
    ):
        """更新代码处理统计"""
        if request_type == "generate":
            self.code_stats["code_generated"] += 1
        elif request_type == "analyze":
            # 假设分析发现了问题
            self.code_stats["bugs_found"] += 1
        elif request_type == "optimize":
            self.code_stats["optimizations_suggested"] += 1

        # 更新语言使用统计
        self.code_stats["languages_used"][language] = \
            self.code_stats["languages_used"].get(language, 0) + 1

    def get_code_stats(self) -> Dict[str, Any]:
        """获取代码处理统计"""
        return self.code_stats.copy()


# ============================================================================
# 2. 专用工具示例：代码分析工具
# ============================================================================

class CodeAnalysisTool(BaseTool):
    """代码分析工具"""

    def __init__(self):
        self.name = "code_analysis"
        self.logger = get_logger("code_analysis_tool")

    async def execute(self, parameters: dict) -> ToolResult:
        """执行代码分析"""
        try:
            code = parameters.get("code", "")
            language = parameters.get("language", "python")
            analysis_type = parameters.get("analysis_type", "basic")

            if not code:
                return ToolResult(
                    success=False,
                    error="代码内容不能为空",
                    tool_name=self.name
                )

            start_time = time.time()

            # 执行分析
            if analysis_type == "basic":
                result = self._basic_analysis(code, language)
            elif analysis_type == "security":
                result = self._security_analysis(code, language)
            elif analysis_type == "performance":
                result = self._performance_analysis(code, language)
            elif analysis_type == "complexity":
                result = self._complexity_analysis(code, language)
            else:
                raise ValueError(f"不支持的分析类型: {analysis_type}")

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=execution_time,
                metadata={
                    "language": language,
                    "analysis_type": analysis_type,
                    "code_lines": len(code.split('\n'))
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"代码分析失败: {str(e)}",
                tool_name=self.name
            )

    def _basic_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """基础代码分析"""
        lines = code.split('\n')

        analysis = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": self._count_comments(code, language),
            "indentation_issues": self._check_indentation(code, language),
            "syntax_issues": self._check_syntax(code, language),
            "naming_conventions": self._check_naming_conventions(code, language)
        }

        # 计算代码质量评分
        quality_score = self._calculate_quality_score(analysis)
        analysis["quality_score"] = quality_score

        return analysis

    def _security_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """安全性分析"""
        security_patterns = {
            "python": {
                "sql_injection": [r"execute\s*\("],
                "command_injection": [r"os\.system", r"subprocess\.call"],
                "hardcoded_secrets": [r"password\s*=\s*['\"][^'\"]+['\"]"],
                "unsafe_eval": [r"eval\s*\(", r"exec\s*\("]
            },
            "javascript": {
                "xss": [r"innerHTML\s*="],
                "sql_injection": [r"query\s*\("],
                "hardcoded_secrets": [r"password\s*=\s*['\"][^'\"]+['\"]"],
                "unsafe_eval": [r"eval\s*\("]
            }
        }

        patterns = security_patterns.get(language, {})
        vulnerabilities = []

        for vuln_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                import re
                if re.search(pattern, code):
                    vulnerabilities.append({
                        "type": vuln_type,
                        "pattern": pattern,
                        "severity": "high" if vuln_type in ["sql_injection", "command_injection"] else "medium"
                    })

        return {
            "vulnerabilities_found": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "security_score": max(0, 100 - len(vulnerabilities) * 20)
        }

    def _performance_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """性能分析"""
        performance_issues = []

        if language == "python":
            # 检查常见性能问题
            if "for i in range(len(" in code:
                performance_issues.append({
                    "issue": "使用range(len())进行迭代",
                    "suggestion": "使用直接迭代或enumerate()",
                    "severity": "low"
                })

            if ".append(" in code and "while" in code:
                performance_issues.append({
                    "issue": "可能的循环性能问题",
                    "suggestion": "考虑使用列表推导式或生成器",
                    "severity": "medium"
                })

        return {
            "performance_issues": performance_issues,
            "performance_score": max(0, 100 - len(performance_issues) * 15)
        }

    def _complexity_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """复杂度分析"""
        # 计算圈复杂度
        complexity_keywords = {
            "python": ["if", "elif", "for", "while", "except", "with"],
            "javascript": ["if", "else if", "for", "while", "catch", "switch"]
        }

        keywords = complexity_keywords.get(language, [])
        cyclomatic_complexity = 1  # 基础复杂度

        for keyword in keywords:
            cyclomatic_complexity += code.count(keyword)

        # 计算认知复杂度（简化版）
        cognitive_complexity = self._calculate_cognitive_complexity(code, language)

        return {
            "cyclomatic_complexity": cyclomatic_complexity,
            "cognitive_complexity": cognitive_complexity,
            "complexity_level": self._get_complexity_level(cyclomatic_complexity),
            "recommendations": self._get_complexity_recommendations(cyclomatic_complexity)
        }

    def _count_comments(self, code: str, language: str) -> int:
        """计算注释行数"""
        comment_patterns = {
            "python": [r"#.*"],
            "javascript": [r"//.*", r"/\*.*?\*/"]
        }

        patterns = comment_patterns.get(language, [])
        comment_count = 0

        import re
        for pattern in patterns:
            comment_count += len(re.findall(pattern, code, re.MULTILINE))

        return comment_count

    def _check_indentation(self, code: str, language: str) -> List[str]:
        """检查缩进问题"""
        if language != "python":
            return []

        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip():
                # 检查是否使用制表符混合空格
                if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                    issues.append(f"第{i}行：制表符和空格混合使用")

        return issues

    def _check_syntax(self, code: str, language: str) -> List[str]:
        """检查语法问题（基础版）"""
        issues = []

        if language == "python":
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                issues.append(f"语法错误: {e}")

        return issues

    def _check_naming_conventions(self, code: str, language: str) -> Dict[str, Any]:
        """检查命名规范"""
        issues = []

        if language == "python":
            import re
            # 检查变量名（应该使用snake_case）
            variable_pattern = r'\b([a-z_][a-z0-9_]*)\s*='
            variables = re.findall(variable_pattern, code)

            for var in variables:
                if not re.match(r'^[a-z_][a-z0-9_]*$', var):
                    issues.append(f"变量名 '{var}' 不符合snake_case规范")

        return {
            "naming_issues": issues,
            "convention_compliance": len(issues) == 0
        }

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """计算代码质量评分"""
        score = 100.0

        # 扣除问题分数
        score -= len(analysis.get("indentation_issues", [])) * 5
        score -= len(analysis.get("syntax_issues", [])) * 10
        score -= len(analysis.get("naming_issues", [])) * 3

        # 注释覆盖率加分
        comment_ratio = analysis.get("comment_lines", 0) / max(1, analysis.get("non_empty_lines", 1))
        if comment_ratio > 0.2:
            score += 5

        return max(0, min(100, score))

    def _calculate_cognitive_complexity(self, code: str, language: str) -> int:
        """计算认知复杂度（简化版）"""
        # 这是一个简化实现，实际认知复杂度计算更复杂
        nesting_level = 0
        complexity = 0

        # 基础复杂度因素
        complexity_keywords = ["if", "for", "while", "try", "except"]
        for keyword in complexity_keywords:
            complexity += code.count(keyword)

        # 嵌套复杂度
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('if ', 'for ', 'while ', 'try:')):
                nesting_level += 1
                complexity += nesting_level
            elif stripped in ('else:', 'elif ', 'except ', 'finally:'):
                complexity += nesting_level

        return complexity

    def _get_complexity_level(self, complexity: int) -> str:
        """获取复杂度等级"""
        if complexity <= 5:
            return "low"
        elif complexity <= 10:
            return "medium"
        elif complexity <= 20:
            return "high"
        else:
            return "very_high"

    def _get_complexity_recommendations(self, complexity: int) -> List[str]:
        """获取复杂度建议"""
        recommendations = []

        if complexity > 10:
            recommendations.append("考虑将复杂函数分解为多个小函数")

        if complexity > 15:
            recommendations.append("使用设计模式简化代码结构")

        if complexity > 20:
            recommendations.append("需要重构代码以降低复杂度")

        return recommendations

    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        return ToolSchema(
            name=self.name,
            description="代码质量分析工具",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要分析的代码"
                    },
                    "language": {
                        "type": "string",
                        "description": "编程语言",
                        "enum": ["python", "javascript", "java", "cpp"],
                        "default": "python"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "分析类型",
                        "enum": ["basic", "security", "performance", "complexity"],
                        "default": "basic"
                    }
                },
                "required": ["code"]
            }
        )


# ============================================================================
# 3. 使用示例
# ============================================================================

async def code_assistant_example():
    """代码助手使用示例"""

    # 配置代码助手
    config = AgentConfig(
        name="code_assistant",
        llm_config=LLMConfig(
            model_name="gpt-4",
            api_key="your-api-key",
            temperature=0.3  # 代码生成使用较低温度
        ),
        memory_enabled=True,
        tools_enabled=True
    )

    # 创建代码助手
    assistant = CodeAssistantAgent(config)
    await assistant.initialize()

    # 注册代码分析工具
    code_tool = CodeAnalysisTool()
    await assistant.register_tool("code_analysis", code_tool.execute)

    print("=== 代码助手示例 ===\n")

    # 示例1：代码生成
    print("1. 代码生成示例:")
    response = await assistant.process_code_request(
        request_type="generate",
        requirement="实现一个快速排序算法",
        language="python"
    )
    print(f"生成的代码:\n{response.content}\n")

    # 示例2：代码分析
    print("2. 代码分析示例:")
    sample_code = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)
"""

    response = await assistant.process_code_request(
        request_type="analyze",
        code=sample_code,
        language="python"
    )
    print(f"分析结果:\n{response.content}\n")

    # 示例3：使用代码分析工具
    print("3. 代码分析工具使用:")
    tool_response = await code_tool.execute({
        "code": sample_code,
        "language": "python",
        "analysis_type": "complexity"
    })

    if tool_response.success:
        analysis = tool_response.result
        print(f"复杂度分析结果:")
        print(f"- 圈复杂度: {analysis['cyclomatic_complexity']}")
        print(f"- 复杂度等级: {analysis['complexity_level']}")
        print(f"- 建议: {', '.join(analysis['recommendations'])}\n")

    # 获取代码统计
    stats = assistant.get_code_stats()
    print("4. 代码处理统计:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

    await assistant.cleanup()


# ============================================================================
# 4. 自定义Agent开发指南
# ============================================================================

class CustomAgentTemplate(IntelligentAgent):
    """
    自定义Agent模板

    开发指南：
    1. 继承IntelligentAgent
    2. 实现专用功能方法
    3. 添加专用统计
    4. 重写提示构建
    5. 实现专用工具
    """

    def __init__(self, config: AgentConfig, domain: str):
        super().__init__(config)
        self.domain = domain
        self.logger = get_logger(f"{domain}_agent")

        # 领域专用统计
        self.domain_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "specialized_operations": 0,
            "user_satisfaction": 0.0
        }

    async def specialized_process(self, operation: str, **kwargs) -> AgentResponse:
        """
        专用处理方法模板

        Args:
            operation: 操作类型
            **kwargs: 操作参数
        """
        try:
            # 1. 构建专用提示
            specialized_prompt = self._build_specialized_prompt(operation, **kwargs)

            # 2. 调用基础处理
            response = await self.process_message(specialized_prompt)

            # 3. 更新专用统计
            self._update_domain_stats(operation, response)

            # 4. 添加专用元数据
            response.metadata.update({
                "domain": self.domain,
                "operation": operation,
                "specialized": True
            })

            return response

        except Exception as e:
            self.logger.error(f"{self.domain}处理失败: {e}")
            raise

    def _build_specialized_prompt(self, operation: str, **kwargs) -> str:
        """构建专用提示（需要子类实现）"""
        raise NotImplementedError("子类必须实现此方法")

    def _update_domain_stats(self, operation: str, response: AgentResponse):
        """更新领域统计（需要子类实现）"""
        self.domain_stats["total_requests"] += 1
        if not response.error:
            self.domain_stats["successful_requests"] += 1
            self.domain_stats["specialized_operations"] += 1

    def get_domain_stats(self) -> Dict[str, Any]:
        """获取领域专用统计"""
        return self.domain_stats.copy()


# ============================================================================
# 5. 开发最佳实践
# ============================================================================

class BestPracticesGuide:
    """自定义Agent开发最佳实践指南"""

    @staticmethod
    def design_principles():
        """设计原则"""
        return {
            "single_responsibility": "每个Agent专注于特定领域",
            "extensibility": "设计时考虑未来扩展需求",
            "maintainability": "保持代码清晰和文档完整",
            "performance": "优化响应时间和资源使用",
            "reliability": "实现错误处理和恢复机制"
        }

    @staticmethod
    def implementation_tips():
        """实现技巧"""
        return {
            "prompt_engineering": "精心设计领域专用提示词",
            "tool_integration": "开发领域相关的专用工具",
            "memory_utilization": "有效利用记忆系统存储领域知识",
            "error_handling": "实现优雅的错误处理和用户反馈",
            "monitoring": "添加领域专用的监控指标"
        }

    @staticmethod
    def testing_strategy():
        """测试策略"""
        return {
            "unit_tests": "测试每个专用功能的正确性",
            "integration_tests": "测试与基础系统的集成",
            "performance_tests": "验证性能要求",
            "user_acceptance_tests": "确保满足用户需求",
            "regression_tests": "防止功能回退"
        }


# ============================================================================
# 主程序入口
# ============================================================================

async def main():
    """主程序入口"""
    print("=== 自定义Agent开发示例 ===\n")

    # 运行代码助手示例
    await code_assistant_example()

    print("\n=== 开发指南 ===")

    # 显示设计原则
    print("\n设计原则:")
    principles = BestPracticesGuide.design_principles()
    for principle, description in principles.items():
        print(f"- {principle}: {description}")

    # 显示实现技巧
    print("\n实现技巧:")
    tips = BestPracticesGuide.implementation_tips()
    for tip, description in tips.items():
        print(f"- {tip}: {description}")

    print("\n=== 开发完成 ===")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
