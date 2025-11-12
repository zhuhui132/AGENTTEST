#!/usr/bin/env python3
"""
AI学习交互式Q&A系统

为学生提供智能问答服务，支持多种问题类型和个性化回答。

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-12
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import importlib.util


class InteractiveQA:
    """交互式Q&A系统"""

    def __init__(self):
        self.qa_history = []
        self.knowledge_base_path = "/Users/a58/Downloads/catpawAi/agentTest"
        self.user_session = {}

        # 问题分类
        self.question_categories = {
            "concept": "概念理解问题",
            "technical": "技术实现问题",
            "practice": "实践操作问题",
            "career": "职业发展问题",
            "resources": "学习资源问题",
            "troubleshooting": "故障排除问题",
            "advanced": "进阶研究问题"
        }

        # 预设知识库
        self.knowledge_answers = {
            "transformer": {
                "什么是Transformer": "Transformer是一种基于注意力机制的深度学习模型架构，由Google在2017年提出。它完全依赖注意力机制，不使用RNN或CNN的循环或卷积结构。",
                "注意力机制如何工作": "注意力机制通过计算查询（Q）、键（K）、值（V）三个矩阵，实现对输入序列的加权关注，从而捕捉长距离依赖关系。",
                "位置编码的作用": "由于Transformer没有时序结构，位置编码为每个位置提供位置信息，使模型理解输入序列的顺序关系。"
            },
            "llm": {
                "什么是大语言模型": "大语言模型是具有数百亿到数万亿参数的语言模型，通过在海量文本数据上训练获得强大的语言理解和生成能力。",
                "涌现能力是什么": "涌现能力是指当模型规模达到一定程度时，突然出现的、小模型不具备的能力，如少样本学习、链式推理等。",
                "什么是少样本学习": "少样本学习是指模型在很少示例（如1-10个）就能学习新任务的能力，无需大量训练数据。"
            },
            "testing": {
                "如何运行测试": "使用pytest命令运行测试：\n```bash\npytest tests/unit/evolution/test_llm_evolution.py -v\n```\n参数说明：-v显示详细输出，--tb=short显示简短错误信息。",
                "测试失败怎么办": "1. 检查错误信息中的文件名和行号\n2. 确认依赖是否正确安装\n3. 检查测试数据是否存在\n4. 查看相关文档了解测试前提条件",
                "覆盖率如何查看": "使用--cov参数：\n```bash\npytest --cov=src tests/\n```\n生成HTML报告：\n```bash\npytest --cov=src --cov-report=html\n```"
            },
            "learning": {
                "如何选择学习路径": "根据你的背景和目标选择：\n• 研究背景→研究型路径\n• 工程背景→工程型路径\n• 产品背景→产品型路径\n• 初学者→入门型路径",
                "每天学习多长时间": "建议每天投入2-3小时，保持学习连续性。可以根据实际情况调整，但最好保持每周至少10小时的学习时间。",
                "如何验证学习效果": "通过以下方式验证：\n1. 完成相关测试用例\n2. 能够独立实现学到的算法\n3. 能够向他人解释概念\n4. 完成项目实践"
            }
        }

    def start_session(self) -> None:
        """开始Q&A会话"""
        print("=" * 60)
        print("🎓 AI学习助手 - 交互式Q&A系统")
        print("=" * 60)
        print("📚 知识库覆盖：AI技术发展历程 + 完整工程实践")
        print("🎯 支持的问题类型：")
        for category, description in self.question_categories.items():
            print(f"  • {category}: {description}")
        print("\n🔍 输入你的问题，或输入 'help' 查看帮助")
        print("💡 输入 'quit' 退出系统")
        print("=" * 60)

        while True:
            try:
                question = input("\n🤔 请输入你的问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 感谢使用，学习愉快！")
                    break
                elif question.lower() in ['help', '帮助', 'h']:
                    self._show_help()
                elif question.lower() in ['history', '历史']:
                    self._show_history()
                elif question.lower() in ['stats', '统计']:
                    self._show_statistics()
                elif not question:
                    continue
                else:
                    answer = self._answer_question(question)
                    self._display_answer(question, answer)

            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                print("💡 请重新输入问题或联系管理员")

    def _answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题"""
        # 记录问题
        question_data = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "category": self._classify_question(question)
        }

        # 尝试从知识库回答
        kb_answer = self._search_knowledge_base(question)

        if kb_answer:
            answer = {
                "source": "knowledge_base",
                "confidence": 0.9,
                "answer": kb_answer,
                "related_resources": self._find_related_resources(question)
            }
        else:
            # 尝试从文档搜索回答
            doc_answer = self._search_documents(question)

            if doc_answer:
                answer = {
                    "source": "document_search",
                    "confidence": 0.7,
                    "answer": doc_answer,
                    "related_resources": self._find_related_resources(question)
                }
            else:
                # 生成通用回答
                answer = {
                    "source": "general_response",
                    "confidence": 0.5,
                    "answer": self._generate_general_answer(question),
                    "related_resources": self._find_related_resources(question)
                }

        # 添加到历史记录
        question_data["answer"] = answer
        self.qa_history.append(question_data)

        return answer

    def _classify_question(self, question: str) -> str:
        """分类问题"""
        question_lower = question.lower()

        # 概念理解问题
        if any(word in question_lower for word in ["什么是", "什么是", "什么叫", "解释", "概念"]):
            return "concept"

        # 技术实现问题
        elif any(word in question_lower for word in ["如何", "怎么", "实现", "代码", "编程"]):
            return "technical"

        # 实践操作问题
        elif any(word in question_lower for word in ["运行", "执行", "测试", "安装", "配置"]):
            return "practice"

        # 职业发展问题
        elif any(word in question_lower for word in ["工作", "职业", "就业", "发展", "路径"]):
            return "career"

        # 学习资源问题
        elif any(word in question_lower for word in ["学习", "资料", "资源", "教程", "课程"]):
            return "resources"

        # 故障排除问题
        elif any(word in question_lower for word in ["错误", "问题", "故障", "失败", "调试"]):
            return "troubleshooting"

        # 进阶研究问题
        elif any(word in question_lower for word in ["研究", "论文", "创新", "前沿", "最新"]):
            return "advanced"

        return "general"

    def _search_knowledge_base(self, question: str) -> Optional[str]:
        """搜索知识库"""
        question_lower = question.lower()

        # 搜索Transformer相关
        if any(word in question_lower for word in ["transformer", "注意力", "位置编码"]):
            return self._find_best_match(question, self.knowledge_answers["transformer"])

        # 搜索LLM相关
        elif any(word in question_lower for word in ["llm", "大语言模型", "涌现", "少样本"]):
            return self._find_best_match(question, self.knowledge_answers["llm"])

        # 搜索测试相关
        elif any(word in question_lower for word in ["测试", "pytest", "覆盖率"]):
            return self._find_best_match(question, self.knowledge_answers["testing"])

        # 搜索学习相关
        elif any(word in question_lower for word in ["学习", "路径", "时间", "效果"]):
            return self._find_best_match(question, self.knowledge_answers["learning"])

        return None

    def _find_best_match(self, question: str, answer_dict: Dict[str, str]) -> str:
        """找到最佳匹配"""
        question_lower = question.lower()

        # 计算关键词匹配度
        best_match = None
        best_score = 0

        for key, answer in answer_dict.items():
            key_lower = key.lower()
            # 简单的关键词匹配
            score = 0
            for word in question_lower.split():
                if word in key_lower or key_lower in word:
                    score += 1
                elif word in key_lower.split():
                    score += 0.5
                elif key_lower in question_lower:
                    score += 0.8

            if score > best_score:
                best_score = score
                best_match = answer

        return best_match if best_score > 0 else next(iter(answer_dict.values()))

    def _search_documents(self, question: str) -> Optional[str]:
        """搜索文档"""
        # 这里可以实现文档搜索功能
        # 暂时返回None，表示没有找到相关文档
        return None

    def _generate_general_answer(self, question: str) -> str:
        """生成通用回答"""
        category = self._classify_question(question)

        general_answers = {
            "concept": f"关于'{question}'的概念问题，建议你：\n1. 首先查看相关知识文档\n2. 运行相关测试用例加深理解\n3. 查阅官方文档获取权威解释",

            "technical": f"关于'{question}'的技术问题，建议你：\n1. 查看相关的代码实现\n2. 运行测试用例了解预期行为\n3. 参考文档中的实现细节\n4. 如果遇到具体错误，请提供错误信息",

            "practice": f"关于'{question}'的实践问题，建议你：\n1. 检查环境配置\n2. 确认依赖是否正确安装\n3. 查看错误日志获取详细信息\n4. 参考测试用例的预期结果",

            "career": f"关于'{question}'的职业问题，建议你：\n1. 评估自己的技能水平和兴趣\n2. 了解不同职业路径的要求\n3. 制定适合自己的学习计划\n4. 积累相关项目经验",

            "resources": f"关于'{question}'的资源问题，建议你：\n1. 查看推荐的学习路径\n2. 访问相关的在线课程\n3. 参与开源项目实践\n4. 加入学习社区交流",

            "troubleshooting": f"关于'{question}'的故障问题，建议你：\n1. 仔细阅读错误信息\n2. 检查环境配置和依赖\n3. 尝试在干净的运行环境中重试\n4. 查看相关的调试指南",

            "advanced": f"关于'{question}'的进阶问题，建议你：\n1. 阅读相关的研究论文\n2. 关注最新的技术发展\n3. 参与相关的研究项目\n4. 与领域专家交流讨论"
        }

        return general_answers.get(category, f"关于'{question}'的问题，我建议你查看相关的技术文档或寻求更具体的帮助。")

    def _find_related_resources(self, question: str) -> List[str]:
        """查找相关资源"""
        resources = []
        question_lower = question.lower()

        # 基于问题类型推荐资源
        if any(word in question_lower for word in ["transformer", "注意力"]):
            resources.extend([
                "📖 docs/knowledge/llm/evolution/03-transformer-revolution.md",
                "🧪 tests/unit/evolution/test_transformer_revolution.py",
                "📚 推荐论文: 'Attention Is All You Need'"
            ])

        elif any(word in question_lower for word in ["llm", "大语言模型"]):
            resources.extend([
                "📖 docs/knowledge/llm/evolution/04-large-language-models.md",
                "🧪 tests/unit/evolution/test_large_language_models.py",
                "📚 推荐论文: 'Language Models are Few-Shot Learners'"
            ])

        elif any(word in question_lower for word in ["测试", "pytest"]):
            resources.extend([
                "📖 tests/测试体系总结.md",
                "🧪 运行: pytest tests/ -v",
                "📚 推荐文档: pytest官方文档"
            ])

        elif any(word in question_lower for word in ["学习", "路径"]):
            resources.extend([
                "📖 studyplan/README.md",
                "🧪 studyplan/learning_path_finder.py",
                "📚 推荐资源: 在线课程、技术博客、开源项目"
            ])

        return resources

    def _display_answer(self, question: str, answer: Dict[str, Any]) -> None:
        """显示回答"""
        print(f"\n🎯 问题: {question}")
        print("-" * 50)
        print(f"📝 回答 (置信度: {answer['confidence']:.1f}):")
        print(answer["answer"])

        if answer["related_resources"]:
            print("\n🔗 相关资源:")
            for resource in answer["related_resources"]:
                print(f"  {resource}")

        print(f"\n📊 回答来源: {answer['source']}")

    def _show_help(self) -> None:
        """显示帮助信息"""
        print("\n" + "=" * 50)
        print("📋 帮助信息")
        print("=" * 50)
        print("🎯 支持的问题类型:")
        for category, description in self.question_categories.items():
            print(f"  • {category}: {description}")

        print("\n💡 使用技巧:")
        print("  • 描述问题时尽量具体")
        print("  • 包含关键信息如错误信息")
        print("  • 可以询问学习建议")
        print("  • 可以咨询职业发展")

        print("\n🔍 可用命令:")
        print("  • help/h: 显示帮助")
        print("  • history: 查看历史记录")
        print("  • stats: 显示统计信息")
        print("  • quit/q/exit: 退出系统")

        print("\n📚 知识库覆盖范围:")
        print("  • AI技术发展历程 (1943-2025)")
        print("  • 大语言模型核心技术")
        print("  • 工程化实践和测试")
        print("  • 学习路径规划")
        print("=" * 50)

    def _show_history(self) -> None:
        """显示历史记录"""
        if not self.qa_history:
            print("\n📝 暂无历史记录")
            return

        print(f"\n📝 历史记录 (最近{min(10, len(self.qa_history))}条):")
        print("-" * 50)

        for i, qa in enumerate(self.qa_history[-10:], 1):
            print(f"{i}. {qa['question'][:50]}...")
            print(f"   📊 类别: {qa['category']}")
            print(f"   🕒 时间: {qa['timestamp']}")
            print()

    def _show_statistics(self) -> None:
        """显示统计信息"""
        if not self.qa_history:
            print("\n📊 暂无统计数据")
            return

        # 统计各类型问题数量
        category_count = {}
        for qa in self.qa_history:
            category = qa['category']
            category_count[category] = category_count.get(category, 0) + 1

        print(f"\n📊 会话统计:")
        print("-" * 50)
        print(f"📝 总问题数: {len(self.qa_history)}")
        print(f"🕒 会话开始时间: {self.qa_history[0]['timestamp']}")

        print("\n📊 问题类型分布:")
        for category, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category}: {count} ({count/len(self.qa_history)*100:.1f}%)")

        # 统计回答来源
        source_count = {}
        for qa in self.qa_history:
            source = qa['answer']['source']
            source_count[source] = source_count.get(source, 0) + 1

        print("\n📊 回答来源分布:")
        for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {source}: {count} ({count/len(self.qa_history)*100:.1f}%)")

        print("-" * 50)


def main():
    """主函数"""
    qa_system = InteractiveQA()
    qa_system.start_session()


if __name__ == "__main__":
    main()
