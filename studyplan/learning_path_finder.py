#!/usr/bin/env python3
"""
AI学习路径查找器

根据用户的学习目标、背景和时间，推荐最适合的学习路径
并提供个性化的学习计划。

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-12
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class LearningPathFinder:
    """AI学习路径查找器"""

    def __init__(self):
        self.learning_paths = {
            "researcher": {
                "name": "研究型学习路径",
                "description": "专注于AI理论研究和算法创新",
                "duration": "8-12周",
                "difficulty": "高级",
                "focus_areas": [
                    "数学基础", "算法原理", "论文阅读", "实验设计", "创新思维"
                ],
                "recommended_resources": [
                    "数学基础教程", "机器学习理论", "深度学习原理",
                    "学术论文", "研究方法", "实验设计"
                ],
                "career_outcomes": [
                    "AI研究员", "算法工程师", "机器学习科学家", "博士生"
                ]
            },
            "engineer": {
                "name": "工程型学习路径",
                "description": "专注于AI系统工程化实现和部署",
                "duration": "6-10周",
                "difficulty": "中级-高级",
                "focus_areas": [
                    "系统架构", "MLOps", "性能优化", "部署运维", "工具开发"
                ],
                "recommended_resources": [
                    "系统设计", "容器技术", "云计算", "数据库",
                    "监控告警", "CI/CD流水线", "性能调优"
                ],
                "career_outcomes": [
                    "AI工程师", "MLOps工程师", "系统架构师", "技术经理"
                ]
            },
            "product": {
                "name": "产品型学习路径",
                "description": "专注于AI产品设计、开发和商业化",
                "duration": "4-8周",
                "difficulty": "中级",
                "focus_areas": [
                    "产品设计", "用户研究", "商业分析", "产品管理", "市场策略"
                ],
                "recommended_resources": [
                    "产品设计", "用户研究", "市场分析", "商业模型",
                    "项目管理", "用户增长", "数据分析"
                ],
                "career_outcomes": [
                    "AI产品经理", "产品设计师", "创业者", "技术顾问"
                ]
            },
            "beginner": {
                "name": "入门型学习路径",
                "description": "适合AI初学者的系统化学习路径",
                "duration": "8-12周",
                "difficulty": "初级",
                "focus_areas": [
                    "基础概念", "编程技能", "工具使用", "实践项目", "基础算法"
                ],
                "recommended_resources": [
                    "Python基础", "数学基础", "AI概念入门",
                    "工具教程", "基础项目", "学习社区"
                ],
                "career_outcomes": [
                    "AI助理工程师", "数据分析师", "技术支持", "初级工程师"
                ]
            }
        }

        self.skill_assessment = {
            "programming": {
                "questions": [
                    "你的Python编程水平如何？",
                    "是否熟悉数据结构（数组、链表、树等）？",
                    "是否有面向对象编程经验？"
                ],
                "levels": ["初级", "中级", "高级", "专家"]
            },
            "mathematics": {
                "questions": [
                    "你的线性代数基础如何？",
                    "是否熟悉概率论和统计学？",
                    "对微积分的理解程度如何？"
                ],
                "levels": ["基础", "良好", "熟练", "精通"]
            },
            "ai_knowledge": {
                "questions": [
                    "对机器学习的基本概念了解程度？",
                    "是否使用过机器学习框架（如TensorFlow、PyTorch）？",
                    "是否完成过AI相关的项目？"
                ],
                "levels": ["入门", "初级", "中级", "高级"]
            },
            "domain_experience": {
                "questions": [
                    "所在的专业领域是什么？",
                    "是否有相关工作经验？",
                    "希望应用AI的具体场景是什么？"
                ],
                "levels": ["无经验", "1-2年", "3-5年", "5年以上"]
            }
        }

    def find_best_path(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """根据用户档案找到最佳学习路径"""
        scores = {}

        # 计算每个路径的匹配分数
        for path_name, path_info in self.learning_paths.items():
            scores[path_name] = self._calculate_path_score(user_profile, path_info)

        # 找到最高分数的路径
        best_path_name = max(scores.keys(), key=lambda x: scores[x])
        best_path = self.learning_paths[best_path_name]

        return {
            "recommended_path": best_path_name,
            "path_info": best_path,
            "scores": scores,
            "recommendation_confidence": scores[best_path_name],
            "alternative_paths": self._get_alternative_paths(scores, best_path_name)
        }

    def _calculate_path_score(self, user_profile: Dict[str, Any], path_info: Dict[str, Any]) -> float:
        """计算路径匹配分数"""
        score = 0.0

        # 根据用户技能水平评分
        if "skill_level" in user_profile:
            skill_level = user_profile["skill_level"]
            path_difficulty = path_info["difficulty"]

            if skill_level == "初级" and path_difficulty in ["初级", "中级"]:
                score += 30
            elif skill_level == "中级" and path_difficulty in ["中级", "高级"]:
                score += 30
            elif skill_level == "高级" and path_difficulty == "高级":
                score += 30
            elif skill_level == "初级" and path_difficulty == "高级":
                score += 10  # 仍然可以尝试，但分数较低
            else:
                score += 20

        # 根据学习目标评分
        if "learning_goal" in user_profile:
            goal = user_profile["learning_goal"]
            if goal == "research" and path_info["name"] == "研究型学习路径":
                score += 40
            elif goal == "engineering" and path_info["name"] == "工程型学习路径":
                score += 40
            elif goal == "product" and path_info["name"] == "产品型学习路径":
                score += 40
            elif goal == "beginner" and path_info["name"] == "入门型学习路径":
                score += 40
            else:
                score += 20

        # 根据时间投入评分
        if "time_commitment" in user_profile:
            time_hours = user_profile["time_commitment"]  # 每周小时数
            duration_weeks = int(path_info["duration"].split("-")[0])

            # 计算总学习时间
            total_hours = time_hours * duration_weeks

            # 根据路径难度，推荐合理的学习时间
            if path_info["difficulty"] == "初级" and total_hours >= 80:
                score += 20
            elif path_info["difficulty"] == "中级" and total_hours >= 120:
                score += 20
            elif path_info["difficulty"] == "高级" and total_hours >= 160:
                score += 20
            else:
                score += 10

        # 根据背景经验评分
        if "background" in user_profile:
            background = user_profile["background"]
            if background == "计算机科学":
                score += 10
            elif background == "数学统计":
                score += 10
            elif background == "其他工程":
                score += 5
            else:
                score += 0

        return score

    def _get_alternative_paths(self, scores: Dict[str, float], best_path: str) -> List[str]:
        """获取备选路径"""
        sorted_paths = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        alternatives = [path for path in sorted_paths if path != best_path]
        return alternatives[:2]  # 返回前2个备选路径

    def generate_personalized_plan(self, user_profile: Dict[str, Any], path_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成个性化学习计划"""
        # 基础信息
        duration_weeks = int(path_info["duration"].split("-")[0])
        time_commitment = user_profile.get("time_commitment", 10)  # 每周小时数
        start_date = datetime.now()
        end_date = start_date + timedelta(weeks=duration_weeks)

        # 生成周计划
        weekly_plan = self._generate_weekly_plan(path_info, duration_weeks, time_commitment)

        # 生成里程碑
        milestones = self._generate_milestones(path_info, duration_weeks)

        # 生成学习资源推荐
        resources = self._recommend_resources(user_profile, path_info)

        return {
            "plan_name": f"{path_info['name']}学习计划",
            "duration_weeks": duration_weeks,
            "time_commitment_weekly": time_commitment,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "weekly_plan": weekly_plan,
            "milestones": milestones,
            "recommended_resources": resources,
            "assessment_methods": self._get_assessment_methods(path_info),
            "success_metrics": self._get_success_metrics(path_info)
        }

    def _generate_weekly_plan(self, path_info: Dict[str, Any], duration_weeks: int, time_commitment: int) -> List[Dict[str, Any]]:
        """生成周计划"""
        weekly_plan = []

        # 根据路径类型分配学习内容
        for week in range(1, duration_weeks + 1):
            week_plan.append({
                "week": week,
                "topic": self._get_week_topic(path_info, week, duration_weeks),
                "learning_objectives": self._get_week_objectives(path_info, week),
                "time_allocation": self._get_time_allocation(time_commitment),
                "practice_activities": self._get_practice_activities(path_info, week),
                "reading_materials": self._get_reading_materials(path_info, week)
            })

        return weekly_plan

    def _get_week_topic(self, path_info: Dict[str, Any], week: int, total_weeks: int) -> str:
        """获取周主题"""
        path_name = path_info["name"]

        topics = {
            "研究型学习路径": [
                "数学基础强化", "机器学习理论", "深度学习原理", "经典论文研读",
                "实验设计方法", "算法创新实践", "论文写作技巧", "研究方向确定"
            ],
            "工程型学习路径": [
                "AI系统架构", "数据处理工程", "模型部署实践", "CI/CD流水线",
                "性能优化技术", "监控告警系统", "容器化部署", "DevOps实践"
            ],
            "产品型学习路径": [
                "AI产品设计", "用户需求分析", "市场调研方法", "产品原型开发",
                "用户测试方法", "商业模型设计", "产品营销策略", "上线运营"
            ],
            "入门型学习路径": [
                "Python编程基础", "AI概念入门", "基础工具使用", "简单项目实践",
                "机器学习基础", "深度学习入门", "项目集成实践", "综合应用"
            ]
        }

        return topics.get(path_name, ["综合学习"])[min(week - 1, len(topics[path_name]) - 1)]

    def _get_week_objectives(self, path_info: Dict[str, Any], week: int) -> List[str]:
        """获取周学习目标"""
        return [
            f"掌握第{week}周的核心概念",
            "完成相关编程练习",
            "通过阶段性测试",
            "记录学习笔记和思考"
        ]

    def _get_time_allocation(self, time_commitment: int) -> Dict[str, int]:
        """获取时间分配"""
        return {
            "理论学习": int(time_commitment * 0.4),
            "实践编程": int(time_commitment * 0.4),
            "阅读研究": int(time_commitment * 0.2)
        }

    def _get_practice_activities(self, path_info: Dict[str, Any], week: int) -> List[str]:
        """获取实践活动"""
        activities = {
            "研究型学习路径": [
                "数学推导练习", "算法实现", "论文复现", "实验设计"
            ],
            "工程型学习路径": [
                "系统设计", "代码实现", "部署配置", "性能测试"
            ],
            "产品型学习路径": [
                "用户调研", "原型设计", "功能开发", "用户测试"
            ],
            "入门型学习路径": [
                "编程练习", "工具使用", "小项目实现", "测试验证"
            ]
        }

        return activities.get(path_info["name"], ["学习实践"])

    def _get_reading_materials(self, path_info: Dict[str, Any], week: int) -> List[str]:
        """获取阅读材料"""
        materials = {
            "研究型学习路径": [
                "数学教材章节", "经典论文", "研究论文", "方法论文档"
            ],
            "工程型学习路径": [
                "技术文档", "最佳实践", "架构案例", "部署指南"
            ],
            "产品型学习路径": [
                "产品文档", "用户指南", "市场报告", "行业分析"
            ],
            "入门型学习路径": [
                "基础教程", "入门指南", "示例代码", "学习笔记"
            ]
        }

        return materials.get(path_info["name"], ["学习材料"])

    def _generate_milestones(self, path_info: Dict[str, Any], duration_weeks: int) -> List[Dict[str, Any]]:
        """生成学习里程碑"""
        milestones = []

        # 计算里程碑间隔
        milestone_interval = max(2, duration_weeks // 4)

        for i in range(1, duration_weeks + 1, milestone_interval):
            milestones.append({
                "week": i,
                "milestone": f"第{i}周里程碑",
                "description": self._get_milestone_description(path_info, i),
                "success_criteria": self._get_success_criteria(path_info, i),
                "deliverables": self._get_deliverables(path_info, i)
            })

        return milestones

    def _get_milestone_description(self, path_info: Dict[str, Any], week: int) -> str:
        """获取里程碑描述"""
        descriptions = {
            "研究型学习路径": {
                2: "掌握数学基础和理论框架",
                4: "理解核心算法和原理",
                6: "完成论文复现和实验",
                8: "确定研究方向和创新点"
            },
            "工程型学习路径": {
                2: "掌握AI系统基础架构",
                4: "完成端到端系统实现",
                6: "掌握部署和运维技术",
                8: "具备工程实践能力"
            },
            "产品型学习路径": {
                2: "完成产品需求分析",
                4: "开发产品原型",
                6: "完成用户测试和迭代",
                8: "具备产品上线能力"
            },
            "入门型学习路径": {
                2: "掌握基础编程和概念",
                4: "完成简单项目实践",
                6: "理解AI技术应用",
                8: "具备基础AI开发能力"
            }
        }

        return descriptions.get(path_info["name"], {}).get(week, f"第{week}周学习目标完成")

    def _get_success_criteria(self, path_info: Dict[str, Any], week: int) -> List[str]:
        """获取成功标准"""
        return [
            "完成本周学习任务",
            "通过相关测试用例",
            "提交学习成果",
            "达到学习目标"
        ]

    def _get_deliverables(self, path_info: Dict[str, Any], week: int) -> List[str]:
        """获取交付物"""
        return [
            "学习笔记",
            "代码实现",
            "测试报告",
            "学习总结"
        ]

    def _recommend_resources(self, user_profile: Dict[str, Any], path_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """推荐学习资源"""
        return {
            "core_resources": path_info.get("recommended_resources", []),
            "online_courses": [
                "Coursera - AI专项课程",
                "edX - 机器学习课程",
                "Udacity - 深度学习课程",
                "网易云课堂 - AI课程"
            ],
            "books": [
                "深度学习",
                "机器学习",
                "Python机器学习",
                "AI实践指南"
            ],
            "tools": [
                "Jupyter Notebook",
                "PyTorch/TensorFlow",
                "Git/GitHub",
                "Docker/Kubernetes"
            ],
            "communities": [
                "GitHub开源项目",
                "Stack Overflow",
                "Reddit ML社区",
                "知乎AI专栏"
            ]
        }

    def _get_assessment_methods(self, path_info: Dict[str, Any]) -> List[str]:
        """获取评估方法"""
        return [
            "每周测试题",
            "编程作业",
            "项目实践",
            "同伴评议",
            "最终项目"
        ]

    def _get_success_metrics(self, path_info: Dict[str, Any]) -> List[str]:
        """获取成功指标"""
        return [
            "测试通过率",
            "代码质量评分",
            "项目完成度",
            "知识掌握程度",
            "实践能力评估"
        ]


def main():
    """主函数示例"""
    # 创建学习路径查找器
    finder = LearningPathFinder()

    # 示例用户档案
    user_profiles = [
        {
            "name": "研究型学生",
            "skill_level": "高级",
            "learning_goal": "research",
            "time_commitment": 20,
            "background": "计算机科学"
        },
        {
            "name": "工程型学生",
            "skill_level": "中级",
            "learning_goal": "engineering",
            "time_commitment": 15,
            "background": "软件工程"
        },
        {
            "name": "产品型学生",
            "skill_level": "中级",
            "learning_goal": "product",
            "time_commitment": 10,
            "background": "工商管理"
        },
        {
            "name": "初学者",
            "skill_level": "初级",
            "learning_goal": "beginner",
            "time_commitment": 8,
            "background": "文科"
        }
    ]

    # 为每个用户生成学习计划
    for user_profile in user_profiles:
        print(f"\n=== {user_profile['name']}的学习计划 ===")

        # 找到最佳学习路径
        result = finder.find_best_path(user_profile)

        print(f"推荐路径: {result['recommended_path']}")
        print(f"置信度: {result['recommendation_confidence']:.1f}")
        print(f"路径描述: {result['path_info']['description']}")
        print(f"预计时间: {result['path_info']['duration']}")

        # 生成个性化计划
        plan = finder.generate_personalized_plan(user_profile, result['path_info'])

        print(f"\n学习计划概览:")
        print(f"- 计划名称: {plan['plan_name']}")
        print(f"- 学习周期: {plan['duration_weeks']}周")
        print(f"- 每周投入: {plan['time_commitment_weekly']}小时")
        print(f"- 开始日期: {plan['start_date']}")
        print(f"- 结束日期: {plan['end_date']}")

        print(f"\n里程碑数量: {len(plan['milestones'])}")
        for milestone in plan['milestones'][:2]:  # 显示前2个里程碑
            print(f"- {milestone['week']}周: {milestone['milestone']}")

        print(f"\n推荐资源类别: {list(plan['recommended_resources'].keys())}")
        print("-" * 50)


if __name__ == "__main__":
    main()
