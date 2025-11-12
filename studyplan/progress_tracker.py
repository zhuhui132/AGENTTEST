#!/usr/bin/env python3
"""
学习进度跟踪器

帮助学生跟踪学习进度，生成学习报告，提供学习建议。

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-12
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd


class LearningProgressTracker:
    """学习进度跟踪器"""

    def __init__(self, db_path: str = None):
        """初始化跟踪器"""
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "learning_progress.db")

        self.db_path = db_path
        self.init_database()

        # 学习里程碑定义
        self.milestones = {
            "基础理论": {
                "ai_development_history": "完成AI发展历史学习",
                "neural_networks": "掌握神经网络基础",
                "machine_learning": "理解机器学习概念",
                "programming_skills": "具备基础编程技能"
            },
            "核心技术": {
                "transformer": "理解Transformer架构",
                "attention_mechanism": "掌握注意力机制",
                "large_language_models": "了解大语言模型",
                "few_shot_learning": "掌握少样本学习"
            },
            "工程实践": {
                "data_engineering": "完成数据工程实践",
                "mlops": "掌握MLOps工程",
                "testing": "通过相关测试用例",
                "deployment": "掌握部署技术"
            },
            "应用创新": {
                "project_completion": "完成综合项目",
                "problem_solving": "具备问题解决能力",
                "innovation_thinking": "培养创新思维",
                "collaboration": "具备团队协作能力"
            }
        }

    def init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                skill_level TEXT,
                learning_goal TEXT,
                start_date TEXT,
                target_end_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建学习记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                topic TEXT,
                hours_spent REAL,
                completion_percentage REAL,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # 创建测试记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                test_name TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                score REAL,
                execution_time TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # 创建里程碑完成记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS milestone_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                milestone_category TEXT,
                milestone_name TEXT,
                completed_date TEXT,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.commit()
        conn.close()

    def create_user(self, name: str, email: str = None, skill_level: str = "beginner",
                   learning_goal: str = "general", target_weeks: int = 8) -> int:
        """创建用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = datetime.now().isoformat()
        target_end_date = (datetime.now() + timedelta(weeks=target_weeks)).isoformat()

        cursor.execute("""
            INSERT INTO users (name, email, skill_level, learning_goal, start_date, target_end_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, email, skill_level, learning_goal, start_date, target_end_date))

        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return user_id

    def add_learning_record(self, user_id: int, date: str, topic: str,
                         hours_spent: float, completion_percentage: float,
                         notes: str = "") -> bool:
        """添加学习记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO learning_records (user_id, date, topic, hours_spent, completion_percentage, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, date, topic, hours_spent, completion_percentage, notes))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"添加学习记录失败: {str(e)}")
            return False

    def add_test_record(self, user_id: int, test_name: str, total_tests: int,
                       passed_tests: int, execution_time: str = "") -> bool:
        """添加测试记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            score = (passed_tests / total_tests * 100) if total_tests > 0 else 0

            cursor.execute("""
                INSERT INTO test_records (user_id, test_name, total_tests, passed_tests, score, execution_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, test_name, total_tests, passed_tests, score, execution_time))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"添加测试记录失败: {str(e)}")
            return False

    def add_milestone(self, user_id: int, category: str, name: str, notes: str = "") -> bool:
        """添加里程碑完成记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            completed_date = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO milestone_records (user_id, milestone_category, milestone_name, completed_date, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, category, name, completed_date, notes))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"添加里程碑记录失败: {str(e)}")
            return False

    def get_user_progress(self, user_id: int) -> Dict[str, Any]:
        """获取用户学习进度"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取用户基本信息
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()

        if not user:
            return {"error": "用户不存在"}

        user_info = {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "skill_level": user[3],
            "learning_goal": user[4],
            "start_date": user[5],
            "target_end_date": user[6],
            "created_at": user[7]
        }

        # 获取学习记录统计
        cursor.execute("""
            SELECT COUNT(*), SUM(hours_spent), AVG(completion_percentage),
                   MIN(date), MAX(date)
            FROM learning_records WHERE user_id = ?
        """, (user_id,))

        learning_stats = cursor.fetchone()
        learning_info = {
            "total_days": learning_stats[0],
            "total_hours": learning_stats[1] or 0,
            "avg_completion": learning_stats[2] or 0,
            "start_date": learning_stats[3],
            "last_date": learning_stats[4]
        }

        # 获取测试记录统计
        cursor.execute("""
            SELECT COUNT(*), AVG(score), MAX(score), SUM(passed_tests), SUM(total_tests)
            FROM test_records WHERE user_id = ?
        """, (user_id,))

        test_stats = cursor.fetchone()
        test_info = {
            "total_tests_taken": test_stats[0],
            "avg_score": test_stats[1] or 0,
            "best_score": test_stats[2] or 0,
            "total_passed": test_stats[3] or 0,
            "total_test_questions": test_stats[4] or 0,
            "overall_pass_rate": (test_stats[3] or 0) / (test_stats[4] or 1) * 100 if test_stats[4] else 0
        }

        # 获取里程碑完成情况
        cursor.execute("""
            SELECT milestone_category, milestone_name, completed_date, notes
            FROM milestone_records WHERE user_id = ?
            ORDER BY completed_date
        """, (user_id,))

        milestone_records = cursor.fetchall()
        milestones = []
        for record in milestone_records:
            milestones.append({
                "category": record[0],
                "name": record[1],
                "completed_date": record[2],
                "notes": record[3]
            })

        # 计算各阶段完成度
        stage_progress = self._calculate_stage_progress(milestones)

        conn.close()

        return {
            "user_info": user_info,
            "learning_info": learning_info,
            "test_info": test_info,
            "milestones": milestones,
            "stage_progress": stage_progress
        }

    def _calculate_stage_progress(self, milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算各阶段完成度"""
        stage_progress = {}

        for stage, stage_milestones in self.milestones.items():
            completed_count = 0
            total_count = len(stage_milestones)

            completed_milestones = [m["name"] for m in milestones if m["category"] == stage]

            for milestone_name in stage_milestones.values():
                if milestone_name in completed_milestones:
                    completed_count += 1

            stage_progress[stage] = {
                "completed": completed_count,
                "total": total_count,
                "completion_rate": (completed_count / total_count * 100) if total_count > 0 else 0,
                "completed_milestones": completed_milestones
            }

        return stage_progress

    def generate_progress_report(self, user_id: int) -> str:
        """生成学习进度报告"""
        progress = self.get_user_progress(user_id)

        if "error" in progress:
            return progress["error"]

        user_info = progress["user_info"]
        learning_info = progress["learning_info"]
        test_info = progress["test_info"]
        stage_progress = progress["stage_progress"]

        report = f"""
# 🎓 {user_info['name']} 的学习进度报告

## 📊 基本信息
- **姓名**: {user_info['name']}
- **技能水平**: {user_info['skill_level']}
- **学习目标**: {user_info['learning_goal']}
- **开始日期**: {user_info['start_date'][:10]}
- **目标日期**: {user_info['target_end_date'][:10]}

## 📚 学习统计
- **学习天数**: {learning_info['total_days']} 天
- **总学习时间**: {learning_info['total_hours']:.1f} 小时
- **平均完成度**: {learning_info['avg_completion']:.1f}%
- **学习跨度**: {learning_info['start_date'][:10]} 至 {learning_info['last_date'][:10]}

## 🧪 测试统计
- **参加测试次数**: {test_info['total_tests_taken']} 次
- **平均得分**: {test_info['avg_score']:.1f}%
- **最高得分**: {test_info['best_score']:.1f}%
- **总通过率**: {test_info['overall_pass_rate']:.1f}%
- **通过题目/总题目**: {test_info['total_passed']}/{test_info['total_test_questions']}

## 🎯 里程碑完成情况
"""

        for stage, progress in stage_progress.items():
            stage_emoji = {
                "基础理论": "📖",
                "核心技术": "🔬",
                "工程实践": "🛠️",
                "应用创新": "🚀"
            }.get(stage, "📌")

            report += f"""
### {stage_emoji} {stage}
- **完成度**: {progress['completion_rate']:.1f}% ({progress['completed']}/{progress['total']})
- **已完成里程碑**: {', '.join(progress['completed_milestones']) if progress['completed_milestones'] else '暂无'}
"""

        # 计算总体进度
        overall_progress = sum(p['completion_rate'] for p in stage_progress.values()) / len(stage_progress)

        report += f"""
## 📈 总体进度
- **总体完成度**: {overall_progress:.1f}%
- **学习状态**: {'🔥 学习状态优秀' if overall_progress >= 80 else '💪 继续加油' if overall_progress >= 50 else '🌱 刚刚开始'}
"""

        # 学习建议
        report += "\n## 💡 学习建议\n"

        if overall_progress < 30:
            report += "- 🌱 **初学者建议**: 专注于基础理论，建立扎实的基础\n"
            report += "- 📚 **学习节奏**: 保持每天1-2小时的学习时间\n"
            report += "- 🎯 **目标设定**: 设定小的、可实现的学习目标\n"
        elif overall_progress < 60:
            report += "- 📈 **进阶建议**: 在基础扎实后，开始接触核心技术\n"
            report += "- 🧪 **实践验证**: 通过测试验证理论知识\n"
            report += "- 🤝 **寻求帮助**: 遇到问题时及时寻求帮助\n"
        elif overall_progress < 85:
            report += "- 🚀 **提升建议**: 专注于工程实践和应用创新\n"
            report += "- 🛠️ **项目实践**: 完成综合项目提升实战能力\n"
            report += "- 🔍 **深入研究**: 在感兴趣的领域进行深入研究\n"
        else:
            report += "- 🌟 **卓越表现**: 已经完成了大部分学习目标\n"
            report += "- 🎓 **分享经验**: 与其他学习者分享你的经验\n"
            report += "- 📝 **持续学习**: 继续关注最新的技术发展\n"

        # 时间建议
        if learning_info['total_hours'] < 50:
            report += "\n## ⏰ 时间管理建议\n"
            report += "- 📅 建议增加每日学习时间到2-3小时\n"
            report += "- 🎯 设定固定的学习时间段，养成学习习惯\n"

        # 测试建议
        if test_info['avg_score'] < 70:
            report += "\n## 🧪 测试提升建议\n"
            report += "- 📖 加强理论知识的学习和复习\n"
            report += "- 💻 增加编程实践，巩固理论知识\n"
            report += "- 🤝 与其他学习者交流，讨论问题\n"

        report += f"""
---
## 📅 下一步学习计划
### 推荐学习重点
{self._get_next_learning_focus(stage_progress)}
### 推荐测试
{self._get_recommended_tests(stage_progress)}

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return report

    def _get_next_learning_focus(self, stage_progress: Dict[str, Any]) -> str:
        """获取下一步学习重点"""
        # 找到完成度最低的阶段
        min_stage = min(stage_progress.items(), key=lambda x: x[1]['completion_rate'])

        stage_name = min_stage[0]
        progress = min_stage[1]

        if stage_name == "基础理论":
            return "- 📖 深入学习AI发展历史和神经网络基础\n- 🧮 加强数学基础，特别是线性代数和概率论\n- 💻 提升编程技能，掌握Python和数据结构"
        elif stage_name == "核心技术":
            return "- ⚡ 专注学习Transformer架构和注意力机制\n- 🤖 深入理解大语言模型的工作原理\n- 🔧 实践少样本学习和思维链推理"
        elif stage_name == "工程实践":
            return "- 🏗️ 学习数据工程和MLOps工程实践\n- 🧪 掌握测试框架和自动化测试\n- 🚀 了解模型部署和运维技术"
        else:
            return "- 🎯 完成综合项目，整合所学知识\n- 💡 培养创新思维，思考技术改进\n- 🤝 参与开源项目，提升协作能力"

    def _get_recommended_tests(self, stage_progress: Dict[str, Any]) -> str:
        """获取推荐测试"""
        lowest_stage = min(stage_progress.items(), key=lambda x: x[1]['completion_rate'])[0]

        test_recommendations = {
            "基础理论": """
- tests/unit/evolution/test_neural_networks_foundation.py
- tests/unit/evolution/test_deep_learning_breakthrough.py
- tests/unit/test_machine_learning.py""",
            "核心技术": """
- tests/unit/evolution/test_transformer_revolution.py
- tests/unit/evolution/test_large_language_models.py
- tests/unit/evolution/test_llm_evolution.py""",
            "工程实践": """
- tests/specialized/test_data_engineering.py
- tests/specialized/test_mlops_engineering.py
- tests/unit/evolution/test_llm_evolution.py""",
            "应用创新": """
- tests/integration/test_agent_full_integration.py
- tests/e2e/test_complete_conversation_flow.py
- tests/performance/test_agent_performance.py"""
        }

        return test_recommendations.get(lowest_stage, "- 建议完成所有基础测试")

    def generate_progress_chart(self, user_id: int, save_path: str = None) -> str:
        """生成学习进度图表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取每日学习记录
        cursor.execute("""
            SELECT date, SUM(hours_spent) as daily_hours, AVG(completion_percentage) as avg_completion
            FROM learning_records WHERE user_id = ?
            GROUP BY date ORDER BY date
        """, (user_id,))

        daily_data = cursor.fetchall()
        conn.close()

        if not daily_data:
            return "没有学习数据，无法生成图表"

        # 准备数据
        dates = [row[0] for row in daily_data]
        hours = [row[1] for row in daily_data]
        completions = [row[2] for row in daily_data]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 每日学习时间图表
        ax1.plot(dates, hours, marker='o', linewidth=2, markersize=6)
        ax1.set_title('📚 每日学习时间趋势')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('学习时间 (小时)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 每日完成度图表
        ax2.plot(dates, completions, marker='s', linewidth=2, markersize=6, color='orange')
        ax2.set_title('🎯 每日学习完成度')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('完成度 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = f"progress_chart_user_{user_id}.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return f"图表已保存到: {save_path}"

    def export_progress_data(self, user_id: int, format: str = "json") -> str:
        """导出学习进度数据"""
        progress = self.get_user_progress(user_id)

        if format.lower() == "json":
            filename = f"progress_data_user_{user_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            return f"数据已导出到: {filename}"

        elif format.lower() == "csv":
            filename = f"progress_data_user_{user_id}.csv"

            # 将学习记录导出为CSV
            conn = sqlite3.connect(self.db_path)

            learning_df = pd.read_sql_query(
                "SELECT date, topic, hours_spent, completion_percentage, notes FROM learning_records WHERE user_id = ?",
                conn, params=(user_id,)
            )

            learning_df.to_csv(filename, index=False)
            conn.close()

            return f"学习记录已导出到: {filename}"

        else:
            return "不支持的格式，请使用 'json' 或 'csv'"

    def get_learning_recommendations(self, user_id: int) -> Dict[str, Any]:
        """获取个性化学习建议"""
        progress = self.get_user_progress(user_id)

        if "error" in progress:
            return {"error": "无法获取用户进度"}

        user_info = progress["user_info"]
        learning_info = progress["learning_info"]
        test_info = progress["test_info"]
        stage_progress = progress["stage_progress"]

        recommendations = {
            "user_info": user_info,
            "current_status": self._evaluate_current_status(progress),
            "learning_focus": self._get_learning_focus(stage_progress),
            "time_management": self._get_time_management_advice(learning_info),
            "test_preparation": self._get_test_preparation_advice(test_info),
            "next_steps": self._get_next_steps(stage_progress),
            "resources": self._get_recommended_resources(stage_progress)
        }

        return recommendations

    def _evaluate_current_status(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        """评估当前学习状态"""
        overall_progress = sum(p['completion_rate'] for p in progress["stage_progress"].values()) / len(progress["stage_progress"])

        status = "优秀" if overall_progress >= 80 else "良好" if overall_progress >= 60 else "需要加油"

        return {
            "overall_progress": overall_progress,
            "status": status,
            "strengths": self._identify_strengths(progress),
            "weaknesses": self._identify_weaknesses(progress)
        }

    def _identify_strengths(self, progress: Dict[str, Any]) -> List[str]:
        """识别学习优势"""
        strengths = []
        stage_progress = progress["stage_progress"]

        # 找出完成度高的阶段
        for stage, progress_data in stage_progress.items():
            if progress_data['completion_rate'] >= 80:
                strengths.append(f"{stage}领域掌握良好")

        # 学习时间充足
        if progress['learning_info']['total_hours'] >= 100:
            strengths.append("学习投入时间充足")

        # 测试表现良好
        if progress['test_info']['avg_score'] >= 80:
            strengths.append("测试表现优异")

        return strengths if strengths else ["继续努力，积累更多优势"]

    def _identify_weaknesses(self, progress: Dict[str, Any]) -> List[str]:
        """识别学习弱点"""
        weaknesses = []
        stage_progress = progress["stage_progress"]

        # 找出完成度低的阶段
        for stage, progress_data in stage_progress.items():
            if progress_data['completion_rate'] < 50:
                weaknesses.append(f"{stage}领域需要加强")

        # 学习时间不足
        if progress['learning_info']['total_hours'] < 50:
            weaknesses.append("学习时间投入不足")

        # 测试表现需要提升
        if progress['test_info']['avg_score'] < 70:
            weaknesses.append("测试表现需要提升")

        return weaknesses if weaknesses else ["无明显弱点，继续保持"]

    def _get_learning_focus(self, stage_progress: Dict[str, Any]) -> List[str]:
        """获取学习重点"""
        focus_areas = []

        # 找到完成度最低的阶段
        min_stage = min(stage_progress.items(), key=lambda x: x[1]['completion_rate'])
        min_stage_name = min_stage[0]
        min_stage_progress = min_stage[1]

        if min_stage_progress['completion_rate'] < 50:
            focus_areas.append(f"重点攻克{min_stage_name}领域")

        # 检查是否有未完成的里程碑
        for stage, progress_data in stage_progress.items():
            if progress_data['completion_rate'] > 0 and progress_data['completion_rate'] < 100:
                focus_areas.append(f"完成{stage}的剩余里程碑")

        return focus_areas if focus_areas else ["保持当前学习节奏"]

    def _get_time_management_advice(self, learning_info: Dict[str, Any]) -> List[str]:
        """获取时间管理建议"""
        advice = []
        total_hours = learning_info['total_hours']
        avg_completion = learning_info['avg_completion']

        if total_hours < 50:
            advice.append("建议每天投入2-3小时学习时间")
        elif total_hours < 100:
            advice.append("保持当前学习节奏，可适当增加学习时间")

        if avg_completion < 70:
            advice.append("提高学习效率，专注于理解而非仅仅完成任务")

        return advice if advice else ["时间管理良好"]

    def _get_test_preparation_advice(self, test_info: Dict[str, Any]) -> List[str]:
        """获取测试准备建议"""
        advice = []
        avg_score = test_info['avg_score']

        if avg_score < 70:
            advice.append("加强理论知识学习，多做练习题")
            advice.append("分析测试失败的原因，针对性改进")
        elif avg_score < 85:
            advice.append("保持良好的测试表现，尝试更有挑战性的测试")
        else:
            advice.append("测试表现优秀，可以考虑参加竞赛或项目")

        return advice

    def _get_next_steps(self, stage_progress: Dict[str, Any]) -> List[str]:
        """获取下一步学习步骤"""
        next_steps = []

        # 检查基础理论阶段
        if stage_progress.get("基础理论", {}).get("completion_rate", 0) < 100:
            next_steps.extend([
                "完成神经网络基础理论学习",
                "通过基础理论相关测试",
                "实现简单的神经网络模型"
            ])

        # 检查核心技术阶段
        if stage_progress.get("核心技术", {}).get("completion_rate", 0) < 100:
            next_steps.extend([
                "深入学习Transformer架构",
                "掌握注意力机制原理",
                "实践大语言模型应用"
            ])

        # 检查工程实践阶段
        if stage_progress.get("工程实践", {}).get("completion_rate", 0) < 100:
            next_steps.extend([
                "学习数据工程实践",
                "掌握MLOps工程技能",
                "通过工程相关测试"
            ])

        return next_steps[:5]  # 最多返回5个步骤

    def _get_recommended_resources(self, stage_progress: Dict[str, Any]) -> List[str]:
        """获取推荐学习资源"""
        resources = []

        # 根据当前阶段推荐资源
        for stage, progress_data in stage_progress.items():
            if progress_data['completion_rate'] > 0 and progress_data['completion_rate'] < 100:
                if stage == "基础理论":
                    resources.extend([
                        "docs/knowledge/ai-development-timeline.md",
                        "docs/knowledge/llm/evolution/01-neural-networks-foundation.md"
                    ])
                elif stage == "核心技术":
                    resources.extend([
                        "docs/knowledge/llm/evolution/03-transformer-revolution.md",
                        "tests/unit/evolution/test_transformer_revolution.py"
                    ])
                elif stage == "工程实践":
                    resources.extend([
                        "tests/specialized/test_data_engineering.py",
                        "tests/specialized/test_mlops_engineering.py"
                    ])

        return resources[:10]  # 最多返回10个资源


def main():
    """主函数示例"""
    tracker = LearningProgressTracker()

    # 创建示例用户
    user_id = tracker.create_user(
        name="张三",
        email="zhangsan@example.com",
        skill_level="beginner",
        learning_goal="research",
        target_weeks=8
    )

    print(f"创建用户成功，用户ID: {user_id}")

    # 添加学习记录示例
    today = datetime.now().strftime("%Y-%m-%d")
    tracker.add_learning_record(
        user_id=user_id,
        date=today,
        topic="神经网络基础",
        hours_spent=2.5,
        completion_percentage=75.0,
        notes="学习了McCulloch-Pitts神经元模型"
    )

    # 添加测试记录示例
    tracker.add_test_record(
        user_id=user_id,
        test_name="test_neural_networks_foundation.py",
        total_tests=15,
        passed_tests=12,
        execution_time=datetime.now().strftime("%H:%M:%S")
    )

    # 添加里程碑示例
    tracker.add_milestone(
        user_id=user_id,
        category="基础理论",
        name="掌握神经网络基础",
        notes="完成了神经网络基础章节的学习和测试"
    )

    # 生成进度报告
    report = tracker.generate_progress_report(user_id)
    print(report)

    # 生成进度图表
    chart_result = tracker.generate_progress_chart(user_id)
    print(chart_result)

    # 获取学习建议
    recommendations = tracker.get_learning_recommendations(user_id)
    print("个性化学习建议:")
    for key, value in recommendations.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
