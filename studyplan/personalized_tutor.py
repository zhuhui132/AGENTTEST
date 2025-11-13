#!/usr/bin/env python3
"""
个性化学习导师模块

提供智能化的个人学习指导，包括：
- 学习风格适配
- 个性化内容推荐
- 智能学习计划调整
- 学习方法指导
- 学习障碍诊断
- 学习动机激励

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-13
"""

import os
import json
import sqlite3
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TutoringMode(Enum):
    """导师模式枚举"""
    GENTLE = "gentle"           # 温和引导
    STRICT = "strict"             # 严格要求
    MOTIVATIONAL = "motivational" # 激励型
    TECHNICAL = "technical"      # 技术专家
    FRIENDLY = "friendly"         # 朋友式


class LearningStrategy(Enum):
    """学习策略枚举"""
    VISUAL_LEARNING = "visual_learning"      # 视觉学习
    AUDITORY_LEARNING = "auditory_learning"   # 听觉学习
    KINESTHETIC_LEARNING = "kinesthetic_learning" # 动觉学习
    READING_WRITING = "reading_writing"     # 读写学习
    MIXED_MODALITY = "mixed_modality"          # 混合模态


class DifficultyLevel(Enum):
    """难度级别枚举"""
    BEGINNER = "beginner"       # 初级
    ELEMENTARY = "elementary"   # 基础
    INTERMEDIATE = "intermediate" # 中级
    ADVANCED = "advanced"       # 高级
    EXPERT = "expert"           # 专家


class FeedbackType(Enum):
    """反馈类型枚举"""
    ENCOURAGEMENT = "encouragement"     # 鼓励
    CORRECTION = "correction"           # 纠正
    GUIDANCE = "guidance"               # 指导
    SUGGESTION = "suggestion"           # 建议
    WARNING = "warning"                 # 警告
    PRAISE = "praise"                   # 赞扬


@dataclass
class LearningProfile:
    """学习者档案"""
    user_id: int
    name: str
    age: int
    education_level: str
    learning_goals: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    learning_style: str = ""
    preferred_strategy: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    attention_span: int = 30  # 分钟
    preferred_difficulty: str = DifficultyLevel.INTERMEDIATE.value
    motivation_factors: List[str] = field(default_factory=list)
    learning_schedule: Dict[str, Any] = field(default_factory=dict)
    communication_preference: str = "friendly"
    feedback_preference: str = "constructive"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TutoringSession:
    """导师会话"""
    session_id: str
    user_id: int
    start_time: str
    end_time: Optional[str] = None
    topic: str = ""
    objectives: List[str] = field(default_factory=list)
    strategy_used: str = ""
    interactions: List[Dict] = field(default_factory=list)
    feedback_given: List[str] = field(default_factory=list)
    progress_made: float = 0.0
    user_satisfaction: int = 0
    next_steps: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TutoringFeedback:
    """导师反馈"""
    feedback_id: str
    user_id: int
    session_id: str
    feedback_type: FeedbackType
    content: str
    severity: int  # 1-10
    is_positive: bool
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_read: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearningChallenge:
    """学习挑战"""
    challenge_id: str
    user_id: int
    title: str
    description: str
    difficulty: str
    estimated_time: int  # 分钟
    learning_objectives: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    score: Optional[float] = None


class PersonalizedTutor:
    """个性化学习导师"""

    def __init__(self, db_path: str = "learning_tutor.db"):
        self.db_path = db_path
        self.init_database()
        self.response_templates = self._load_response_templates()
        self.motivational_quotes = self._load_motivational_quotes()

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 学习者档案表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_profiles (
                user_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                education_level TEXT,
                learning_goals TEXT,
                interests TEXT,
                learning_style TEXT,
                preferred_strategy TEXT,
                strengths TEXT,
                weaknesses TEXT,
                attention_span INTEGER DEFAULT 30,
                preferred_difficulty TEXT DEFAULT 'intermediate',
                motivation_factors TEXT,
                learning_schedule TEXT,
                communication_preference TEXT DEFAULT 'friendly',
                feedback_preference TEXT DEFAULT 'constructive',
                last_updated TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 导师会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tutoring_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                start_time TEXT NOT NULL,
                end_time TEXT,
                topic TEXT,
                objectives TEXT,
                strategy_used TEXT,
                interactions TEXT,
                feedback_given TEXT,
                progress_made REAL DEFAULT 0.0,
                user_satisfaction INTEGER DEFAULT 0,
                next_steps TEXT,
                notes TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES learning_profiles (user_id)
            )
        ''')

        # 导师反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tutoring_feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id INTEGER,
                session_id TEXT,
                feedback_type TEXT,
                content TEXT,
                severity INTEGER,
                is_positive BOOLEAN,
                timestamp TEXT,
                metadata TEXT,
                is_read BOOLEAN DEFAULT FALSE,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES learning_profiles (user_id),
                FOREIGN KEY (session_id) REFERENCES tutoring_sessions (session_id)
            )
        ''')

        # 学习挑战表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_challenges (
                challenge_id TEXT PRIMARY KEY,
                user_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                difficulty TEXT,
                estimated_time INTEGER,
                learning_objectives TEXT,
                prerequisites TEXT,
                resources_needed TEXT,
                success_criteria TEXT,
                hints TEXT,
                completed_at TEXT,
                score REAL,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES learning_profiles (user_id)
            )
        ''')

        # 学习进度跟踪表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT NOT NULL,
                topic TEXT,
                skill_level REAL DEFAULT 0.0,
                confidence_level REAL DEFAULT 0.0,
                time_spent INTEGER DEFAULT 0,
                achievements TEXT,
                challenges_completed INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES learning_profiles (user_id)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")

    def _load_response_templates(self) -> Dict[str, Dict[str, str]]:
        """加载响应模板"""
        return {
            "encouragement": {
                "start": "太棒了！让我们开始今天的愉快学习吧！",
                "progress": "你的进步让我印象深刻，继续保持这种势头！",
                "difficulty": "遇到困难是正常的，这正是你成长的机会！",
                "completion": "祝贺你完成了这个任务，你做得非常出色！"
            },
            "correction": {
                "gentle": "我注意到这里有一个小小的改进空间...",
                "technical": "从技术角度来看，这个部分可以这样优化...",
                "motivational": "让我们把这个困难变成你成功的垫脚石！"
            },
            "guidance": {
                "step_by_step": "让我们一步步来解决这个问题：",
                "conceptual": "首先，让我们理解一下这个核心概念...",
                "practical": "理论很重要，但让我们先看看实际应用..."
            },
            "suggestion": {
                "resource": "基于你的学习风格，我推荐这个资源...",
                "strategy": "试试这个学习方法，可能更适合你...",
                "time_management": "为了提高效率，建议你这样安排时间..."
            },
            "warning": {
                "time": "注意你的学习时间，记得要适时休息。",
                "difficulty": "这个内容可能有些难度，我们先做一些准备...",
                "attention": "我注意到你可能有些疲劳，要不要休息一下？"
            },
            "praise": {
                "achievement": "你在这个方面取得了突破，值得庆祝！",
                "effort": "我看到了你的努力和坚持，这是成功的关键！",
                "insight": "你的这个想法很有创意，展现了深度的思考！"
            }
        }

    def _load_motivational_quotes(self) -> List[str]:
        """加载激励名言"""
        return [
            "学习是知识积累的阶梯，每一步都让你离目标更近。",
            "困难是成功的催化剂，挑战是成长的催化剂。",
            "今天的学习投资，明天的人生收获。",
            "知识就是力量，实践就是应用力量的方式。",
            "相信自己的潜力，你就是自己最好的导师。",
            "每一次尝试都是向成功迈出的一步。",
            "学习不是为了考试，而是为了更好的未来。",
            "保持好奇心，它是学习最美的动力。",
            "坚持不一定成功，但放弃一定失败。",
            "专注当下，成就未来。"
        ]

    # ========================================================================
    # 学习者档案管理
    # ========================================================================

    def create_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """创建学习者档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO learning_profiles
                (user_id, name, age, education_level, learning_goals, interests,
                 learning_style, preferred_strategy, strengths, weaknesses,
                 attention_span, preferred_difficulty, motivation_factors,
                 learning_schedule, communication_preference, feedback_preference,
                 last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                profile_data.get('name', ''),
                profile_data.get('age', 0),
                profile_data.get('education_level', ''),
                json.dumps(profile_data.get('learning_goals', [])),
                json.dumps(profile_data.get('interests', [])),
                profile_data.get('learning_style', ''),
                profile_data.get('preferred_strategy', ''),
                json.dumps(profile_data.get('strengths', [])),
                json.dumps(profile_data.get('weaknesses', [])),
                profile_data.get('attention_span', 30),
                profile_data.get('preferred_difficulty', 'intermediate'),
                json.dumps(profile_data.get('motivation_factors', [])),
                json.dumps(profile_data.get('learning_schedule', {})),
                profile_data.get('communication_preference', 'friendly'),
                profile_data.get('feedback_preference', 'constructive'),
                datetime.now().isoformat()
            ))

            conn.commit()
            logger.info(f"学习者档案创建成功: 用户ID {user_id}")
            return True

        except Exception as e:
            logger.error(f"创建学习者档案失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_profile(self, user_id: int) -> Optional[LearningProfile]:
        """获取学习者档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id, name, age, education_level, learning_goals, interests,
                   learning_style, preferred_strategy, strengths, weaknesses,
                   attention_span, preferred_difficulty, motivation_factors,
                   learning_schedule, communication_preference, feedback_preference,
                   last_updated, created_at
            FROM learning_profiles
            WHERE user_id = ?
        ''', (user_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return LearningProfile(
                user_id=row[0],
                name=row[1],
                age=row[2],
                education_level=row[3],
                learning_goals=json.loads(row[4]) if row[4] else [],
                interests=json.loads(row[5]) if row[5] else [],
                learning_style=row[6],
                preferred_strategy=row[7],
                strengths=json.loads(row[8]) if row[8] else [],
                weaknesses=json.loads(row[9]) if row[9] else [],
                attention_span=row[10],
                preferred_difficulty=row[11],
                motivation_factors=json.loads(row[12]) if row[12] else [],
                learning_schedule=json.loads(row[13]) if row[13] else {},
                communication_preference=row[14],
                feedback_preference=row[15],
                last_updated=row[16]
            )

        return None

    def update_profile(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """更新学习者档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 构建SET语句
            set_clauses = []
            params = []

            for key, value in updates.items():
                if key in ['learning_goals', 'interests', 'strengths', 'weaknesses',
                           'motivation_factors', 'learning_schedule']:
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ?")
                params.append(value)

            if set_clauses:
                set_clauses.append("last_updated = ?")
                params.append(datetime.now().isoformat())
                params.append(user_id)

                sql = f"UPDATE learning_profiles SET {', '.join(set_clauses)} WHERE user_id = ?"
                cursor.execute(sql, params)

                conn.commit()
                logger.info(f"学习者档案更新成功: 用户ID {user_id}")
                return True

        except Exception as e:
            logger.error(f"更新学习者档案失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

        return False

    # ========================================================================
    # 智能学习分析
    # ========================================================================

    def analyze_learning_patterns(self, user_id: int) -> Dict[str, Any]:
        """分析学习模式"""
        profile = self.get_profile(user_id)
        if not profile:
            return {"error": "学习者档案不存在"}

        # 获取学习会话数据
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT strategy_used, progress_made, user_satisfaction, duration_minutes
            FROM tutoring_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (user_id,))

        sessions = cursor.fetchall()
        conn.close()

        # 分析学习策略效果
        strategy_performance = {}
        for session in sessions:
            strategy, progress, satisfaction, duration = session
            if strategy:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "progress_scores": [],
                        "satisfaction_scores": [],
                        "durations": []
                    }
                strategy_performance[strategy]["progress_scores"].append(progress)
                strategy_performance[strategy]["satisfaction_scores"].append(satisfaction)
                strategy_performance[strategy]["durations"].append(duration)

        # 计算策略效果评估
        best_strategy = ""
        best_score = 0
        for strategy, data in strategy_performance.items():
            if data["progress_scores"]:
                avg_progress = sum(data["progress_scores"]) / len(data["progress_scores"])
                avg_satisfaction = sum(data["satisfaction_scores"]) / len(data["satisfaction_scores"])
                avg_duration = sum(data["durations"]) / len(data["durations"])

                # 综合评分
                score = (avg_progress * 0.4 + avg_satisfaction * 0.4 - avg_duration / 60 * 0.2)
                strategy_performance[strategy]["score"] = score

                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        # 学习时间段分析
        optimal_time = self._analyze_optimal_learning_time(user_id)

        # 注意力模式分析
        attention_pattern = self._analyze_attention_pattern(profile, sessions)

        # 学习动机分析
        motivation_analysis = self._analyze_motivation(profile)

        return {
            "strategy_performance": strategy_performance,
            "recommended_strategy": best_strategy,
            "optimal_learning_time": optimal_time,
            "attention_pattern": attention_pattern,
            "motivation_analysis": motivation_analysis,
            "recommendations": self._generate_pattern_recommendations(profile, sessions)
        }

    def _analyze_optimal_learning_time(self, user_id: int) -> Dict[str, Any]:
        """分析最佳学习时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DATE(start_time) as date,
                   strftime('%H', start_time) as hour,
                   progress_made,
                   user_satisfaction
            FROM tutoring_sessions
            WHERE user_id = ?
            ORDER BY date DESC, hour
        ''', (user_id,))

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {"message": "数据不足"}

        # 按小时统计表现
        hourly_performance = {}
        for date, hour, progress, satisfaction in results:
            if hour not in hourly_performance:
                hourly_performance[hour] = {
                    "progress_scores": [],
                    "satisfaction_scores": []
                }
            hourly_performance[hour]["progress_scores"].append(progress)
            hourly_performance[hour]["satisfaction_scores"].append(satisfaction)

        # 计算各小时的综合表现
        hour_scores = {}
        for hour, data in hourly_performance.items():
            if data["progress_scores"]:
                avg_progress = sum(data["progress_scores"]) / len(data["progress_scores"])
                avg_satisfaction = sum(data["satisfaction_scores"]) / len(data["satisfaction_scores"])
                hour_scores[hour] = avg_progress * 0.5 + avg_satisfaction * 0.5

        # 找出最佳时间段
        if hour_scores:
            best_hour = max(hour_scores.items(), key=lambda x: x[1])[0]
            best_score = hour_scores[best_hour]
        else:
            best_hour = "10"
            best_score = 0

        return {
            "best_hour": best_hour,
            "best_score": best_score,
            "hourly_scores": hour_scores,
            "recommendation": f"建议在{best_hour}:00左右进行学习，这是你表现最好的时间段"
        }

    def _analyze_attention_pattern(self, profile: LearningProfile, sessions: List) -> Dict[str, Any]:
        """分析注意力模式"""
        attention_span = profile.attention_span
        session_durations = [s[3] for s in sessions if s[3]]  # duration_minutes

        if not session_durations:
            return {"message": "数据不足"}

        avg_duration = sum(session_durations) / len(session_durations)
        max_duration = max(session_durations)
        min_duration = min(session_durations)

        # 注意力稳定性分析
        if len(session_durations) > 1:
            duration_variance = sum((d - avg_duration) ** 2 for d in session_durations) / len(session_durations)
            stability_score = 1 - (duration_variance / (max_duration ** 2)) if max_duration > 0 else 0
        else:
            stability_score = 1.0

        pattern_type = ""
        recommendations = []

        if avg_duration > attention_span * 1.5:
            pattern_type = "over_extended"
            recommendations.append("建议适当缩短学习时间，增加休息频率")
        elif avg_duration < attention_span * 0.5:
            pattern_type = "under_utilized"
            recommendations.append("可以尝试延长学习时间，充分利用注意力资源")
        else:
            pattern_type = "well_matched"
            recommendations.append("学习时长与注意力匹配良好")

        if stability_score < 0.5:
            pattern_type += "_unstable"
            recommendations.append("学习时间不够稳定，建议制定固定的学习计划")

        return {
            "pattern_type": pattern_type,
            "attention_span": attention_span,
            "average_session_duration": avg_duration,
            "duration_variance": duration_variance if len(session_durations) > 1 else 0,
            "stability_score": stability_score,
            "recommendations": recommendations
        }

    def _analyze_motivation(self, profile: LearningProfile) -> Dict[str, Any]:
        """分析学习动机"""
        motivation_factors = profile.motivation_factors
        learning_goals = profile.learning_goals

        if not motivation_factors and not learning_goals:
            return {"message": "动机信息不足"}

        # 动机类型分析
        motivation_types = {
            "intrinsic": ["兴趣", "爱好", "好奇心", "自我提升", "热爱学习"],
            "extrinsic": ["工作", "考试", "证书", "金钱", "认可", "压力"],
            "social": ["家人", "朋友", "团队", "社交", "竞争"],
            "altruistic": ["帮助他人", "贡献社会", "教书育人", "志愿服务"]
        }

        motivation_scores = {}
        for factor in motivation_factors:
            factor_lower = factor.lower()
            for motive_type, keywords in motivation_types.items():
                if any(keyword in factor_lower for keyword in keywords):
                    motivation_scores[motive_type] = motivation_scores.get(motive_type, 0) + 1

        # 目标清晰度分析
        goal_clarity = len(learning_goals) > 0
        goal_specificity = any(len(goal.split()) > 5 for goal in learning_goals)

        # 动机强度评估
        motivation_strength = sum(motivation_scores.values())

        # 生成分析结果
        dominant_motivation = max(motivation_scores.items(), key=lambda x: x[1])[0] if motivation_scores else "unknown"

        return {
            "motivation_scores": motivation_scores,
            "dominant_motivation": dominant_motivation,
            "motivation_strength": motivation_strength,
            "goal_clarity": goal_clarity,
            "goal_specificity": goal_specificity,
            "recommendations": self._generate_motivation_recommendations(motivation_scores, learning_goals)
        }

    def _generate_pattern_recommendations(self, profile: LearningProfile, sessions: List) -> List[str]:
        """基于模式分析生成推荐"""
        recommendations = []

        # 基于学习风格推荐
        if profile.learning_style == "visual":
            recommendations.append("多使用图表、思维导图等视觉化工具")
        elif profile.learning_style == "auditory":
            recommendations.append("尝试听音频教程、参与讨论组")
        elif profile.learning_style == "kinesthetic":
            recommendations.append("通过实践项目和动手操作来学习")

        # 基于注意力跨度推荐
        if profile.attention_span < 20:
            recommendations.append("采用番茄工作法，25分钟学习+5分钟休息")
        elif profile.attention_span > 60:
            recommendations.append("可以安排较长的深度学习时段")

        # 基于优劣势推荐
        if profile.weaknesses:
            for weakness in profile.weaknesses[:2]:  # 专注前2个弱点
                if "数学" in weakness:
                    recommendations.append("加强数学基础练习，每天固定时间训练")
                elif "记忆" in weakness:
                    recommendations.append("使用间隔重复记忆法提高记忆效果")
                elif "理解" in weakness:
                    recommendations.append("尝试类比和实例来帮助理解抽象概念")

        return recommendations

    def _generate_motivation_recommendations(self, motivation_scores: Dict, learning_goals: List) -> List[str]:
        """生成动机相关推荐"""
        recommendations = []

        if not learning_goals:
            recommendations.append("建议设定明确的学习目标，这有助于提高学习动力")

        dominant_motivation = max(motivation_scores.items(), key=lambda x: x[1])[0] if motivation_scores else "unknown"

        if dominant_motivation == "intrinsic":
            recommendations.append("保持对学习的热爱，探索更多感兴趣的话题")
        elif dominant_motivation == "extrinsic":
            recommendations.append("适当培养内在兴趣，让学习过程更有意义")
        elif dominant_motivation == "social":
            recommendations.append("多参与学习社群，与他人分享学习心得")
        elif dominant_motivation == "altruistic":
            recommendations.append("考虑将所学知识用于帮助他人，增强学习价值感")

        if motivation_scores.get("unknown", 0) > 0:
            recommendations.append("深入思考学习的原因，找到真正驱动你的因素")

        return recommendations

    # ========================================================================
    # 智能导师会话
    # ========================================================================

    def start_tutoring_session(self, user_id: int, topic: str, objectives: List[str]) -> str:
        """开始导师会话"""
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建会话记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        profile = self.get_profile(user_id)
        if not profile:
            return "请先创建学习者档案"

        # 根据用户档案选择导师模式
        tutoring_mode = self._select_tutoring_mode(profile, topic)

        cursor.execute('''
            INSERT INTO tutoring_sessions
            (session_id, user_id, start_time, topic, objectives, strategy_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, user_id, datetime.now().isoformat(),
            topic, json.dumps(objectives), tutoring_mode.value
        ))

        conn.commit()
        conn.close()

        # 生成开场白
        greeting = self._generate_greeting(profile, topic, tutoring_mode)

        # 记录互动
        self._log_interaction(session_id, "tutor", greeting, "greeting")

        return f"会话开始 - {session_id}\n\n{greeting}"

    def process_user_message(self, session_id: str, user_message: str) -> str:
        """处理用户消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取会话信息
        cursor.execute('''
            SELECT user_id, topic, objectives, strategy_used
            FROM tutoring_sessions
            WHERE session_id = ?
        ''', (session_id,))

        session_info = cursor.fetchone()
        if not session_info:
            return "会话不存在"

        user_id, topic, objectives_json, strategy_used = session_info
        objectives = json.loads(objectives_json) if objectives_json else []
        profile = self.get_profile(user_id)

        if not profile:
            return "学习者档案不存在"

        # 记录用户消息
        self._log_interaction(session_id, "user", user_message, "message")

        # 分析用户消息并生成回复
        response_type = self._analyze_message_type(user_message, topic, objectives)
        response = self._generate_response(user_message, response_type, profile, topic, strategy_used)

        # 记录导师回复
        self._log_interaction(session_id, "tutor", response, "response")

        # 提供相关反馈
        feedback = self._generate_feedback(user_message, response, profile)
        if feedback:
            self._add_feedback(session_id, user_id, feedback)

        conn.close()
        return response

    def _select_tutoring_mode(self, profile: LearningProfile, topic: str) -> TutoringMode:
        """选择导师模式"""
        # 根据用户偏好和主题特点选择模式
        if profile.communication_preference == "strict":
            return TutoringMode.STRICT
        elif profile.communication_preference == "friendly":
            return TutoringMode.FRIENDLY
        elif "困难" in topic.lower() or "挑战" in topic.lower():
            return TutoringMode.MOTIVATIONAL
        elif "技术" in topic.lower() or "编程" in topic.lower():
            return TutoringMode.TECHNICAL
        else:
            return TutoringMode.GENTLE

    def _generate_greeting(self, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """生成开场白"""
        greetings = {
            TutoringMode.GENTLE: f"你好{profile.name}！今天我们要一起学习'{topic}'，我会温柔地引导你，让我们一起愉快地进步吧！",
            TutoringMode.STRICT: f"{profile.name}，今天我们要学习'{topic}'。我会严格要求，确保你真正掌握每个知识点。",
            TutoringMode.MOTIVATIONAL: f"{profile.name}！太棒了，选择学习'{topic}'说明你很有上进心！我会一直鼓励你，相信你一定能成功！",
            TutoringMode.TECHNICAL: f"{profile.name}，'{topic}'是一个很有深度的技术领域。我会用专业的方法指导你，确保你理解每个技术细节。",
            TutoringMode.FRIENDLY: f"嗨{profile.name}！很高兴和你一起学习'{topic}'。我们可以像朋友一样交流，有任何问题都可以问我哦！"
        }

        base_greeting = greetings.get(mode, greetings[TutoringMode.GENTLE])

        # 添加个性化元素
        if profile.interests:
            relevant_interests = [interest for interest in profile.interests
                                if interest.lower() in topic.lower()]
            if relevant_interests:
                base_greeting += f"\n我注意到你对{relevant_interests[0]}很感兴趣，这和今天的话题很匹配！"

        return base_greeting

    def _analyze_message_type(self, message: str, topic: str, objectives: List) -> str:
        """分析消息类型"""
        message_lower = message.lower()

        # 问题类型分析
        if any(qword in message_lower for qword in ["什么", "如何", "为什么", "怎么", "能否", "可以吗"]):
            return "question"
        elif any(sword in message_lower for sword in ["我懂了", "明白了", "清楚了", "知道了"]):
            return "understanding"
        elif any(fword in message_lower for fword in ["不会", "不懂", "困难", "问题", "错了", "卡住"]):
            return "difficulty"
        elif any(cword in message_lower for cword in ["试试", "练习", "做题", "实践", "操作"]):
            return "practice_request"
        elif any(rword in message_lower for rword in ["继续", "下一步", "然后", "接下来"]):
            return "continue_request"
        elif any(hword in message_lower for hword in ["累了", "休息", "困了", "烦"]):
            return "fatigue"
        else:
            return "general"

    def _generate_response(self, user_message: str, message_type: str, profile: LearningProfile,
                           topic: str, mode: TutoringMode) -> str:
        """生成回复"""
        response_generators = {
            "question": self._handle_question,
            "understanding": self._handle_understanding,
            "difficulty": self._handle_difficulty,
            "practice_request": self._handle_practice_request,
            "continue_request": self._handle_continue_request,
            "fatigue": self._handle_fatigue,
            "general": self._handle_general
        }

        generator = response_generators.get(message_type, self._handle_general)
        return generator(user_message, profile, topic, mode)

    def _handle_question(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理问题类消息"""
        # 根据学习风格提供不同的回答方式
        if profile.learning_style == "visual":
            return f"让我用一个图表来解释这个问题...\n关于'{message}'，我们可以这样理解：\n1. 首先画出核心概念\n2. 然后建立联系\n3. 最后形成完整的知识结构"
        elif profile.learning_style == "auditory":
            return f"让我们通过对话来探讨这个问题...\n关于'{message}'，我想听听你的理解，然后我们一起分析：\n1. 你对这个概念的第一印象是什么？"
        else:
            return f"这是个很好的问题！关于'{message}'，我来帮你详细解释：\n首先，让我们理解一下核心概念...\n这涉及哪些关键要点？\n最后，我们看看如何在实际中应用。"

    def _handle_understanding(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理理解类消息"""
        praises = [
            "太棒了！你的理解很准确。",
            "非常好！看来你已经掌握了这个要点。",
            "优秀！你的学习能力很强。",
            "很好！继续保持这种理解力。"
        ]

        praise = random.choice(praises)

        if mode == TutoringMode.MOTIVATIONAL:
            return f"{praise} 你的进步让我印象深刻！让我们继续挑战下一个内容吧。"
        elif mode == TutoringMode.TECHNICAL:
            return f"{praise} 现在让我们深入探讨一下这个概念的技术细节..."
        else:
            return f"{praise} 为了巩固你的理解，我们来做一个小练习怎么样？"

    def _handle_difficulty(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理困难类消息"""
        if mode == TutoringMode.MOTIVATIONAL:
            return f"遇到困难是完全正常的，这说明你正在挑战自己的边界！\n让我们换个角度来看这个问题，也许会更容易理解。"
        elif mode == TutoringMode.TECHNICAL:
            return f"这个技术点确实有难度，让我们从最基础的部分重新梳理一遍。"
        else:
            return f"别担心，每个学习过程中都会遇到困难。我们一步一步来解决这个问题。"

    def _handle_practice_request(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理练习请求"""
        return f"很好！练习是巩固知识的最好方式。\n让我为你设计一个合适的练习：\n1. 基础练习题\n2. 应用练习题\n3. 挑战练习题\n\n你想从哪个难度开始？"

    def _handle_continue_request(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理继续请求"""
        return f"好的，让我们继续前进！\n下一个内容是...\n这与我们刚才学的内容有什么联系？"

    def _handle_fatigue(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理疲劳类消息"""
        return f"看起来你有些累了，这很正常。学习需要张弛有度。\n让我们休息5分钟，然后可以换个轻松的话题，或者做一些简单的练习来保持状态。"

    def _handle_general(self, message: str, profile: LearningProfile, topic: str, mode: TutoringMode) -> str:
        """处理一般消息"""
        return f"我理解你的意思。关于'{topic}'，让我从另一个角度来解释一下。"

    def _log_interaction(self, session_id: str, speaker: str, message: str, interaction_type: str):
        """记录会话互动"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当前互动列表
        cursor.execute('''
            SELECT interactions FROM tutoring_sessions WHERE session_id = ?
        ''', (session_id,))

        result = cursor.fetchone()
        if result and result[0]:
            interactions = json.loads(result[0])
        else:
            interactions = []

        # 添加新互动
        interactions.append({
            "speaker": speaker,
            "message": message,
            "type": interaction_type,
            "timestamp": datetime.now().isoformat()
        })

        # 更新互动记录
        cursor.execute('''
            UPDATE tutoring_sessions
            SET interactions = ?
            WHERE session_id = ?
        ''', (json.dumps(interactions), session_id))

        conn.commit()
        conn.close()

    def _generate_feedback(self, user_message: str, tutor_response: str, profile: LearningProfile) -> Optional[TutoringFeedback]:
        """生成反馈"""
        feedback = None

        # 根据用户表现生成反馈
        if "困难" in user_message or "不会" in user_message:
            feedback = TutoringFeedback(
                feedback_id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=profile.user_id,
                session_id="",
                feedback_type=FeedbackType.ENCOURAGEMENT,
                content="遇到困难时保持积极态度是很好的学习品质。",
                severity=3,
                is_positive=True,
                timestamp=datetime.now().isoformat()
            )

        elif "我懂了" in user_message or "明白了" in user_message:
            feedback = TutoringFeedback(
                feedback_id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=profile.user_id,
                session_id="",
                feedback_type=FeedbackType.PRAISE,
                content="理解很快，继续保持这种学习状态！",
                severity=2,
                is_positive=True,
                timestamp=datetime.now().isoformat()
            )

        return feedback

    def _add_feedback(self, session_id: str, user_id: int, feedback: TutoringFeedback):
        """添加反馈到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO tutoring_feedback
            (feedback_id, user_id, session_id, feedback_type, content,
             severity, is_positive, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id, user_id, session_id,
            feedback.feedback_type.value, feedback.content,
            feedback.severity, feedback.is_positive,
            feedback.timestamp, feedback.created_at
        ))

        conn.commit()
        conn.close()

    # ========================================================================
    # 学习挑战系统
    # ========================================================================

    def create_challenge(self, user_id: int, challenge_data: Dict[str, Any]) -> str:
        """创建学习挑战"""
        challenge_id = f"challenge_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        challenge = LearningChallenge(
            challenge_id=challenge_id,
            user_id=user_id,
            title=challenge_data.get('title', '学习挑战'),
            description=challenge_data.get('description', ''),
            difficulty=challenge_data.get('difficulty', 'intermediate'),
            estimated_time=challenge_data.get('estimated_time', 30),
            learning_objectives=challenge_data.get('learning_objectives', []),
            prerequisites=challenge_data.get('prerequisites', []),
            resources_needed=challenge_data.get('resources_needed', []),
            success_criteria=challenge_data.get('success_criteria', []),
            hints=challenge_data.get('hints', [])
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO learning_challenges
                (challenge_id, user_id, title, description, difficulty,
                 estimated_time, learning_objectives, prerequisites,
                 resources_needed, success_criteria, hints, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                challenge.challenge_id, challenge.user_id, challenge.title,
                challenge.description, challenge.difficulty,
                challenge.estimated_time, json.dumps(challenge.learning_objectives),
                json.dumps(challenge.prerequisites), json.dumps(challenge.resources_needed),
                json.dumps(challenge.success_criteria), json.dumps(challenge.hints),
                challenge.created_at
            ))

            conn.commit()
            logger.info(f"学习挑战创建成功: {challenge_id}")
            return challenge_id

        except Exception as e:
            logger.error(f"创建学习挑战失败: {e}")
            conn.rollback()
            return ""
        finally:
            conn.close()

    def get_challenge(self, challenge_id: str) -> Optional[LearningChallenge]:
        """获取挑战详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT challenge_id, user_id, title, description, difficulty,
                   estimated_time, learning_objectives, prerequisites,
                   resources_needed, success_criteria, hints, completed_at, score
            FROM learning_challenges
            WHERE challenge_id = ?
        ''', (challenge_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return LearningChallenge(
                challenge_id=row[0], user_id=row[1], title=row[2],
                description=row[3], difficulty=row[4], estimated_time=row[5],
                learning_objectives=json.loads(row[6]) if row[6] else [],
                prerequisites=json.loads(row[7]) if row[7] else [],
                resources_needed=json.loads(row[8]) if row[8] else [],
                success_criteria=json.loads(row[9]) if row[9] else [],
                hints=json.loads(row[10]) if row[10] else [],
                completed_at=row[11], score=row[12]
            )

        return None

    def complete_challenge(self, challenge_id: str, score: float, feedback: str = "") -> bool:
        """完成挑战"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE learning_challenges
                SET completed_at = ?, score = ?
                WHERE challenge_id = ?
            ''', (datetime.now().isoformat(), score, challenge_id))

            conn.commit()
            logger.info(f"挑战完成: {challenge_id}, 得分: {score}")
            return True

        except Exception as e:
            logger.error(f"完成挑战失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # ========================================================================
    # 智能推荐系统
    # ========================================================================

    def get_personalized_recommendations(self, user_id: int) -> Dict[str, Any]:
        """获取个性化推荐"""
        profile = self.get_profile(user_id)
        if not profile:
            return {"error": "学习者档案不存在"}

        # 学习资源推荐
        resource_recommendations = self._recommend_resources(profile)

        # 学习策略推荐
        strategy_recommendations = self._recommend_strategies(profile)

        # 学习时间推荐
        time_recommendations = self._recommend_study_time(profile)

        # 学习目标推荐
        goal_recommendations = self._recommend_goals(profile)

        return {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "resource_recommendations": resource_recommendations,
            "strategy_recommendations": strategy_recommendations,
            "time_recommendations": time_recommendations,
            "goal_recommendations": goal_recommendations
        }

    def _recommend_resources(self, profile: LearningProfile) -> List[Dict[str, Any]]:
        """推荐学习资源"""
        recommendations = []

        # 根据兴趣推荐
        for interest in profile.interests[:3]:  # 最多推荐3个兴趣相关的资源
            recommendations.append({
                "type": "interest_based",
                "title": f"{interest}学习资源",
                "description": f"基于你对{interest}的兴趣推荐",
                "difficulty": profile.preferred_difficulty,
                "format": self._get_preferred_format(profile)
            })

        # 根据学习风格推荐
        style_resources = {
            "visual": ["视频教程", "图表资料", "思维导图工具"],
            "auditory": ["播客", "音频教程", "讨论组"],
            "kinesthetic": ["实践项目", "实验工具", "动手练习"],
            "reading": ["电子书", "文档资料", "技术博客"]
        }

        if profile.learning_style in style_resources:
            for resource_type in style_resources[profile.learning_style][:2]:
                recommendations.append({
                    "type": "style_based",
                    "title": resource_type,
                    "description": f"适合你的{profile.learning_style}学习风格",
                    "difficulty": profile.preferred_difficulty
                })

        return recommendations

    def _recommend_strategies(self, profile: LearningProfile) -> List[str]:
        """推荐学习策略"""
        strategies = []

        # 根据弱点推荐策略
        if "注意力" in str(profile.weaknesses):
            strategies.append("使用番茄工作法：25分钟专注+5分钟休息")

        if "记忆" in str(profile.weaknesses):
            strategies.append("采用间隔重复记忆法：Ebbinghaus遗忘曲线原理")

        if "理解" in str(profile.weaknesses):
            strategies.append("使用类比法：将抽象概念与熟悉事物联系起来")

        # 根据学习风格推荐策略
        style_strategies = {
            "visual": "制作思维导图和概念图，用图像帮助理解",
            "auditory": "大声朗读材料，参加讨论组和语音交流",
            "kinesthetic": "通过实践项目和动手操作来巩固知识",
            "reading": "做详细的笔记，写总结和概念解释"
        }

        if profile.learning_style in style_strategies:
            strategies.append(style_strategies[profile.learning_style])

        return strategies

    def _recommend_study_time(self, profile: LearningProfile) -> List[str]:
        """推荐学习时间"""
        recommendations = []

        # 根据注意力跨度推荐
        if profile.attention_span < 30:
            recommendations.append(f"建议单次学习时间不超过{profile.attention_span}分钟")
            recommendations.append("增加学习频率，减少单次时长")
        elif profile.attention_span > 60:
            recommendations.append("可以安排较长的深度学习时间")
            recommendations.append("注意设置休息时间，保持学习效率")

        # 根据日程推荐
        if profile.learning_schedule:
            recommendations.append("按照你偏好的时间安排学习")

        # 通用建议
        recommendations.append("选择精力最充沛的时间段学习")
        recommendations.append("建立固定的学习时间和地点")

        return recommendations

    def _recommend_goals(self, profile: LearningProfile) -> List[str]:
        """推荐学习目标"""
        recommendations = []

        if not profile.learning_goals:
            recommendations.append("设定具体的、可衡量的学习目标")
            recommendations.append("将大目标分解为小目标，逐步实现")
            recommendations.append("设定时间限制，提高学习效率")
        else:
            recommendations.append("定期回顾和调整学习目标")
            recommendations.append("为目标设定明确的完成标准")

        return recommendations

    def _get_preferred_format(self, profile: LearningProfile) -> str:
        """获取偏好格式"""
        format_mapping = {
            "visual": "视频",
            "auditory": "音频",
            "kinesthetic": "实践",
            "reading": "文字"
        }
        return format_mapping.get(profile.learning_style, "mixed")


def demo_personalized_tutor():
    """演示个性化导师功能"""
    print("=" * 70)
    print("🎓 个性化学习导师系统演示")
    print("=" * 70)

    # 创建导师实例
    tutor = PersonalizedTutor()

    # 演示用户ID
    demo_user_id = 1

    # 创建示例学习者档案
    profile_data = {
        "name": "小明",
        "age": 20,
        "education_level": "大学生",
        "learning_goals": ["掌握Python编程", "提高算法能力"],
        "interests": ["人工智能", "机器学习", "编程"],
        "learning_style": "visual",
        "preferred_strategy": "practice_first",
        "strengths": ["逻辑思维", "学习热情"],
        "weaknesses": ["注意力不够集中", "容易分心"],
        "attention_span": 25,
        "preferred_difficulty": "intermediate",
        "motivation_factors": ["兴趣驱动", "职业发展"],
        "communication_preference": "friendly",
        "feedback_preference": "constructive"
    }

    print(f"\n👤 1. 创建学习者档案")
    success = tutor.create_profile(demo_user_id, profile_data)
    print(f"   {'✅ 档案创建成功' if success else '❌ 档案创建失败'}")

    # 开始导师会话
    print(f"\n🎓 2. 开始导师会话")
    session_response = tutor.start_tutoring_session(
        demo_user_id,
        "Python编程基础",
        ["理解变量和数据类型", "掌握基本语法"]
    )
    print(f"   {session_response}")

    # 模拟用户对话
    print(f"\n💬 3. 模拟对话交互")

    sample_messages = [
        "什么是Python变量？",
        "我理解了基本概念",
        "这个有点难，我不太懂",
        "我想练习一下",
        "我累了"
    ]

    session_id = session_response.split('-')[1].strip()

    for message in sample_messages:
        print(f"\n👤 用户: {message}")
        response = tutor.process_user_message(session_id, message)
        print(f"🎓 导师: {response}")
        time.sleep(1)  # 模拟对话间隔

    # 学习模式分析
    print(f"\n📊 4. 学习模式分析")
    analysis = tutor.analyze_learning_patterns(demo_user_id)

    print(f"   推荐策略: {analysis.get('recommended_strategy', '未知')}")
    print(f"   最佳学习时间: {analysis.get('optimal_learning_time', {}).get('recommendation', '未知')}")
    print(f"   注意力模式: {analysis.get('attention_pattern', {}).get('pattern_type', '未知')}")

    recommendations = analysis.get('recommendations', [])
    print(f"   个性化推荐 ({len(recommendations)}条):")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"     {i}. {rec}")

    # 创建学习挑战
    print(f"\n🏆 5. 创建学习挑战")
    challenge_data = {
        "title": "Python变量和数据类型练习",
        "description": "通过实际编程练习巩固Python变量和数据类型的知识",
        "difficulty": "beginner",
        "estimated_time": 45,
        "learning_objectives": [
            "理解Python变量的概念",
            "掌握基本数据类型",
            "能够创建和使用变量"
        ],
        "success_criteria": [
            "正确创建不同类型的变量",
            "能够解释变量赋值的过程",
            "完成所有练习题"
        ],
        "hints": [
            "变量就像容器，可以存储数据",
            "Python是动态类型语言，会自动推断类型"
        ]
    }

    challenge_id = tutor.create_challenge(demo_user_id, challenge_data)
    print(f"   ✅ 挑战创建成功: {challenge_id}")

    # 获取个性化推荐
    print(f"\n🎯 6. 个性化推荐")
    recommendations = tutor.get_personalized_recommendations(demo_user_id)

    print(f"   资源推荐 ({len(recommendations['resource_recommendations'])}项):")
    for i, rec in enumerate(recommendations['resource_recommendations'][:2], 1):
        print(f"     {i}. {rec['title']} - {rec['description']}")

    print(f"   策略推荐 ({len(recommendations['strategy_recommendations'])}项):")
    for i, rec in enumerate(recommendations['strategy_recommendations'][:2], 1):
        print(f"     {i}. {rec}")

    print(f"   时间推荐 ({len(recommendations['time_recommendations'])}项):")
    for i, rec in enumerate(recommendations['time_recommendations'][:2], 1):
        print(f"     {i}. {rec}")

    print("\n" + "=" * 70)
    print("🎉 个性化导师系统演示完成！")
    print("=" * 70)
    print("\n💡 系统特色:")
    print("   • 智能学习档案分析")
    print("   • 个性化导师模式选择")
    print("   • 实时对话交互和反馈")
    print("   • 学习模式深度分析")
    print("   • 个性化学习挑战")
    print("   • 智能资源和方法推荐")
    print("   • 学习动机和注意力管理")


if __name__ == "__main__":
    demo_personalized_tutor()
