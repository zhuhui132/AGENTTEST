#!/usr/bin/env python3
"""
AI学习系统高级功能模块

扩展学习计划系统的功能，包括：
- 智能学习推荐
- 学习资源管理
- 学习社区功能
- 学习成就系统
- 个性化学习分析
- 学习路径优化

作者: AI学习团队
版本: 2.0.0
日期: 2025-11-13
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import asyncio
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningStyle(Enum):
    """学习风格枚举"""
    VISUAL = "visual"           # 视觉型
    AUDITORY = "auditory"         # 听觉型
    KINESTHETIC = "kinesthetic"   # 动觉型
    READING = "reading"           # 阅读型
    MIXED = "mixed"                # 混合型


class DifficultyLevel(Enum):
    """难度等级枚举"""
    BEGINNER = "beginner"       # 初级
    INTERMEDIATE = "intermediate" # 中级
    ADVANCED = "advanced"       # 高级
    EXPERT = "expert"           # 专家级


class ResourceType(Enum):
    """资源类型枚举"""
    ARTICLE = "article"         # 文章
    VIDEO = "video"             # 视频
    BOOK = "book"               # 书籍
    COURSE = "course"           # 课程
    TUTORIAL = "tutorial"       # 教程
    PROJECT = "project"         # 项目
    PAPER = "paper"             # 论文
    CODE = "code"               # 代码
    DATASET = "dataset"         # 数据集
    TOOL = "tool"               # 工具


class AchievementType(Enum):
    """成就类型枚举"""
    LEARNING_STREAK = "learning_streak"     # 学习连续天数
    TOPIC_MASTER = "topic_master"         # 主题掌握
    SPEED_LEARNER = "speed_learner"       # 快速学习
    PRACTICE_EXPERT = "practice_expert"   # 练习专家
    COMMUNITY_HELPER = "community_helper"   # 社区助手
    KNOWledge_SHARER = "knowledge_sharer"   # 知识分享者


@dataclass
class LearningResource:
    """学习资源类"""
    id: str
    title: str
    description: str
    resource_type: ResourceType
    difficulty: DifficultyLevel
    duration_minutes: int
    url: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    rating: float = 0.0
    rating_count: int = 0
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserAchievement:
    """用户成就类"""
    id: str
    user_id: int
    achievement_type: AchievementType
    title: str
    description: str
    icon: str
    points: int
    earned_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSession:
    """学习会话类"""
    id: str
    user_id: int
    start_time: str
    end_time: Optional[str] = None
    duration_minutes: int = 0
    topics_studied: List[str] = field(default_factory=list)
    resources_used: List[str] = field(default_factory=list)
    notes: str = ""
    self_rating: int = 0  # 1-5
    difficulty_rating: int = 0  # 1-5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CommunityPost:
    """社区帖子类"""
    id: str
    user_id: int
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    likes: int = 0
    replies: int = 0
    views: int = 0
    is_question: bool = False
    is_answered: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AdvancedLearningFeatures:
    """高级学习功能类"""

    def __init__(self, db_path: str = "learning_advanced.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 用户扩展信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_extended (
                user_id INTEGER PRIMARY KEY,
                learning_style TEXT,
                preferred_difficulty TEXT,
                daily_goal_minutes INTEGER,
                favorite_times TEXT,
                notification_preferences TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 学习资源表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_resources (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                resource_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                duration_minutes INTEGER,
                url TEXT,
                content TEXT,
                tags TEXT,
                author TEXT,
                rating REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                prerequisites TEXT,
                learning_objectives TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')

        # 用户成就表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_achievements (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                achievement_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                icon TEXT,
                points INTEGER,
                earned_at TEXT,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES user_extended (user_id)
            )
        ''')

        # 学习会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes INTEGER DEFAULT 0,
                topics_studied TEXT,
                resources_used TEXT,
                notes TEXT,
                self_rating INTEGER DEFAULT 0,
                difficulty_rating INTEGER DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_extended (user_id)
            )
        ''')

        # 社区帖子表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS community_posts (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                likes INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                is_question BOOLEAN DEFAULT FALSE,
                is_answered BOOLEAN DEFAULT FALSE,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_extended (user_id)
            )
        ''')

        # 用户学习分析表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_learning_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT NOT NULL,
                total_minutes INTEGER DEFAULT 0,
                sessions_count INTEGER DEFAULT 0,
                avg_session_duration REAL DEFAULT 0.0,
                topics_completed TEXT,
                difficulty_distribution TEXT,
                engagement_score REAL DEFAULT 0.0,
                FOREIGN KEY (user_id) REFERENCES user_extended (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    # ========================================================================
    # 智能学习推荐系统
    # ========================================================================

    def analyze_learning_style(self, user_id: int) -> Dict[str, Any]:
        """分析用户学习风格"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取用户的学习会话数据
        cursor.execute('''
            SELECT duration_minutes, self_rating, difficulty_rating, notes
            FROM learning_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 50
        ''', (user_id,))

        sessions = cursor.fetchall()

        # 分析学习偏好
        style_scores = {
            LearningStyle.VISUAL.value: 0,
            LearningStyle.AUDITORY.value: 0,
            LearningStyle.KINESTHETIC.value: 0,
            LearningStyle.READING.value: 0
        }

        # 基于会话数据分析学习风格
        for session in sessions:
            duration, self_rating, difficulty_rating, notes = session

            if notes:
                notes_lower = notes.lower()
                if any(word in notes_lower for word in ['视频', '图像', '图表', '可视化']):
                    style_scores[LearningStyle.VISUAL.value] += 1
                if any(word in notes_lower for word in ['音频', '听力', '讲解', '播客']):
                    style_scores[LearningStyle.AUDITORY.value] += 1
                if any(word in notes_lower for word in ['实践', '操作', '动手', '项目']):
                    style_scores[LearningStyle.KINESTHETIC.value] += 1
                if any(word in notes_lower for word in ['阅读', '文档', '书籍', '文章']):
                    style_scores[LearningStyle.READING.value] += 1

        # 找出主导学习风格
        dominant_style = max(style_scores.items(), key=lambda x: x[1])[0] if sum(style_scores.values()) > 0 else LearningStyle.MIXED.value

        # 计算学习效率
        if sessions:
            avg_rating = sum(s[1] for s in sessions if s[1] > 0) / len([s for s in sessions if s[1] > 0])
            avg_duration = sum(s[0] for s in sessions) / len(sessions)
        else:
            avg_rating = 0
            avg_duration = 0

        analysis = {
            "dominant_style": dominant_style,
            "style_scores": style_scores,
            "avg_session_rating": avg_rating,
            "avg_session_duration": avg_duration,
            "total_sessions": len(sessions),
            "recommendations": self._generate_style_recommendations(dominant_style)
        }

        conn.close()
        return analysis

    def _generate_style_recommendations(self, style: str) -> List[str]:
        """根据学习风格生成推荐"""
        recommendations = {
            LearningStyle.VISUAL.value: [
                "推荐使用图表和思维导图来学习概念",
                "尝试观看视频教程和演示",
                "使用颜色编码来组织笔记",
                "制作流程图来理解复杂过程"
            ],
            LearningStyle.AUDITORY.value: [
                "推荐录制学习笔记并反复收听",
                "参与讨论组和语音交流",
                "使用播客和音频教程",
                "尝试向他人解释概念来加深理解"
            ],
            LearningStyle.KINESTHETIC.value: [
                "推荐通过实践项目来学习",
                "制作实体模型和实验",
                "边学习边操作代码",
                "参与动手实验和模拟"
            ],
            LearningStyle.READING.value: [
                "推荐阅读详细文档和书籍",
                "制作详细的书面笔记",
                "写总结和思维导图",
                "使用文字解释来巩固理解"
            ],
            LearningStyle.MIXED.value: [
                "结合多种学习方法",
                "根据内容类型选择最适合的学习方式",
                "灵活切换学习模式",
                "制作多模态学习材料"
            ]
        }

        return recommendations.get(style, ["尝试不同的学习方法来找到最适合的方式"])

    def get_personalized_recommendations(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取个性化推荐"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取用户学习分析
        analysis = self.analyze_learning_style(user_id)
        dominant_style = analysis["dominant_style"]

        # 获取用户最近学习的主题
        cursor.execute('''
            SELECT DISTINCT topics_studied
            FROM learning_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        ''', (user_id,))

        recent_topics = []
        for row in cursor.fetchall():
            if row[0]:
                topics = row[0].split(',')
                recent_topics.extend([t.strip() for t in topics if t.strip()])

        # 获取用户偏好难度
        cursor.execute('''
            SELECT preferred_difficulty
            FROM user_extended
            WHERE user_id = ?
        ''', (user_id,))

        result = cursor.fetchone()
        preferred_difficulty = result[0] if result else DifficultyLevel.INTERMEDIATE.value

        # 获取推荐资源
        recommendations = []

        # 基于最近主题推荐相关资源
        if recent_topics:
            placeholders = ','.join(['?' for _ in recent_topics])
            cursor.execute(f'''
                SELECT id, title, description, resource_type, difficulty,
                       duration_minutes, url, rating, tags
                FROM learning_resources
                WHERE tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags LIKE ?
                ORDER BY rating DESC, created_at DESC
                LIMIT ?
            ''', tuple([f'%{topic}%' for topic in recent_topics[:5]] + (limit,)))

            for row in cursor.fetchall():
                resource = {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "type": row[3],
                    "difficulty": row[4],
                    "duration": row[5],
                    "url": row[6],
                    "rating": row[7],
                    "tags": row[8].split(',') if row[8] else [],
                    "recommendation_reason": "基于最近学习主题",
                    "match_score": 0.9
                }
                recommendations.append(resource)

        # 基于学习风格推荐
        style_based_resources = self._get_style_based_resources(dominant_style, limit - len(recommendations))
        for resource in style_based_resources:
            resource["recommendation_reason"] = f"适合{dominant_style}学习风格"
            resource["match_score"] = 0.8
            recommendations.append(resource)

        # 基于难度推荐
        if len(recommendations) < limit:
            cursor.execute('''
                SELECT id, title, description, resource_type, difficulty,
                       duration_minutes, url, rating, tags
                FROM learning_resources
                WHERE difficulty = ?
                ORDER BY rating DESC
                LIMIT ?
            ''', (preferred_difficulty, limit - len(recommendations)))

            for row in cursor.fetchall():
                resource = {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "type": row[3],
                    "difficulty": row[4],
                    "duration": row[5],
                    "url": row[6],
                    "rating": row[7],
                    "tags": row[8].split(',') if row[8] else [],
                    "recommendation_reason": f"适合{preferred_difficulty}难度",
                    "match_score": = 0.7
                }
                recommendations.append(resource)

        conn.close()
        return recommendations[:limit]

    def _get_style_based_resources(self, style: str, limit: int) -> List[Dict[str, Any]]:
        """根据学习风格获取资源"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 根据学习风格筛选资源类型
        style_resource_mapping = {
            LearningStyle.VISUAL.value: ['video', 'course'],
            LearningStyle.AUDITORY.value: ['video', 'course'],
            LearningStyle.KINESTHETIC.value: ['project', 'tutorial', 'code'],
            LearningStyle.READING.value: ['article', 'book', 'paper'],
            LearningStyle.MIXED.value: ['video', 'article', 'course', 'project']
        }

        preferred_types = style_resource_mapping.get(style, ['article', 'video'])
        placeholders = ','.join(['?' for _ in preferred_types])

        cursor.execute(f'''
            SELECT id, title, description, resource_type, difficulty,
                   duration_minutes, url, rating, tags
            FROM learning_resources
            WHERE resource_type IN ({placeholders})
            ORDER BY rating DESC
            LIMIT ?
        ''', tuple(preferred_types) + (limit,))

        resources = []
        for row in cursor.fetchall():
            resources.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "type": row[3],
                "difficulty": row[4],
                "duration": row[5],
                "url": row[6],
                "rating": row[7],
                "tags": row[8].split(',') if row[8] else []
            })

        conn.close()
        return resources

    # ========================================================================
    # 学习资源管理
    # ========================================================================

    def add_resource(self, resource: LearningResource) -> bool:
        """添加学习资源"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO learning_resources
                (id, title, description, resource_type, difficulty, duration_minutes,
                 url, content, tags, author, rating, rating_count, prerequisites,
                 learning_objectives, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resource.id, resource.title, resource.description,
                resource.resource_type.value, resource.difficulty.value,
                resource.duration_minutes, resource.url, resource.content,
                ','.join(resource.tags), resource.author, resource.rating,
                resource.rating_count, ','.join(resource.prerequisites),
                ','.join(resource.learning_objectives), resource.created_at,
                resource.updated_at
            ))

            conn.commit()
            logger.info(f"资源添加成功: {resource.title}")
            return True

        except Exception as e:
            logger.error(f"添加资源失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def search_resources(self, query: str, resource_type: Optional[str] = None,
                        difficulty: Optional[str] = None, tags: Optional[List[str]] = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """搜索学习资源"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 构建搜索条件
        conditions = []
        params = []

        if query:
            conditions.append("(title LIKE ? OR description LIKE ? OR tags LIKE ?)")
            params.extend([f'%{query}%', f'%{query}%', f'%{query}%'])

        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)

        if difficulty:
            conditions.append("difficulty = ?")
            params.append(difficulty)

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%{tag}%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(f'''
            SELECT id, title, description, resource_type, difficulty,
                   duration_minutes, url, rating, rating_count, tags, author
            FROM learning_resources
            WHERE {where_clause}
            ORDER BY rating DESC, created_at DESC
            LIMIT ?
        ''', params + [limit])

        resources = []
        for row in cursor.fetchall():
            resources.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "type": row[3],
                "difficulty": row[4],
                "duration": row[5],
                "url": row[6],
                "rating": row[7],
                "rating_count": row[8],
                "tags": row[9].split(',') if row[9] else [],
                "author": row[10]
            })

        conn.close()
        return resources

    def rate_resource(self, resource_id: str, user_id: int, rating: int) -> bool:
        """评价学习资源"""
        if rating < 1 or rating > 5:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 获取当前评分
            cursor.execute('''
                SELECT rating, rating_count FROM learning_resources WHERE id = ?
            ''', (resource_id,))

            result = cursor.fetchone()
            if not result:
                return False

            current_rating, count = result

            # 计算新评分
            new_count = count + 1
            new_rating = (current_rating * count + rating) / new_count

            # 更新资源评分
            cursor.execute('''
                UPDATE learning_resources
                SET rating = ?, rating_count = ?, updated_at = ?
                WHERE id = ?
            ''', (new_rating, new_count, datetime.now().isoformat(), resource_id))

            conn.commit()
            logger.info(f"资源评分更新: {resource_id}, 新评分: {new_rating:.2f}")
            return True

        except Exception as e:
            logger.error(f"资源评分失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # ========================================================================
    # 成就系统
    # ========================================================================

    def check_and_award_achievements(self, user_id: int) -> List[UserAchievement]:
        """检查并颁发成就"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        new_achievements = []

        # 获取用户学习数据
        cursor.execute('''
            SELECT COUNT(*) as total_sessions,
                   SUM(duration_minutes) as total_minutes,
                   MAX(created_at) as last_session
            FROM learning_sessions
            WHERE user_id = ?
        ''', (user_id,))

        session_data = cursor.fetchone()
        total_sessions = session_data[0] or 0
        total_minutes = session_data[1] or 0
        last_session = session_data[2]

        # 检查连续学习天数成就
        if self._check_learning_streak(user_id, conn):
            achievement = self._create_achievement(
                user_id, AchievementType.LEARNING_STREAK,
                "连续学习者", "连续7天学习",
                "🔥", 50, {"streak_days": 7}
            )
            new_achievements.append(achievement)

        # 检查学习时长成就
        if total_minutes >= 1000:  # 1000分钟 = 约16.7小时
            achievement = self._create_achievement(
                user_id, AchievementType.SPEED_LEARNER,
                "学习达人", "累计学习超过1000分钟",
                "⏰", 100, {"total_minutes": total_minutes}
            )
            new_achievements.append(achievement)

        # 检查学习会话数成就
        if total_sessions >= 50:
            achievement = self._create_achievement(
                user_id, AchievementType.PRACTICE_EXPERT,
                "勤奋学习者", "完成50个学习会话",
                "📚", 75, {"total_sessions": total_sessions}
            )
            new_achievements.append(achievement)

        # 保存新成就
        for achievement in new_achievements:
            cursor.execute('''
                INSERT OR IGNORE INTO user_achievements
                (id, user_id, achievement_type, title, description, icon, points, earned_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                achievement.id, achievement.user_id, achievement.achievement_type.value,
                achievement.title, achievement.description, achievement.icon,
                achievement.points, achievement.earned_at, json.dumps(achievement.metadata)
            ))

        conn.commit()
        conn.close()

        return new_achievements

    def _check_learning_streak(self, user_id: int, conn) -> bool:
        """检查学习连续性"""
        cursor = conn.cursor()

        # 获取最近7天的学习记录
        cursor.execute('''
            SELECT DISTINCT DATE(start_time) as learning_date
            FROM learning_sessions
            WHERE user_id = ? AND start_time >= date('now', '-7 days')
            ORDER BY learning_date DESC
        ''', (user_id,))

        dates = [row[0] for row in cursor.fetchall()]

        # 检查是否连续7天
        if len(dates) >= 7:
            # 验证连续性
            expected_dates = []
            for i in range(7):
                expected_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                expected_dates.append(expected_date)

            return set(dates) >= set(expected_dates)

        return False

    def _create_achievement(self, user_id: int, achievement_type: AchievementType,
                          title: str, description: str, icon: str, points: int,
                          metadata: Dict[str, Any]) -> UserAchievement:
        """创建成就对象"""
        achievement_id = hashlib.md5(f"{user_id}{achievement_type.value}{title}{datetime.now()}".encode()).hexdigest()

        return UserAchievement(
            id=achievement_id,
            user_id=user_id,
            achievement_type=achievement_type,
            title=title,
            description=description,
            icon=icon,
            points=points,
            earned_at=datetime.now().isoformat(),
            metadata=metadata
        )

    def get_user_achievements(self, user_id: int) -> List[UserAchievement]:
        """获取用户成就列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, achievement_type, title, description, icon, points, earned_at, metadata
            FROM user_achievements
            WHERE user_id = ?
            ORDER BY earned_at DESC
        ''', (user_id,))

        achievements = []
        for row in cursor.fetchall():
            achievements.append(UserAchievement(
                id=row[0],
                user_id=user_id,
                achievement_type=AchievementType(row[1]),
                title=row[2],
                description=row[3],
                icon=row[4],
                points=row[5],
                earned_at=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            ))

        conn.close()
        return achievements

    # ========================================================================
    # 社区功能
    # ========================================================================

    def create_post(self, user_id: int, title: str, content: str, category: str,
                   tags: List[str], is_question: bool = False) -> str:
        """创建社区帖子"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        post_id = hashlib.md5(f"{user_id}{title}{content}{datetime.now()}".encode()).hexdigest()

        try:
            cursor.execute('''
                INSERT INTO community_posts
                (id, user_id, title, content, category, tags, is_question, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_id, user_id, title, content, category,
                ','.join(tags), is_question, datetime.now().isoformat(),
                datetime.now().isoformat()
            ))

            conn.commit()
            logger.info(f"帖子创建成功: {title}")
            return post_id

        except Exception as e:
            logger.error(f"创建帖子失败: {e}")
            conn.rollback()
            return ""
        finally:
            conn.close()

    def get_posts(self, category: Optional[str] = None, limit: int = 20,
                  is_question: Optional[bool] = None) -> List[CommunityPost]:
        """获取社区帖子列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if is_question is not None:
            conditions.append("is_question = ?")
            params.append(is_question)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(f'''
            SELECT id, user_id, title, content, category, tags, likes, replies, views,
                   is_question, is_answered, created_at, updated_at
            FROM community_posts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        ''', params + [limit])

        posts = []
        for row in cursor.fetchall():
            posts.append(CommunityPost(
                id=row[0], user_id=row[1], title=row[2], content=row[3],
                category=row[4], tags=row[5].split(',') if row[5] else [],
                likes=row[6], replies=row[7], views=row[8],
                is_question=bool(row[9]), is_answered=bool(row[10]),
                created_at=row[11], updated_at=row[12]
            ))

        conn.close()
        return posts

    def like_post(self, post_id: str, user_id: int) -> bool:
        """点赞帖子"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE community_posts
                SET likes = likes + 1
                WHERE id = ?
            ''', (post_id,))

            conn.commit()
            logger.info(f"帖子点赞成功: {post_id}")
            return True

        except Exception as e:
            logger.error(f"帖子点赞失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # ========================================================================
    # 学习分析和报告
    # ========================================================================

    def generate_learning_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """生成学习分析报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取时间范围内的学习数据
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT DATE(start_time) as date,
                   COUNT(*) as sessions_count,
                   SUM(duration_minutes) as total_minutes,
                   AVG(self_rating) as avg_rating,
                   AVG(difficulty_rating) as avg_difficulty
            FROM learning_sessions
            WHERE user_id = ? AND start_time >= ?
            GROUP BY DATE(start_time)
            ORDER BY date DESC
        ''', (user_id, start_date))

        daily_data = cursor.fetchall()

        # 计算总体统计
        total_sessions = sum(row[1] for row in daily_data)
        total_minutes = sum(row[2] for row in daily_data)
        avg_daily_sessions = total_sessions / days
        avg_daily_minutes = total_minutes / days

        # 最活跃的学习时间
        cursor.execute('''
            SELECT strftime('%H', start_time) as hour, COUNT(*) as count
            FROM learning_sessions
            WHERE user_id = ? AND start_time >= ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        ''', (user_id, start_date))

        peak_hour_result = cursor.fetchone()
        peak_hour = peak_hour_result[0] if peak_hour_result else "未知"

        # 最喜欢的学习主题
        cursor.execute('''
            SELECT topics_studied, COUNT(*) as count
            FROM learning_sessions
            WHERE user_id = ? AND start_time >= ?
            GROUP BY topics_studied
            ORDER BY count DESC
            LIMIT 5
        ''', (user_id, start_date))

        favorite_topics = []
        for row in cursor.fetchall():
            if row[0]:
                topics = [t.strip() for t in row[0].split(',')]
                for topic in topics:
                    favorite_topics.append({"topic": topic, "count": row[1]})

        # 学习效率分析
        efficiency_scores = []
        for row in daily_data:
            if row[3]:  # avg_rating
                efficiency_scores.append(row[3])

        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0

        # 学习趋势
        recent_avg = sum(row[1] for row in daily_data[:7]) / 7  # 最近7天
        earlier_avg = sum(row[1] for row in daily_data[7:14]) / 7 if len(daily_data) > 7 else recent_avg

        trend = "上升" if recent_avg > earlier_avg else "下降" if recent_avg < earlier_avg else "稳定"

        analytics = {
            "period_days": days,
            "summary": {
                "total_sessions": total_sessions,
                "total_minutes": total_minutes,
                "avg_daily_sessions": round(avg_daily_sessions, 2),
                "avg_daily_minutes": round(avg_daily_minutes, 2),
                "total_hours": round(total_minutes / 60, 2),
                "avg_efficiency": round(avg_efficiency, 2)
            },
            "patterns": {
                "peak_learning_hour": f"{peak_hour}:00",
                "most_active_days": [row[0] for row in sorted(daily_data, key=lambda x: x[1], reverse=True)[:3]],
                "favorite_topics": favorite_topics[:5]
            },
            "trends": {
                "learning_trend": trend,
                "recent_7_days_avg": recent_avg,
                "previous_7_days_avg": earlier_avg
            },
            "daily_breakdown": [
                {
                    "date": row[0],
                    "sessions": row[1],
                    "minutes": row[2],
                    "avg_rating": round(row[3], 2) if row[3] else 0,
                    "avg_difficulty": round(row[4], 2) if row[4] else 0
                }
                for row in daily_data
            ],
            "recommendations": self._generate_analytics_recommendations(analytics)
        }

        conn.close()
        return analytics

    def _generate_analytics_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """基于分析结果生成推荐"""
        recommendations = []

        # 基于学习频率推荐
        if analytics["summary"]["avg_daily_sessions"] < 1:
            recommendations.append("建议增加每日学习频率，保持学习连续性")

        # 基于学习时长推荐
        if analytics["summary"]["avg_daily_minutes"] < 30:
            recommendations.append("建议增加每日学习时长，目标至少30分钟")

        # 基于学习效率推荐
        if analytics["summary"]["avg_efficiency"] < 3.5:
            recommendations.append("学习效率偏低，建议尝试不同的学习方法和时间")

        # 基于学习趋势推荐
        if analytics["trends"]["learning_trend"] == "下降":
            recommendations.append("最近学习频率下降，建议重新评估学习计划")

        # 基于最活跃时间推荐
        peak_hour = analytics["patterns"]["peak_learning_hour"]
        recommendations.append(f"你在{peak_hour}学习效果最好，建议将重要学习安排在此时间")

        return recommendations

    def export_learning_data(self, user_id: int, format_type: str = "json") -> str:
        """导出学习数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取所有相关数据
        cursor.execute('''
            SELECT * FROM learning_sessions WHERE user_id = ? ORDER BY created_at
        ''', (user_id,))

        sessions = cursor.fetchall()

        cursor.execute('''
            SELECT * FROM user_achievements WHERE user_id = ? ORDER BY earned_at
        ''', (user_id,))

        achievements = cursor.fetchall()

        cursor.execute('''
            SELECT * FROM community_posts WHERE user_id = ? ORDER BY created_at
        ''', (user_id,))

        posts = cursor.fetchall()

        # 构建导出数据
        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "learning_sessions": [],
            "achievements": [],
            "community_posts": []
        }

        # 转换学习会话数据
        for session in sessions:
            export_data["learning_sessions"].append({
                "id": session[0],
                "start_time": session[2],
                "end_time": session[3],
                "duration_minutes": session[4],
                "topics_studied": session[5].split(',') if session[5] else [],
                "resources_used": session[6].split(',') if session[6] else [],
                "notes": session[7],
                "self_rating": session[8],
                "difficulty_rating": session[9],
                "created_at": session[10]
            })

        # 转换成就数据
        for achievement in achievements:
            export_data["achievements"].append({
                "id": achievement[0],
                "achievement_type": achievement[2],
                "title": achievement[3],
                "description": achievement[4],
                "icon": achievement[5],
                "points": achievement[6],
                "earned_at": achievement[7],
                "metadata": json.loads(achievement[8]) if achievement[8] else {}
            })

        # 转换社区帖子数据
        for post in posts:
            export_data["community_posts"].append({
                "id": post[0],
                "title": post[2],
                "content": post[3],
                "category": post[4],
                "tags": post[5].split(',') if post[5] else [],
                "likes": post[6],
                "replies": post[7],
                "views": post[8],
                "is_question": bool(post[9]),
                "is_answered": bool(post[10]),
                "created_at": post[11],
                "updated_at": post[12]
            })

        conn.close()

        # 根据格式返回数据
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        elif format_type.lower() == "csv":
            # 简化的CSV格式
            return self._convert_to_csv(export_data)
        else:
            return json.dumps(export_data, indent=2, ensure_ascii=False)

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """转换为CSV格式"""
        csv_lines = []

        # 学习会话CSV
        csv_lines.append("# 学习会话")
        csv_lines.append("日期,时长(分钟),主题,自评,难度评分,笔记")
        for session in data["learning_sessions"]:
            topics = ','.join(session["topics_studied"])
            csv_lines.append(f"{session['created_at'][:10]},{session['duration_minutes']},{topics},{session['self_rating']},{session['difficulty_rating']},{session['notes']}")

        csv_lines.append("\n# 成就")
        csv_lines.append("获得时间,成就类型,标题,积分")
        for achievement in data["achievements"]:
            csv_lines.append(f"{achievement['earned_at']},{achievement['achievement_type']},{achievement['title']},{achievement['points']}")

        return '\n'.join(csv_lines)


# ========================================================================
# 高级功能演示
# ========================================================================

def demo_advanced_features():
    """演示高级功能"""
    print("=" * 70)
    print("🚀 AI学习系统高级功能演示")
    print("=" * 70)

    # 创建高级功能实例
    advanced = AdvancedLearningFeatures()

    # 演示用户ID
    demo_user_id = 1

    print(f"\n📊 1. 学习风格分析 (用户ID: {demo_user_id})")
    analysis = advanced.analyze_learning_style(demo_user_id)
    print(f"   主导学习风格: {analysis['dominant_style']}")
    print(f"   平均会话评分: {analysis['avg_session_rating']:.2f}")
    print(f"   平均会话时长: {analysis['avg_session_duration']:.1f}分钟")
    print("   学习建议:")
    for i, rec in enumerate(analysis['recommendations'][:3], 1):
        print(f"     {i}. {rec}")

    print(f"\n🎯 2. 个性化推荐")
    recommendations = advanced.get_personalized_recommendations(demo_user_id, limit=5)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']}")
        print(f"      类型: {rec['type']}, 难度: {rec['difficulty']}")
        print(f"      推荐原因: {rec['recommendation_reason']}")
        print(f"      匹配度: {rec['match_score']:.1f}")

    print(f"\n🏆 3. 成就检查")
    new_achievements = advanced.check_and_award_achievements(demo_user_id)
    if new_achievements:
        print("   🎉 恭喜获得新成就:")
        for achievement in new_achievements:
            print(f"   • {achievement.icon} {achievement.title}: {achievement.description} (+{achievement.points}分)")
    else:
        print("   暂无新成就，继续努力！")

    print(f"\n📈 4. 学习分析报告")
    analytics = advanced.generate_learning_analytics(demo_user_id, days=30)
    summary = analytics['summary']
    print(f"   总学习时长: {summary['total_hours']}小时")
    print(f"   日均学习: {summary['avg_daily_minutes']}分钟")
    print(f"   学习效率: {summary['avg_efficiency']:.1f}/5")
    print(f"   学习趋势: {analytics['trends']['learning_trend']}")

    print(f"\n💡 5. 个性化建议")
    for i, rec in enumerate(analytics['recommendations'][:3], 1):
        print(f"   {i}. {rec}")

    print(f"\n🌐 6. 社区功能演示")
    # 创建示例帖子
    post_id = advanced.create_post(
        demo_user_id,
        "Transformer学习心得",
        "我最近在学习Transformer架构，有什么好的学习资源推荐吗？",
        "学习讨论",
        ["transformer", "深度学习", "nlp"],
        is_question=True
    )

    if post_id:
        print(f"   ✅ 帖子创建成功: {post_id}")

    # 获取社区帖子
    posts = advanced.get_posts(category="学习讨论", limit=3)
    print(f"   📋 最新帖子 ({len(posts)}条):")
    for i, post in enumerate(posts, 1):
        print(f"   {i}. {post.title} ({'问答' if post.is_question else '讨论'})")

    print(f"\n📤 7. 数据导出演示")
    export_data = advanced.export_learning_data(demo_user_id, format_type="json")
    print(f"   📊 数据导出完成: {len(export_data)}字符")
    print(f"   💾 包含学习会话、成就和社区数据")

    print("\n" + "=" * 70)
    print("✨ 高级功能演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_advanced_features()
