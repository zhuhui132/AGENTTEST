#!/usr/bin/env python3
"""
AI学习分析模块

提供深度的学习数据分析和可视化功能，包括：
- 学习进度可视化
- 知识掌握度分析
- 学习效率评估
- 个性化学习洞察
- 学习路径优化建议
- 学习预测和规划

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-13
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AnalyticsType(Enum):
    """分析类型枚举"""
    PROGRESS_TREND = "progress_trend"           # 进度趋势
    KNOWLEDGE_MASTERY = "knowledge_mastery"     # 知识掌握度
    EFFICIENCY_ANALYSIS = "efficiency_analysis"   # 效率分析
    LEARNING_PATTERN = "learning_pattern"         # 学习模式
    PREDICTIVE_ANALYSIS = "predictive_analysis"   # 预测分析
    COMPARATIVE_ANALYSIS = "comparative_analysis" # 对比分析


class VisualizationType(Enum):
    """可视化类型枚举"""
    LINE_CHART = "line_chart"                 # 折线图
    BAR_CHART = "bar_chart"                   # 柱状图
    HEATMAP = "heatmap"                        # 热力图
    SCATTER_PLOT = "scatter_plot"               # 散点图
    RADAR_CHART = "radar_chart"                 # 雷达图
    PIE_CHART = "pie_chart"                     # 饼图
    AREA_CHART = "area_chart"                   # 面积图
    BOX_PLOT = "box_plot"                       # 箱线图


@dataclass
class LearningInsight:
    """学习洞察类"""
    category: str
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearningPrediction:
    """学习预测类"""
    prediction_type: str
    predicted_value: Any
    confidence: float
    timeframe: str
    factors: List[str] = field(default_factory=list)
    methodology: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class LearningAnalytics:
    """学习分析类"""

    def __init__(self, db_path: str = "learning_system.db"):
        self.db_path = db_path
        self.ensure_database_connection()

    def ensure_database_connection(self):
        """确保数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table'")
            conn.close()
        except:
            raise ConnectionError(f"无法连接到数据库: {self.db_path}")

    def generate_comprehensive_analysis(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """生成综合学习分析"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取基础学习数据
        learning_data = self._get_learning_data(cursor, user_id, days)
        test_data = self._get_test_data(cursor, user_id, days)
        resource_data = self._get_resource_data(cursor, user_id, days)

        # 各项分析
        analysis = {
            "user_id": user_id,
            "analysis_period": days,
            "analysis_date": datetime.now().isoformat(),
            "progress_trend": self._analyze_progress_trend(learning_data),
            "knowledge_mastery": self._analyze_knowledge_mastery(test_data, learning_data),
            "efficiency_analysis": self._analyze_efficiency(learning_data, test_data),
            "learning_patterns": self._analyze_learning_patterns(learning_data),
            "resource_utilization": self._analyze_resource_utilization(resource_data),
            "predictions": self._generate_predictions(learning_data, test_data),
            "insights": self._generate_insights(learning_data, test_data),
            "recommendations": self._generate_recommendations(learning_data, test_data)
        }

        conn.close()
        return analysis

    def _get_learning_data(self, cursor, user_id: int, days: int) -> List[Dict]:
        """获取学习数据"""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT date, duration_minutes, topics_studied, self_rating, difficulty_rating, notes
            FROM learning_sessions
            WHERE user_id = ? AND date >= ?
            ORDER BY date ASC
        ''', (user_id, start_date))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "date": row[0],
                "duration": row[1],
                "topics": row[2].split(',') if row[2] else [],
                "self_rating": row[3] or 0,
                "difficulty_rating": row[4] or 0,
                "notes": row[5] or ""
            })

        return sessions

    def _get_test_data(self, cursor, user_id: int, days: int) -> List[Dict]:
        """获取测试数据"""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT date, test_name, total_questions, correct_answers, score, execution_time
            FROM test_results
            WHERE user_id = ? AND date >= ?
            ORDER BY date ASC
        ''', (user_id, start_date))

        tests = []
        for row in cursor.fetchall():
            tests.append({
                "date": row[0],
                "test_name": row[1],
                "total_questions": row[2],
                "correct_answers": row[3],
                "score": row[4],
                "execution_time": row[5] or 0
            })

        return tests

    def _get_resource_data(self, cursor, user_id: int, days: int) -> List[Dict]:
        """获取资源使用数据"""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT date, resources_used
            FROM learning_sessions
            WHERE user_id = ? AND date >= ?
            ORDER BY date ASC
        ''', (user_id, start_date))

        resources = []
        for row in cursor.fetchall():
            if row[1]:
                resource_list = row[1].split(',')
                for resource in resource_list:
                    if resource.strip():
                        resources.append({
                            "date": row[0],
                            "resource": resource.strip()
                        })

        return resources

    def _analyze_progress_trend(self, learning_data: List[Dict]) -> Dict[str, Any]:
        """分析进度趋势"""
        if not learning_data:
            return {"trend": "no_data", "slope": 0, "correlation": 0}

        # 转换为DataFrame
        df = pd.DataFrame(learning_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 计算累计学习时长
        df['cumulative_hours'] = df['duration'].cumsum() / 60

        # 计算滑动平均
        df['rolling_avg_duration'] = df['duration'].rolling(window=7, min_periods=1).mean()
        df['rolling_avg_rating'] = df['self_rating'].rolling(window=7, min_periods=1).mean()

        # 趋势分析
        from scipy import stats

        # 线性回归分析学习时长趋势
        x = np.arange(len(df))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, df['duration'])

        # 学习频率分析
        daily_counts = df.groupby('date').size()
        recent_freq = daily_counts.tail(7).mean()
        earlier_freq = daily_counts.head(-7).mean() if len(daily_counts) > 7 else recent_freq

        return {
            "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "total_hours": float(df['cumulative_hours'].iloc[-1] if len(df) > 0 else 0),
            "avg_session_duration": float(df['duration'].mean()),
            "avg_self_rating": float(df['self_rating'].mean()),
            "rolling_avg_duration": df['rolling_avg_duration'].tolist() if len(df) > 0 else [],
            "rolling_avg_rating": df['rolling_avg_rating'].tolist() if len(df) > 0 else [],
            "frequency_trend": "increasing" if recent_freq > earlier_freq else "decreasing",
            "recent_frequency": float(recent_freq),
            "earlier_frequency": float(earlier_freq),
            "daily_hours": (df.groupby('date')['duration'].sum() / 60).to_dict()
        }

    def _analyze_knowledge_mastery(self, test_data: List[Dict], learning_data: List[Dict]) -> Dict[str, Any]:
        """分析知识掌握度"""
        if not test_data:
            return {"overall_mastery": 0, "topic_mastery": {}, "weak_areas": []}

        # 转换为DataFrame
        test_df = pd.DataFrame(test_data)
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df = test_df.sort_values('date')

        # 总体掌握度
        overall_mastery = test_df['score'].mean()

        # 按主题分组分析
        topic_scores = {}
        for _, row in test_df.iterrows():
            test_name = row['test_name']
            # 简单的主题提取（实际应用中需要更复杂的逻辑）
            topic = self._extract_topic_from_test_name(test_name)

            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(row['score'])

        # 计算各主题平均分
        topic_mastery = {}
        for topic, scores in topic_scores.items():
            topic_mastery[topic] = {
                "avg_score": np.mean(scores),
                "max_score": np.max(scores),
                "min_score": np.min(scores),
                "std_score": np.std(scores),
                "test_count": len(scores),
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
            }

        # 识别弱项
        weak_areas = []
        for topic, data in topic_mastery.items():
            if data['avg_score'] < 70:  # 70分以下为弱项
                weak_areas.append({
                    "topic": topic,
                    "avg_score": data['avg_score'],
                    "test_count": data['test_count']
                })

        # 按难度分析
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            difficulty_scores = learning_df.groupby('difficulty_rating')['self_rating'].mean().to_dict()
        else:
            difficulty_scores = {}

        return {
            "overall_mastery": float(overall_mastery),
            "topic_mastery": topic_mastery,
            "weak_areas": weak_areas,
            "difficulty_mastery": difficulty_scores,
            "score_distribution": {
                "excellent": len([s for s in test_df['score'] if s >= 90]),
                "good": len([s for s in test_df['score'] if 80 <= s < 90]),
                "average": len([s for s in test_df['score'] if 70 <= s < 80]),
                "below_average": len([s for s in test_df['score'] if s < 70])
            },
            "improvement_rate": self._calculate_improvement_rate(test_df)
        }

    def _analyze_efficiency(self, learning_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """分析学习效率"""
        if not learning_data:
            return {"overall_efficiency": 0, "efficiency_trend": "stable"}

        learning_df = pd.DataFrame(learning_data)

        # 计算效率指标
        # 1. 时间效率：学习时长与自评等级的关系
        time_efficiency = []
        for _, row in learning_df.iterrows():
            if row['duration'] > 0:
                # 效率 = 自评等级 / 学习时长（小时）
                efficiency_score = (row['self_rating'] / 5.0) / (row['duration'] / 60.0)
                time_efficiency.append(efficiency_score)

        # 2. 学习稳定性：自评等级的方差
        rating_stability = 1 - (learning_df['self_rating'].std() / 5.0) if learning_df['self_rating'].std() > 0 else 1.0

        # 3. 持续性效率：连续学习天数与平均效率的关系
        consecutive_days = self._calculate_consecutive_learning_days(learning_data)

        # 4. 知识转化效率：学习时长与测试成绩的关系
        knowledge_conversion = 0
        if test_data:
            test_df = pd.DataFrame(test_data)
            test_df['date'] = pd.to_datetime(test_df['date'])

            # 合并学习数据和测试数据
            for _, test_row in test_df.iterrows():
                # 查找该测试日期前7天的学习数据
                week_before = test_row['date'] - timedelta(days=7)
                week_learning = learning_df[
                    (pd.to_datetime(learning_df['date']) >= week_before) &
                    (pd.to_datetime(learning_df['date']) <= test_row['date'])
                ]

                if not week_learning.empty:
                    total_hours = week_learning['duration'].sum() / 60
                    if total_hours > 0:
                        conversion_rate = test_row['score'] / total_hours
                        knowledge_conversion += conversion_rate

            knowledge_conversion = knowledge_conversion / len(test_df)

        # 综合效率评分
        avg_time_efficiency = np.mean(time_efficiency) if time_efficiency else 0
        overall_efficiency = (
            avg_time_efficiency * 0.3 +
            rating_stability * 0.3 +
            (consecutive_days / 30) * 0.2 +  # 假设30天为满分
            knowledge_conversion * 0.2
        )

        # 效率趋势
        if len(learning_df) >= 14:  # 至少两周数据
            first_week_efficiency = self._calculate_week_efficiency(learning_df.head(7))
            second_week_efficiency = self._calculate_week_efficiency(learning_df.tail(7))
            efficiency_trend = "improving" if second_week_efficiency > first_week_efficiency else "declining"
        else:
            efficiency_trend = "stable"

        return {
            "overall_efficiency": float(overall_efficiency),
            "time_efficiency": float(avg_time_efficiency),
            "rating_stability": float(rating_stability),
            "consecutive_days": consecutive_days,
            "knowledge_conversion": float(knowledge_conversion),
            "efficiency_trend": efficiency_trend,
            "daily_efficiency": [
                {
                    "date": row['date'],
                    "efficiency": (row['self_rating'] / 5.0) / (row['duration'] / 60.0) if row['duration'] > 0 else 0
                }
                for _, row in learning_df.iterrows()
            ]
        }

    def _analyze_learning_patterns(self, learning_data: List[Dict]) -> Dict[str, Any]:
        """分析学习模式"""
        if not learning_data:
            return {"patterns": {}, "insights": []}

        learning_df = pd.DataFrame(learning_data)
        learning_df['date'] = pd.to_datetime(learning_df['date'])
        learning_df['weekday'] = learning_df['date'].dt.day_name()
        learning_df['hour'] = learning_df['date'].dt.hour

        # 1. 时间模式分析
        weekday_hours = learning_df.groupby(['weekday', 'hour'])['duration'].mean().unstack(fill_value=0)
        peak_weekday = weekday_hours.sum(axis=1).idxmax()
        peak_hour = weekday_hours.loc[peak_weekday].idxmax()

        # 2. 学习时长分布
        duration_stats = {
            "mean": learning_df['duration'].mean(),
            "median": learning_df['duration'].median(),
            "std": learning_df['duration'].std(),
            "min": learning_df['duration'].min(),
            "max": learning_df['duration'].max(),
            "quartiles": np.percentile(learning_df['duration'], [25, 50, 75]).tolist()
        }

        # 3. 主题偏好分析
        topic_frequency = {}
        for _, row in learning_df.iterrows():
            for topic in row['topics']:
                if topic.strip():
                    topic_frequency[topic.strip()] = topic_frequency.get(topic.strip(), 0) + 1

        # 4. 学习节奏分析
        if len(learning_df) > 1:
            learning_df['time_diff'] = learning_df['date'].diff().dt.days
            avg_interval = learning_df['time_diff'].mean()
            regularity_score = 1 / (1 + learning_df['time_diff'].std()) if learning_df['time_diff'].std() > 0 else 1.0
        else:
            avg_interval = 0
            regularity_score = 1.0

        # 5. 难度偏好分析
        difficulty_preference = learning_df.groupby('difficulty_rating').size().to_dict()

        patterns = {
            "temporal_patterns": {
                "peak_weekday": peak_weekday,
                "peak_hour": peak_hour,
                "weekday_heatmap": weekday_hours.to_dict(),
                "hourly_distribution": learning_df.groupby('hour')['duration'].mean().to_dict()
            },
            "duration_patterns": duration_stats,
            "topic_preferences": topic_frequency,
            "rhythm_patterns": {
                "avg_interval_days": float(avg_interval),
                "regularity_score": float(regularity_score),
                "consistency": "regular" if regularity_score > 0.7 else "irregular"
            },
            "difficulty_preferences": difficulty_preference,
            "learning_velocity": len(learning_df) / 30.0  # 假设30天内的学习速度
        }

        # 生成洞察
        insights = []

        # 时间洞察
        if peak_hour >= 20 or peak_hour <= 6:
            insights.append("倾向于在夜间或清晨学习")
        elif 9 <= peak_hour <= 17:
            insights.append("倾向于在白天学习")

        # 难度洞察
        if difficulty_preference:
            max_difficulty = max(difficulty_preference.keys(), key=lambda k: difficulty_preference[k])
            insights.append(f"偏好挑战{max_difficulty}难度的内容")

        # 节奏洞察
        if avg_interval < 2:
            insights.append("学习节奏紧凑，建议注意休息")
        elif avg_interval > 7:
            insights.append("学习间隔较长，建议提高学习频率")

        return {
            "patterns": patterns,
            "insights": insights
        }

    def _analyze_resource_utilization(self, resource_data: List[Dict]) -> Dict[str, Any]:
        """分析资源使用情况"""
        if not resource_data:
            return {"utilization_rate": 0, "resource_preferences": {}}

        # 统计资源使用频率
        resource_df = pd.DataFrame(resource_data)
        resource_count = resource_df.groupby('resource').size().sort_values(ascending=False)

        # 资源类型分析
        def classify_resource(resource_name):
            name = resource_name.lower()
            if any(keyword in name for keyword in ['video', '视频', 'course']):
                return 'video'
            elif any(keyword in name for keyword in ['book', '书籍', 'article']):
                return 'reading'
            elif any(keyword in name for keyword in ['tutorial', '教程']):
                return 'tutorial'
            elif any(keyword in name for keyword in ['practice', '练习', 'lab']):
                return 'practice'
            else:
                return 'other'

        resource_df['resource_type'] = resource_df['resource'].apply(classify_resource)
        type_distribution = resource_df['resource_type'].value_counts().to_dict()

        # 资源使用趋势
        resource_df['date'] = pd.to_datetime(resource_df['date'])
        daily_resources = resource_df.groupby('date').size()
        recent_avg = daily_resources.tail(7).mean()
        earlier_avg = daily_resources.head(-7).mean() if len(daily_resources) > 7 else recent_avg

        return {
            "utilization_rate": float(len(resource_data) / 30),  # 假设30天基准
            "most_used_resources": resource_count.head(10).to_dict(),
            "resource_type_distribution": type_distribution,
            "utilization_trend": "increasing" if recent_avg > earlier_avg else "stable",
            "diversity_score": float(len(resource_count) / len(resource_data)) if resource_data else 0,
            "daily_usage": daily_resources.to_dict()
        }

    def _generate_predictions(self, learning_data: List[Dict], test_data: List[Dict]) -> List[LearningPrediction]:
        """生成学习预测"""
        predictions = []

        # 1. 学习进度预测
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            learning_df['date'] = pd.to_datetime(learning_df['date'])
            learning_df = learning_df.sort_values('date')

            # 使用线性回归预测未来学习时长
            if len(learning_df) >= 7:
                x = np.arange(len(learning_df))
                y = learning_df['duration'].values

                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # 预测未来7天的学习时长
                future_x = np.arange(len(learning_df), len(learning_df) + 7)
                future_y = slope * future_x + intercept

                predictions.append(LearningPrediction(
                    prediction_type="learning_hours",
                    predicted_value=future_y.tolist(),
                    confidence=float(r_value ** 2),
                    timeframe="7_days",
                    factors=["historical_trend", "seasonal_patterns"],
                    methodology="linear_regression"
                ))

        # 2. 测试成绩预测
        if test_data and learning_data:
            # 使用最近的学习表现预测测试成绩
            test_df = pd.DataFrame(test_data)
            learning_df = pd.DataFrame(learning_data)

            # 计算最近一周的学习指标
            recent_learning = learning_df.tail(7)
            if not recent_learning.empty:
                avg_duration = recent_learning['duration'].mean()
                avg_rating = recent_learning['self_rating'].mean()

                # 简单的预测模型
                predicted_score = min(100, avg_rating * 20 + (avg_duration / 60) * 5)

                predictions.append(LearningPrediction(
                    prediction_type="test_score",
                    predicted_value=float(predicted_score),
                    confidence=0.7,
                    timeframe="next_test",
                    factors=["recent_learning", "self_assessment", "time_investment"],
                    methodology="heuristic_model"
                ))

        # 3. 目标达成预测
        if learning_data:
            # 计算学习速度和当前进度
            learning_df = pd.DataFrame(learning_data)
            total_hours = learning_df['duration'].sum() / 60
            current_velocity = total_hours / 30  # 假设30天内的平均速度

            # 预测达成100小时目标所需时间
            if current_velocity > 0:
                days_to_100_hours = (100 - total_hours) / (current_velocity * 7) * 7

                predictions.append(LearningPrediction(
                    prediction_type="goal_completion",
                    predicted_value=float(days_to_100_hours),
                    confidence=0.6,
                    timeframe="hours_goal",
                    factors=["learning_velocity", "consistency"],
                    methodology="projection_model"
                ))

        return predictions

    def _generate_insights(self, learning_data: List[Dict], test_data: List[Dict]) -> List[LearningInsight]:
        """生成学习洞察"""
        insights = []

        # 1. 学习强度洞察
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            daily_hours = learning_df.groupby('date')['duration'].sum() / 60
            avg_daily = daily_hours.mean()

            if avg_daily >= 3:
                category = "学习强度"
                title = "高强度学习者"
                description = "您每天平均学习时间超过3小时，学习强度很高"
                confidence = 0.9
                recommendations = ["注意劳逸结合，避免过度疲劳", "可以适当减少单次学习时长，增加学习频率"]
            elif avg_daily >= 1:
                category = "学习强度"
                title = "适度学习者"
                description = "您每天学习时间适中，有利于知识的长期积累"
                confidence = 0.8
                recommendations = ["保持当前学习节奏", "适当增加一些挑战性内容"]
            else:
                category = "学习强度"
                title = "低强度学习者"
                description = "您每天学习时间较少，建议增加学习投入"
                confidence = 0.8
                recommendations = ["制定每日学习计划", "从短时间、高频率开始", "寻找有趣的学习内容提高动力"]

            insights.append(LearningInsight(
                category=category,
                title=title,
                description=description,
                data={"avg_daily_hours": float(avg_daily)},
                confidence=confidence,
                recommendations=recommendations
            ))

        # 2. 学习效果洞察
        if test_data:
            test_df = pd.DataFrame(test_data)
            avg_score = test_df['score'].mean()
            score_std = test_df['score'].std()

            if avg_score >= 85 and score_std < 10:
                category = "学习效果"
                title = "稳定优秀型"
                description = "您的测试成绩优秀且稳定，学习效果很好"
                confidence = 0.9
                recommendations = ["继续保持学习方法", "可以尝试更高级的内容"]
            elif avg_score >= 70:
                category = "学习效果"
                title = "良好改进型"
                description = "您的学习成绩良好，还有提升空间"
                confidence = 0.7
                recommendations = ["分析错题原因", "加强薄弱环节练习"]
            else:
                category = "学习效果"
                title = "需要提升型"
                description = "您的学习成绩有待提高，建议调整学习方法"
                confidence = 0.8
                recommendations = ["回到基础，巩固基础知识", "寻求学习方法和技巧指导"]

            insights.append(LearningInsight(
                category=category,
                title=title,
                description=description,
                data={"avg_score": float(avg_score), "score_stability": float(1 - (score_std / 100))},
                confidence=confidence,
                recommendations=recommendations
            ))

        # 3. 学习模式洞察
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            rating_diff = learning_df['self_rating'].max() - learning_df['self_rating'].min()

            if rating_diff >= 3:
                category = "学习模式"
                title = "波动较大型"
                description = "您的学习自评差异较大，学习状态不够稳定"
                confidence = 0.7
                recommendations = ["保持稳定的学习时间", "寻找影响学习状态的因素", "建立学习仪式感"]
            elif rating_diff >= 1:
                category = "学习模式"
                title = "适度波动型"
                description = "您的学习状态有一定波动，属于正常范围"
                confidence = 0.6
                recommendations = ["注意状态管理", "在学习前做好准备工作"]
            else:
                category = "学习模式"
                title = "稳定型"
                description = "您的学习状态稳定，值得肯定"
                confidence = 0.8
                recommendations = ["保持良好的学习习惯", "可以适当增加学习挑战"]

            insights.append(LearningInsight(
                category=category,
                title=title,
                description=description,
                data={"rating_variance": float(rating_diff)},
                confidence=confidence,
                recommendations=recommendations
            ))

        return insights

    def _generate_recommendations(self, learning_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """生成学习建议"""
        recommendations = {
            "time_management": [],
            "content_adjustment": [],
            "learning_strategy": [],
            "goal_setting": [],
            "resource_optimization": []
        }

        # 时间管理建议
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            avg_session = learning_df['duration'].mean()

            if avg_session < 30:  # 30分钟
                recommendations["time_management"].append("建议延长单次学习时间到45-60分钟")
                recommendations["time_management"].append("使用番茄工作法提高专注度")
            elif avg_session > 120:  # 2小时
                recommendations["time_management"].append("建议将长时间学习分割为短时段")
                recommendations["time_management"].append("每45-50分钟休息5-10分钟")

        # 内容调整建议
        if test_data:
            test_df = pd.DataFrame(test_data)
            low_score_tests = test_df[test_df['score'] < 70]

            if not low_score_tests.empty:
                recommendations["content_adjustment"].append("重点复习低分测试相关内容")
                recommendations["content_adjustment"].append("寻求额外的学习资源和帮助")

        # 学习策略建议
        if learning_data:
            learning_df = pd.DataFrame(learning_data)
            self_rating_mean = learning_df['self_rating'].mean()

            if self_rating_mean < 3:
                recommendations["learning_strategy"].append("检查学习方法的适用性")
                recommendations["learning_strategy"].append("尝试不同的学习方式（视觉、听觉、实践）")
            else:
                recommendations["learning_strategy"].append("当前学习策略有效，继续坚持")
                recommendations["learning_strategy"].append("可以尝试更高级的学习技巧")

        return recommendations

    def create_visualizations(self, analysis: Dict[str, Any], output_dir: str = "visualizations") -> Dict[str, str]:
        """创建可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        visualization_files = {}

        # 1. 进度趋势图
        if 'progress_trend' in analysis:
            progress_fig = self._create_progress_trend_chart(analysis['progress_trend'])
            progress_file = os.path.join(output_dir, "progress_trend.html")
            progress_fig.write_html(progress_file)
            visualization_files['progress_trend'] = progress_file

        # 2. 知识掌握度雷达图
        if 'knowledge_mastery' in analysis:
            mastery_fig = self._create_mastery_radar_chart(analysis['knowledge_mastery'])
            mastery_file = os.path.join(output_dir, "knowledge_mastery.html")
            mastery_fig.write_html(mastery_file)
            visualization_files['knowledge_mastery'] = mastery_file

        # 3. 学习模式热力图
        if 'learning_patterns' in analysis:
            patterns_fig = self._create_patterns_heatmap(analysis['learning_patterns'])
            patterns_file = os.path.join(output_dir, "learning_patterns.html")
            patterns_fig.write_html(patterns_file)
            visualization_files['learning_patterns'] = patterns_file

        # 4. 效率分析图
        if 'efficiency_analysis' in analysis:
            efficiency_fig = self._create_efficiency_chart(analysis['efficiency_analysis'])
            efficiency_file = os.path.join(output_dir, "efficiency_analysis.html")
            efficiency_fig.write_html(efficiency_file)
            visualization_files['efficiency_analysis'] = efficiency_file

        return visualization_files

    def _create_progress_trend_chart(self, progress_data: Dict[str, Any]) -> go.Figure:
        """创建进度趋势图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('学习时长趋势', '自评等级趋势', '每日学习时长', '滚动平均'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 学习时长趋势
        if 'daily_hours' in progress_data:
            dates = list(progress_data['daily_hours'].keys())
            hours = list(progress_data['daily_hours'].values())

            fig.add_trace(
                go.Scatter(x=dates, y=hours, mode='lines+markers', name='学习时长'),
                row=1, col=1
            )

        # 自评等级趋势
        if 'rolling_avg_rating' in progress_data:
            fig.add_trace(
                go.Scatter(y=progress_data['rolling_avg_rating'], mode='lines', name='自评等级'),
                row=1, col=2
            )

        # 每日学习时长柱状图
        if 'daily_hours' in progress_data:
            fig.add_trace(
                go.Bar(x=list(progress_data['daily_hours'].keys()),
                       y=list(progress_data['daily_hours'].values()),
                       name='每日时长'),
                row=2, col=1
            )

        # 滚动平均
        if 'rolling_avg_duration' in progress_data:
            fig.add_trace(
                go.Scatter(y=progress_data['rolling_avg_duration'], mode='lines', name='滚动平均时长'),
                row=2, col=2
            )

        fig.update_layout(height=800, title_text="学习进度趋势分析", showlegend=False)
        return fig

    def _create_mastery_radar_chart(self, mastery_data: Dict[str, Any]) -> go.Figure:
        """创建知识掌握度雷达图"""
        if not mastery_data.get('topic_mastery'):
            return go.Figure()

        topics = list(mastery_data['topic_mastery'].keys())
        scores = [mastery_data['topic_mastery'][topic]['avg_score'] for topic in topics]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=topics,
            fill='toself',
            name='知识掌握度'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="知识掌握度雷达图"
        )

        return fig

    def _create_patterns_heatmap(self, patterns_data: Dict[str, Any]) -> go.Figure:
        """创建学习模式热力图"""
        if 'temporal_patterns' not in patterns_data:
            return go.Figure()

        temporal = patterns_data['temporal_patterns']

        if 'weekday_heatmap' in temporal:
            # 这里需要转换为适合热力图的数据格式
            # 简化处理，实际需要更复杂的数据转换
            hours = list(range(24))
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            # 创建示例数据
            z = np.random.rand(7, 24)  # 实际应使用真实数据

            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=hours,
                y=weekdays,
                colorscale='Viridis'
            ))

            fig.update_layout(
                title='学习时间热力图',
                xaxis_title='小时',
                yaxis_title='星期'
            )

            return fig

        return go.Figure()

    def _create_efficiency_chart(self, efficiency_data: Dict[str, Any]) -> go.Figure:
        """创建效率分析图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('效率指标', '每日效率', '效率分布', '时间效率趋势'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )

        # 综合效率指标
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=efficiency_data.get('overall_efficiency', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "综合效率评分"},
                gauge={'axis': {'range': [None, 100]},
                threshold={'line': {"color": "red", "width": 4}, 'thickness': 0.75, 'value': 70},
                steps=[{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 70], 'color': "gray"}]
            ),
            row=1, col=1
        )

        # 每日效率柱状图
        if 'daily_efficiency' in efficiency_data:
            daily_eff = efficiency_data['daily_efficiency']
            dates = [item['date'] for item in daily_eff]
            efficiencies = [item['efficiency'] * 100 for item in daily_eff]

            fig.add_trace(
                go.Bar(x=dates, y=efficiencies, name='每日效率'),
                row=1, col=2
            )

        # 效率分布箱线图
        if 'daily_efficiency' in efficiency_data:
            daily_eff = efficiency_data['daily_efficiency']
            efficiencies = [item['efficiency'] * 100 for item in daily_eff]

            fig.add_trace(
                go.Box(y=efficiencies, name='效率分布'),
                row=2, col=1
            )

        # 时间效率趋势
        if 'daily_efficiency' in efficiency_data:
            daily_eff = efficiency_data['daily_efficiency']
            efficiencies = [item['efficiency'] * 100 for item in daily_eff]

            fig.add_trace(
                go.Scatter(y=efficiencies, mode='lines+markers', name='效率趋势'),
                row=2, col=2
            )

        fig.update_layout(height=800, title_text="学习效率分析")
        return fig

    # 辅助方法
    def _extract_topic_from_test_name(self, test_name: str) -> str:
        """从测试名称提取主题"""
        # 简化实现，实际应用中需要更复杂的NLP处理
        keywords = {
            "Python": ["python", "py", "programming"],
            "机器学习": ["machine", "ml", "learning"],
            "深度学习": ["deep", "neural", "network"],
            "算法": ["algorithm", "sorting", "search"],
            "数据库": ["database", "sql", "query"]
        }

        name_lower = test_name.lower()
        for topic, kw_list in keywords.items():
            if any(kw in name_lower for kw in kw_list):
                return topic

        return "其他"

    def _calculate_improvement_rate(self, test_df: pd.DataFrame) -> float:
        """计算改进率"""
        if len(test_df) < 2:
            return 0.0

        # 计算最近与最早的平均分差异
        recent_avg = test_df.tail(5)['score'].mean() if len(test_df) >= 5 else test_df.tail(1)['score'].iloc[0]
        early_avg = test_df.head(5)['score'].mean() if len(test_df) >= 10 else test_df.head(1)['score'].iloc[0]

        improvement = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        return float(improvement)

    def _calculate_consecutive_learning_days(self, learning_data: List[Dict]) -> int:
        """计算连续学习天数"""
        if not learning_data:
            return 0

        dates = sorted(set([d['date'] for d in learning_data]))
        consecutive = 1
        max_consecutive = 1

        for i in range(1, len(dates)):
            curr_date = datetime.strptime(dates[i], '%Y-%m-%d')
            prev_date = datetime.strptime(dates[i-1], '%Y-%m-%d')

            if (curr_date - prev_date).days == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1

        return max_consecutive

    def _calculate_week_efficiency(self, week_data: pd.DataFrame) -> float:
        """计算一周的学习效率"""
        if week_data.empty:
            return 0.0

        # 计算效率分数
        total_efficiency = sum(
            (row['self_rating'] / 5.0) / (row['duration'] / 60.0)
            for _, row in week_data.iterrows() if row['duration'] > 0
        )

        return total_efficiency / len(week_data) if len(week_data) > 0 else 0.0


def demo_learning_analytics():
    """演示学习分析功能"""
    print("=" * 70)
    print("📊 AI学习分析系统演示")
    print("=" * 70)

    # 创建分析器
    analytics = LearningAnalytics()

    # 演示用户ID
    demo_user_id = 1

    print(f"\n🔍 为用户 {demo_user_id} 生成综合学习分析...")

    # 生成综合分析
    try:
        analysis = analytics.generate_comprehensive_analysis(demo_user_id, days=30)

        print("✅ 分析生成成功！")

        # 显示关键指标
        print("\n📈 关键指标:")
        progress = analysis.get('progress_trend', {})
        print(f"   总学习时长: {progress.get('total_hours', 0):.1f} 小时")
        print(f"   平均每次学习: {progress.get('avg_session_duration', 0):.1f} 分钟")
        print(f"   学习趋势: {progress.get('trend', 'stable')}")

        mastery = analysis.get('knowledge_mastery', {})
        print(f"   知识掌握度: {mastery.get('overall_mastery', 0):.1f}%")

        efficiency = analysis.get('efficiency_analysis', {})
        print(f"   学习效率: {efficiency.get('overall_efficiency', 0):.2f}")

        # 显示学习洞察
        insights = analysis.get('insights', [])
        if insights:
            print(f"\n💡 学习洞察 ({len(insights)}条):")
            for i, insight in enumerate(insights[:3], 1):
                print(f"   {i}. [{insight.category}] {insight.title}")
                print(f"      {insight.description}")

        # 显示预测
        predictions = analysis.get('predictions', [])
        if predictions:
            print(f"\n🔮 学习预测 ({len(predictions)}个):")
            for i, pred in enumerate(predictions[:2], 1):
                print(f"   {i}. {pred.prediction_type}: {pred.predicted_value}")
                print(f"      置信度: {pred.confidence:.1%}")

        # 显示建议
        recommendations = analysis.get('recommendations', {})
        print(f"\n💭 个性化建议:")
        for category, recs in recommendations.items():
            if recs:
                print(f"   {category}:")
                for rec in recs[:2]:
                    print(f"     • {rec}")

        # 生成可视化
        print(f"\n📊 生成可视化图表...")
        viz_files = analytics.create_visualizations(analysis)

        print("✅ 可视化图表已生成:")
        for chart_type, file_path in viz_files.items():
            print(f"   {chart_type}: {file_path}")

        # 导出分析报告
        report_file = f"learning_analysis_report_{demo_user_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n📄 完整分析报告已保存: {report_file}")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        print("💡 请确保数据库中有足够的学习数据")

    print("\n" + "=" * 70)
    print("🎉 学习分析演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_learning_analytics()
