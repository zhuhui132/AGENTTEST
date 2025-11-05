"""
上下文管理组件
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

class ContextManager:
    """上下文管理器"""

    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.context_cache = {}
        self.priorities = {}

    def build_context(
        self,
        conversation_history: List[Dict],
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """构建上下文"""
        context = {
            "conversation_summary": self._summarize_conversation(conversation_history),
            "key_entities": self._extract_key_entities(conversation_history),
            "user_intent": self._infer_user_intent(conversation_history),
            "context_window": self._build_context_window(conversation_history),
            "additional_info": additional_context or {}
        }

        # 应用优先级
        context = self._apply_priorities(context)

        return context

    def _summarize_conversation(self, history: List[Dict]) -> Dict:
        """总结对话历史"""
        if not history:
            return {
                "total_messages": 0,
                "last_message_time": None,
                "conversation_duration": None,
                "topics_discussed": []
            }

        # 分析对话主题
        topics = self._identify_topics(history)

        # 计算对话时长
        first_time = history[0]["timestamp"]
        last_time = history[-1]["timestamp"]

        # 处理datetime对象和字符串
        if isinstance(first_time, datetime):
            first_dt = first_time
        else:
            first_dt = datetime.fromisoformat(first_time)

        if isinstance(last_time, datetime):
            last_dt = last_time
        else:
            last_dt = datetime.fromisoformat(last_time)

        duration = last_dt - first_dt

        return {
            "total_messages": len(history),
            "last_message_time": history[-1]["timestamp"],
            "conversation_duration": str(duration),
            "topics_discussed": topics,
            "user_messages": sum(1 for msg in history if msg["role"] == "user"),
            "assistant_messages": sum(1 for msg in history if msg["role"] == "assistant")
        }

    def _extract_key_entities(self, history: List[Dict]) -> List[Dict]:
        """提取关键实体"""
        entities = []
        seen_entities = set()

        for message in history[-10:]:  # 只分析最近10条消息
            content = message["content"].lower()

            # 简单的实体识别（实际应用中应使用NER）
            if "北京" in content or "上海" in content or "广州" in content:
                for city in ["北京", "上海", "广州"]:
                    if city in content and city not in seen_entities:
                        entities.append({
                            "type": "location",
                            "value": city,
                            "context": message["content"],
                            "timestamp": message["timestamp"]
                        })
                        seen_entities.add(city)

            if "天气" in content or "温度" in content:
                entities.append({
                    "type": "topic",
                    "value": "weather",
                    "context": message["content"],
                    "timestamp": message["timestamp"]
                })

        return entities

    def _infer_user_intent(self, history: List[Dict]) -> Dict:
        """推断用户意图"""
        if not history:
            return {"intent": "unknown", "confidence": 0.0}

        # 分析最近几条用户消息
        recent_user_messages = [
            msg for msg in history[-5:]
            if msg["role"] == "user"
        ]

        if not recent_user_messages:
            return {"intent": "unknown", "confidence": 0.0}

        # 简单的意图识别
        intents = {
            "question": ["什么", "如何", "为什么", "怎么"],
            "request": ["请", "帮我", "能否", "可以"],
            "greeting": ["你好", "早上好", "晚上好", "再见"],
            "complaint": ["不满意", "问题", "错误", "失败"]
        }

        intent_scores = {}
        for intent, keywords in intents.items():
            score = 0
            for msg in recent_user_messages:
                content = msg["content"]
                for keyword in keywords:
                    if keyword in content:
                        score += 1

            intent_scores[intent] = score / len(recent_user_messages)

        # 选择最高分的意图
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]

        return {
            "intent": best_intent if confidence > 0.1 else "unknown",
            "confidence": confidence,
            "alternative_intents": [
                {"intent": intent, "confidence": score}
                for intent, score in intent_scores.items()
                if intent != best_intent and score > 0
            ]
        }

    def _build_context_window(self, history: List[Dict]) -> List[Dict]:
        """构建上下文窗口"""
        if not history:
            return []

        # 智能截断：保留重要消息
        context_window = []
        current_length = 0

        # 优先保留最近的消息
        for msg in reversed(history[-20:]):  # 最多保留最近20条
            msg_length = len(msg["content"])

            if current_length + msg_length > self.max_context_length:
                break

            context_window.insert(0, msg)
            current_length += msg_length

        return context_window

    def _identify_topics(self, history: List[Dict]) -> List[str]:
        """识别讨论主题"""
        topics = set()
        topic_keywords = {
            "天气": ["天气", "温度", "下雨", "晴天", "阴天"],
            "计算": ["计算", "加", "减", "乘", "除", "等于"],
            "时间": ["时间", "几点", "日期", "今天", "明天"],
            "问候": ["你好", "再见", "谢谢", "不客气"]
        }

        for message in history:
            content = message["content"].lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    topics.add(topic)

        return list(topics)

    def _apply_priorities(self, context: Dict) -> Dict:
        """应用优先级规则"""
        # 为不同上下文元素分配优先级
        prioritized_context = {
            "high_priority": {
                "user_intent": context["user_intent"],
                "key_entities": context["key_entities"][:5]  # 最多5个关键实体
            },
            "medium_priority": {
                "conversation_summary": {
                    "total_messages": context["conversation_summary"]["total_messages"],
                    "topics_discussed": context["conversation_summary"]["topics_discussed"]
                }
            },
            "low_priority": {
                "context_window": context["context_window"][-5:],  # 最近5条消息
                "additional_info": context["additional_info"]
            }
        }

        return prioritized_context

    def update_context_priority(self, context_key: str, priority: str):
        """更新上下文优先级"""
        self.priorities[context_key] = priority

    def clear_expired_context(self, max_age_hours: int = 24):
        """清理过期上下文"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        expired_keys = []
        for key, context in self.context_cache.items():
            if "created_at" in context:
                created_time = datetime.fromisoformat(context["created_at"])
                if created_time < cutoff_time:
                    expired_keys.append(key)

        for key in expired_keys:
            del self.context_cache[key]
