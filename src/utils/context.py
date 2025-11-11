"""
上下文管理组件
提供与 BaseContextManager 接口兼容的实现，用于构建和维护会话上下文。
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta
import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ..core.exceptions import ContextError
from ..core.interfaces import BaseContextManager
from ..core.types import AgentMessage, ContextInfo

logger = logging.getLogger(__name__)


class ContextManager(BaseContextManager):
    """异步上下文管理器"""

    def __init__(
        self,
        *,
        max_history: int = 15,
        summary_window: int = 6,
        expire_after: timedelta = timedelta(hours=24),
        summarizer: Optional[Callable[[Sequence[AgentMessage]], str]] = None,
        entity_extractors: Optional[Sequence[Callable[[AgentMessage], List[str]]]] = None,
        intent_classifier: Optional[Callable[[AgentMessage], Optional[str]]] = None,
    ):
        self.max_history = max_history
        self.summary_window = summary_window
        self.expire_after = expire_after
        self._summarizer = summarizer
        self._entity_extractors = entity_extractors or ()
        self._intent_classifier = intent_classifier

        self._contexts: Dict[str, ContextInfo] = {}
        self._history: Dict[str, deque[AgentMessage]] = {}
        self._lock = asyncio.Lock()

    async def build_context(
        self,
        messages: List[AgentMessage],
        max_length: int = 4000,
    ) -> ContextInfo:
        """基于给定消息构建新的上下文"""
        truncated = self._truncate_messages(messages, max_length)
        conversation_id = self._resolve_conversation_id(truncated)
        summary = self._summarize(truncated)
        entities = self._extract_entities(truncated)
        intent = self._infer_intent(truncated)

        context = ContextInfo(
            conversation_id=conversation_id,
            context_window=truncated,
            summary=summary,
            key_entities=entities,
            user_intent=intent,
            last_updated=datetime.now(),
        )

        async with self._lock:
            self._contexts[conversation_id] = context
            self._history[conversation_id] = deque(truncated, maxlen=self.max_history)

        return context

    async def update_context(
        self,
        context_id: str,
        message: AgentMessage,
    ) -> bool:
        """向已存在的上下文追加一条消息"""
        async with self._lock:
            if context_id not in self._contexts:
                raise ContextError(f"上下文 {context_id} 不存在")

            history = self._history.setdefault(context_id, deque(maxlen=self.max_history))
            history.append(message)

            updated_list = list(history)
            summary = self._summarize(updated_list)
            entities = self._extract_entities(updated_list)
            intent = self._infer_intent(updated_list)

            context = self._contexts[context_id]
            context.context_window = updated_list
            context.summary = summary
            context.key_entities = entities
            context.user_intent = intent
            context.last_updated = datetime.now()

        return True

    async def get_context(self, context_id: str) -> Optional[ContextInfo]:
        async with self._lock:
            return self._contexts.get(context_id)

    async def get_recent_messages(self, context_id: str, limit: int = 5) -> List[AgentMessage]:
        """获取指定上下文最近的消息"""
        async with self._lock:
            history = self._history.get(context_id)
            if not history:
                return []
            limit = max(1, limit)
            return list(history)[-limit:]

    async def attach_additional_info(self, context_id: str, **info: object) -> None:
        """为上下文附加额外的信息，用于个性化存储"""
        async with self._lock:
            context = self._contexts.get(context_id)
            if not context:
                raise ContextError(f"上下文 {context_id} 不存在")
            context.additional_info.update(info)
            context.last_updated = datetime.now()

    async def cleanup_contexts(self) -> int:
        """清理过期的上下文，返回清理数量"""
        expiry = datetime.now() - self.expire_after

        async with self._lock:
            expired_ids = [
                context_id
                for context_id, info in self._contexts.items()
                if info.last_updated < expiry
            ]

            for context_id in expired_ids:
                self._contexts.pop(context_id, None)
                self._history.pop(context_id, None)

        return len(expired_ids)

    @staticmethod
    def _truncate_messages(messages: Iterable[AgentMessage], max_length: int) -> List[AgentMessage]:
        """对消息进行长度截断，优先保留最新记录"""
        truncated: List[AgentMessage] = []
        total_length = 0

        for message in reversed(list(messages)):
            content_length = len(message.content)
            if total_length + content_length > max_length and truncated:
                break
            truncated.insert(0, message)
            total_length += content_length

        return truncated

    def _resolve_conversation_id(self, messages: Sequence[AgentMessage]) -> str:
        if not messages:
            return ContextInfo().conversation_id

        for message in reversed(messages):
            metadata = message.metadata or {}
            conversation_id = metadata.get("conversation_id") or metadata.get("session_id")
            if isinstance(conversation_id, str) and conversation_id:
                return conversation_id
        return ContextInfo().conversation_id

    def _summarize(self, messages: Iterable[AgentMessage]) -> str:
        """生成简要摘要"""
        if self._summarizer:
            summary = self._summarizer(messages)
            if summary:
                return summary

        recent = list(messages)[-self.summary_window :]
        if not recent:
            return "暂无历史对话摘要。"

        parts = []
        for msg in recent:
            role = "用户" if msg.role == "user" else "助手"
            snippet = msg.content.replace("\n", " ")[:80]
            parts.append(f"{role}: {snippet}")

        return " | ".join(parts)

    def _extract_entities(self, messages: Iterable[AgentMessage]) -> List[str]:
        """提取关键词实体"""
        entities: set[str] = set()
        for message in messages:
            content = message.content
            for extractor in self._entity_extractors:
                try:
                    entities.update(extractor(message))
                except Exception:  # noqa: BLE001
                    logger.exception("实体抽取器执行失败: %s", extractor)
            for keyword in ("天气", "时间", "计划", "任务", "数据"):
                if keyword in content:
                    entities.add(keyword)
        return sorted(entities)

    def _infer_intent(self, messages: Iterable[AgentMessage]) -> str:
        """推断用户意图"""
        for message in reversed(list(messages)):
            if message.role != "user":
                continue
            content = message.content
            if self._intent_classifier:
                try:
                    custom_intent = self._intent_classifier(message)
                except Exception:  # noqa: BLE001
                    logger.exception("意图分类器执行失败")
                    custom_intent = None
                if custom_intent:
                    return custom_intent
            if content.endswith(("吗？", "吗?", "?")) or any(q in content for q in ("什么", "如何", "怎么")):
                return "question"
            if any(term in content for term in ("请", "帮我", "麻烦")):
                return "request"
            if any(term in content for term in ("谢谢", "感谢")):
                return "gratitude"
        return "unknown"


__all__ = ["ContextManager"]
