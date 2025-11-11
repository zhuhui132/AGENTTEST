"""记忆系统组件 - 异步实现"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.exceptions import (
    MemoryError,
    MemoryFullError,
    MemoryNotFoundError,
)
from ..core.interfaces import BaseMemory
from ..core.types import MemoryConfig, MemoryItem, MemoryType


class MemorySystem(BaseMemory):
    """基于内存的数据存储实现，符合 BaseMemory 接口"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._store: Dict[str, MemoryItem] = {}
        self._index_by_type: Dict[MemoryType, set[str]] = defaultdict(set)
        self._metadata_index: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._lock = asyncio.Lock()

    async def add_memory(
        self,
        content: str,
        importance: float | None = None,
        metadata: Optional[Dict[str, object]] = None,
        *,
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> str:
        if not content or not content.strip():
            raise MemoryError("记忆内容不能为空")

        async with self._lock:
            if len(self._store) >= self.config.max_memories:
                raise MemoryFullError("记忆容量已满，请清理或调整配置")

            normalized_importance = self._normalize_importance(importance)
            item = MemoryItem(
                content=content.strip(),
                memory_type=memory_type,
                importance=normalized_importance,
                metadata=metadata or {},
            )

            self._store[item.id] = item
            self._index_by_type[memory_type].add(item.id)
            self._index_metadata(item)
            return item.id

    async def bulk_add(
        self,
        memories: Sequence[Tuple[str, Optional[float], Optional[Dict[str, object]], MemoryType]],
    ) -> List[str]:
        """批量添加记忆，返回创建的记忆 ID 列表"""
        created_ids: List[str] = []
        for content, importance, metadata, memory_type in memories:
            created_ids.append(
                await self.add_memory(
                    content,
                    importance=importance,
                    metadata=metadata,
                    memory_type=memory_type,
                )
            )
        return created_ids

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        memory_type: Optional[str] = None,
    ) -> List[MemoryItem]:
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        retrieval_limit = limit or self.config.retrieval_limit

        async with self._lock:
            if memory_type:
                try:
                    expected_type = MemoryType(memory_type)
                except ValueError as exc:  # noqa: B904
                    raise MemoryError(f"未知的记忆类型: {memory_type}") from exc
                candidate_ids = list(self._index_by_type.get(expected_type, []))
                candidates = [self._store[mid] for mid in candidate_ids]
            else:
                candidates = list(self._store.values())

        scored_items: List[Tuple[float, MemoryItem]] = []
        now = datetime.now()
        for item in candidates:
            relevance = self._compute_relevance(item, query_lower)
            if relevance < self.config.similarity_threshold:
                continue
            recency = self._compute_recency(item, now)
            score = (
                relevance * 0.55
                + self._importance_with_decay(item) * 0.3
                + recency * 0.15
            )
            scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        selected_items = [item for _, item in scored_items[: max(1, retrieval_limit)]]

        # 更新访问统计
        async with self._lock:
            for item in selected_items:
                item.access_count += 1
                item.last_accessed = now
                item.importance = max(
                    0.0,
                    min(
                        item.importance * self.config.importance_decay_rate,
                        10.0,
                    ),
                )

        return selected_items

    async def retrieve_by_metadata(
        self,
        key: str,
        value: str,
        *,
        limit: int | None = None,
    ) -> List[MemoryItem]:
        """按元数据键值检索记忆"""
        if not key:
            raise MemoryError("metadata key 不能为空")

        async with self._lock:
            ids = list(self._metadata_index.get(key, {}).get(str(value), []))
            candidates = [self._store[mid] for mid in ids]

        retrieval_limit = limit or self.config.retrieval_limit
        return candidates[:retrieval_limit]

    async def update_memory(self, memory_id: str, updates: Dict[str, object]) -> bool:
        async with self._lock:
            if memory_id not in self._store:
                raise MemoryNotFoundError(f"记忆 {memory_id} 不存在")

            item = self._store[memory_id]
            old_metadata = dict(item.metadata)
            original_type = item.memory_type
            if "content" in updates:
                content = str(updates["content"]).strip()
                if not content:
                    raise MemoryError("记忆内容不能为空")
                item.content = content
            if "importance" in updates:
                item.importance = self._normalize_importance(updates["importance"])
            if "metadata" in updates and isinstance(updates["metadata"], dict):
                item.metadata.update(updates["metadata"])
            if "memory_type" in updates:
                new_type = MemoryType(str(updates["memory_type"]))
                if new_type != original_type:
                    self._index_by_type[original_type].discard(memory_id)
                    self._index_by_type[new_type].add(memory_id)
                    item.memory_type = new_type

            item.last_accessed = datetime.now()
            self._remove_from_metadata_index(memory_id, old_metadata)
            self._index_metadata(item)

        return True

    async def delete_memory(self, memory_id: str) -> bool:
        async with self._lock:
            if memory_id not in self._store:
                raise MemoryNotFoundError(f"记忆 {memory_id} 不存在")
            item = self._store.pop(memory_id)
            self._index_by_type[item.memory_type].discard(memory_id)
            self._remove_from_metadata_index(memory_id, item.metadata)
        return True

    async def cleanup(self) -> int:
        """根据配置中的过期时间和重要性阈值清理记忆"""
        expire_after = timedelta(seconds=self.config.cleanup_interval)
        cutoff = datetime.now() - expire_after

        async with self._lock:
            expired_ids = [
                memory_id
                for memory_id, item in self._store.items()
                if item.last_accessed < cutoff or item.importance < 0.1
            ]
            for memory_id in expired_ids:
                item = self._store.pop(memory_id, None)
                if not item:
                    continue
                self._index_by_type[item.memory_type].discard(memory_id)
                self._remove_from_metadata_index(memory_id, item.metadata)
        return len(expired_ids)

    async def stats(self) -> Dict[str, object]:
        async with self._lock:
            total = len(self._store)
            importance_values = [item.importance for item in self._store.values()]
            per_type = {
                mem_type.value: len(ids) for mem_type, ids in self._index_by_type.items()
            }

        avg_importance = (
            sum(importance_values) / len(importance_values) if importance_values else 0.0
        )
        return {
            "total_memories": total,
            "max_memories": self.config.max_memories,
            "average_importance": round(avg_importance, 3),
            "per_type": per_type,
        }

    async def export_memories(self) -> List[MemoryItem]:
        """导出所有记忆，主要用于调试或持久化"""
        async with self._lock:
            return list(self._store.values())

    def _normalize_importance(self, importance: float | None) -> float:
        value = (
            self.config.default_importance
            if importance is None
            else max(0.0, float(importance))
        )
        return min(max(value, 0.0), 10.0)

    def _importance_with_decay(self, item: MemoryItem) -> float:
        # 访问次数越多代表价值越大，稍微提升
        return min(item.importance * (1 + item.access_count * 0.05), 10.0)

    def _index_metadata(self, item: MemoryItem) -> None:
        for key, value in (item.metadata or {}).items():
            self._metadata_index[str(key)][str(value)].add(item.id)

    def _remove_from_metadata_index(self, memory_id: str, metadata: Dict[str, object]) -> None:
        for key, value in metadata.items():
            bucket = self._metadata_index.get(str(key))
            if not bucket:
                continue
            ids = bucket.get(str(value))
            if not ids:
                continue
            ids.discard(memory_id)
            if not ids:
                bucket.pop(str(value), None)
        # 清理空bucket
        empty_keys = [key for key, bucket in self._metadata_index.items() if not bucket]
        for key in empty_keys:
            self._metadata_index.pop(key, None)

    @staticmethod
    def _compute_relevance(memory: MemoryItem, query_lower: str) -> float:
        content = memory.content.lower()
        if query_lower in content:
            return 1.0
        query_terms = set(query_lower.split())
        content_terms = set(content.split())
        if not query_terms or not content_terms:
            return 0.0
        overlap = len(query_terms & content_terms)
        return overlap / len(query_terms)

    @staticmethod
    def _compute_recency(memory: MemoryItem, now: datetime) -> float:
        days = (now - memory.last_accessed).total_seconds() / 86400
        return math.exp(-max(days, 0) / 7)


__all__ = ["MemorySystem"]
