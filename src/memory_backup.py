"""
记忆系统组件
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import json

class MemorySystem:
    """记忆系统管理类"""

    def __init__(self):
        self.memories = {}  # 记忆存储
        self.weights = {}   # 记忆权重
        self.timestamps = {}  # 记忆时间戳
        self.max_memories = 10000  # 最大记忆数量

    def add_memory(self, content: str, weight: float = 1.0, metadata: Optional[Dict] = None) -> str:
        """添加记忆"""
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")

        memory_id = str(uuid.uuid4())
        now = datetime.now()

        self.memories[memory_id] = {
            "content": content.strip(),
            "metadata": metadata or {},
            "created_at": now.isoformat()
        }

        self.weights[memory_id] = max(0.1, min(10.0, weight))
        self.timestamps[memory_id] = now

        # 检查内存限制
        self._cleanup_old_memories()

        return memory_id

    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """检索相关记忆"""
        if not query or not query.strip():
            return []

        # 简单的关键词匹配和相关性计算
        scored_memories = []
        query_words = set(query.lower().split())

        for memory_id, memory in self.memories.items():
            content_words = set(memory["content"].lower().split())

            # 计算相关性分数
            common_words = query_words.intersection(content_words)
            relevance_score = len(common_words) / len(query_words) if query_words else 0

            # 应用时间衰减
            time_factor = self._calculate_time_factor(memory_id)

            # 应用权重
            weight_factor = self.weights[memory_id]

            # 综合分数
            final_score = relevance_score * time_factor * weight_factor

            if final_score > 0.1:  # 最低阈值
                scored_memories.append({
                    "memory_id": memory_id,
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "score": final_score,
                    "created_at": memory["created_at"]
                })

        # 按分数排序并返回前N个
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return scored_memories[:limit]

    def update_memory(self, memory_id: str, content: Optional[str] = None, weight: Optional[float] = None) -> bool:
        """更新记忆"""
        if memory_id not in self.memories:
            return False

        if content:
            if not content or not content.strip():
                raise ValueError("记忆内容不能为空")
            self.memories[memory_id]["content"] = content.strip()
            self.memories[memory_id]["updated_at"] = datetime.now().isoformat()

        if weight is not None:
            self.weights[memory_id] = max(0.1, min(10.0, weight))

        return True

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id not in self.memories:
            return False

        del self.memories[memory_id]
        del self.weights[memory_id]
        del self.timestamps[memory_id]

        return True

    def _calculate_time_factor(self, memory_id: str) -> float:
        """计算时间衰减因子"""
        if memory_id not in self.timestamps:
            return 1.0

        memory_age = datetime.now() - self.timestamps[memory_id]
        days_old = memory_age.days

        # 线性衰减，30天后衰减到0.1
        if days_old >= 30:
            return 0.1
        else:
            return 1.0 - (days_old / 30) * 0.9

    def _cleanup_old_memories(self):
        """清理旧记忆，保持内存限制"""
        if len(self.memories) <= self.max_memories:
            return

        # 按分数和时间综合排序，删除最不重要的记忆
        scored_memories = []
        for memory_id in self.memories.keys():
            score = self.weights[memory_id] * self._calculate_time_factor(memory_id)
            scored_memories.append((memory_id, score))

        scored_memories.sort(key=lambda x: x[1])

        # 删除最不重要的记忆
        to_delete = len(self.memories) - self.max_memories + 100  # 多删除一些留余量
        for memory_id, _ in scored_memories[:to_delete]:
            self.delete_memory(memory_id)

    def get_memory_stats(self) -> Dict:
        """获取记忆系统统计信息"""
        return {
            "total_memories": len(self.memories),
            "max_memories": self.max_memories,
            "average_weight": sum(self.weights.values()) / len(self.weights) if self.weights else 0,
            "oldest_memory": min(self.timestamps.values()).isoformat() if self.timestamps else None,
            "newest_memory": max(self.timestamps.values()).isoformat() if self.timestamps else None
        }
