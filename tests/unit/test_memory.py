"""
记忆系统单元测试
"""
import pytest
import time
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memory import MemorySystem

class TestMemorySystemInitialization:
    """记忆系统初始化测试"""

    def test_memory_system_creation(self):
        """测试记忆系统创建"""
        memory = MemorySystem()
        assert memory.memories == {}
        assert memory.weights == {}
        assert memory.timestamps == {}
        assert memory.max_memories == 10000

    def test_memory_system_custom_max_memories(self):
        """测试自定义最大记忆数量"""
        memory = MemorySystem()
        memory.max_memories = 100
        assert memory.max_memories == 100

class TestMemoryAddition:
    """记忆添加测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()

    def test_add_valid_memory(self):
        """测试添加有效记忆"""
        content = "这是一个测试记忆"
        memory_id = self.memory.add_memory(content)

        assert memory_id is not None
        assert memory_id in self.memory.memories
        assert self.memory.memories[memory_id]["content"] == content
        assert memory_id in self.memory.weights
        assert memory_id in self.memory.timestamps

    def test_add_memory_with_weight(self):
        """测试添加带权重的记忆"""
        content = "重要记忆"
        weight = 5.0
        memory_id = self.memory.add_memory(content, weight)

        assert self.memory.weights[memory_id] == weight

    def test_add_memory_with_metadata(self):
        """测试添加带元数据的记忆"""
        content = "带元数据的记忆"
        metadata = {"source": "user", "importance": "high"}
        memory_id = self.memory.add_memory(content, metadata=metadata)

        assert self.memory.memories[memory_id]["metadata"] == metadata

    def test_add_memory_boundary_weight(self):
        """测试边界权重值"""
        content = "测试记忆"

        # 测试最小权重
        memory_id1 = self.memory.add_memory(content, weight=0.0)
        assert self.memory.weights[memory_id1] == 0.1

        # 测试最大权重
        memory_id2 = self.memory.add_memory(content, weight=15.0)
        assert self.memory.weights[memory_id2] == 10.0

    def test_add_empty_memory(self):
        """测试添加空记忆抛出异常"""
        with pytest.raises(ValueError, match="记忆内容不能为空"):
            self.memory.add_memory("")

    def test_add_whitespace_memory(self):
        """测试添加空白记忆抛出异常"""
        with pytest.raises(ValueError, match="记忆内容不能为空"):
            self.memory.add_memory("   ")

    def test_add_memory_auto_trims_content(self):
        """测试记忆内容自动去除空格"""
        content = "  测试记忆  "
        memory_id = self.memory.add_memory(content)

        assert self.memory.memories[memory_id]["content"] == "测试记忆"

    def test_add_multiple_memories(self):
        """测试添加多个记忆"""
        contents = ["记忆1", "记忆2", "记忆3"]
        memory_ids = []

        for content in contents:
            memory_id = self.memory.add_memory(content)
            memory_ids.append(memory_id)

        # 检查所有记忆都被正确添加
        for i, memory_id in enumerate(memory_ids):
            assert memory_id in self.memory.memories
            assert self.memory.memories[memory_id]["content"] == contents[i]

class TestMemoryRetrieval:
    """记忆检索测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()

        # 添加测试记忆
        self.test_memories = [
            "北京的天气很好",
            "上海今天下雨",
            "广州天气炎热",
            "深圳天气晴朗"
        ]

        self.memory_ids = []
        for content in self.test_memories:
            memory_id = self.memory.add_memory(content)
            self.memory_ids.append(memory_id)

    def test_retrieve_with_exact_match(self):
        """测试精确匹配检索"""
        results = self.memory.retrieve("北京")
        assert len(results) >= 1
        assert any("北京" in result["content"] for result in results)

    def test_retrieve_with_partial_match(self):
        """测试部分匹配检索"""
        results = self.memory.retrieve("天气")
        assert len(results) >= 1
        # 应该返回多个包含"天气"的记忆

    def test_retrieve_empty_query(self):
        """测试空查询"""
        results = self.memory.retrieve("")
        assert results == []

    def test_retrieve_whitespace_query(self):
        """测试空白查询"""
        results = self.memory.retrieve("   ")
        assert results == []

    def test_retrieve_limit(self):
        """测试检索数量限制"""
        results = self.memory.retrieve("天气", limit=2)
        assert len(results) <= 2

    def test_retrieve_ordering(self):
        """测试检索结果排序"""
        # 添加一个高权重的记忆
        high_weight_content = "北京天气特别重要"
        high_weight_id = self.memory.add_memory(high_weight_content, weight=10.0)

        results = self.memory.retrieve("北京")

        # 高权重记忆应该排在前面
        if len(results) > 1:
            assert any(result["memory_id"] == high_weight_id for result in results[:2])

    def test_retrieve_no_matches(self):
        """测试无匹配结果"""
        results = self.memory.retrieve("不存在的内容")
        assert results == []

    def test_retrieve_score_calculation(self):
        """测试相关性分数计算"""
        results = self.memory.retrieve("北京")

        for result in results:
            assert "score" in result
            assert 0 <= result["score"] <= 1
            assert "memory_id" in result
            assert "content" in result
            assert "created_at" in result

class TestMemoryUpdate:
    """记忆更新测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()
        self.memory_id = self.memory.add_memory("原始记忆")

    def test_update_memory_content(self):
        """测试更新记忆内容"""
        new_content = "更新后的记忆"
        success = self.memory.update_memory(self.memory_id, content=new_content)

        assert success is True
        assert self.memory.memories[self.memory_id]["content"] == new_content
        assert "updated_at" in self.memory.memories[self.memory_id]

    def test_update_memory_weight(self):
        """测试更新记忆权重"""
        new_weight = 7.5
        success = self.memory.update_memory(self.memory_id, weight=new_weight)

        assert success is True
        assert self.memory.weights[self.memory_id] == new_weight

    def test_update_both_content_and_weight(self):
        """测试同时更新内容和权重"""
        new_content = "完全新的记忆"
        new_weight = 8.0

        success = self.memory.update_memory(
            self.memory_id,
            content=new_content,
            weight=new_weight
        )

        assert success is True
        assert self.memory.memories[self.memory_id]["content"] == new_content
        assert self.memory.weights[self.memory_id] == new_weight

    def test_update_nonexistent_memory(self):
        """测试更新不存在的记忆"""
        success = self.memory.update_memory("不存在的ID", content="新内容")
        assert success is False

    def test_update_memory_empty_content(self):
        """测试更新为空内容抛出异常"""
        with pytest.raises(ValueError, match="记忆内容不能为空"):
            self.memory.update_memory(self.memory_id, content="")

class TestMemoryDeletion:
    """记忆删除测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()
        self.memory_id = self.memory.add_memory("待删除的记忆")

    def test_delete_existing_memory(self):
        """测试删除存在的记忆"""
        success = self.memory.delete_memory(self.memory_id)

        assert success is True
        assert self.memory_id not in self.memory.memories
        assert self.memory_id not in self.memory.weights
        assert self.memory_id not in self.memory.timestamps

    def test_delete_nonexistent_memory(self):
        """测试删除不存在的记忆"""
        success = self.memory.delete_memory("不存在的ID")
        assert success is False

    def test_delete_memory_then_retrieve(self):
        """测试删除记忆后无法检索"""
        self.memory.delete_memory(self.memory_id)
        results = self.memory.retrieve("记忆")

        # 删除的记忆不应该出现在检索结果中
        memory_ids = [result["memory_id"] for result in results]
        assert self.memory_id not in memory_ids

class TestTimeDecay:
    """时间衰减测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()

    def test_time_factor_calculation(self):
        """测试时间衰减因子计算"""
        memory_id = self.memory.add_memory("测试记忆")

        # 新记忆的时间因子应该接近1
        time_factor = self.memory._calculate_time_factor(memory_id)
        assert time_factor > 0.9

    def test_old_memory_decay(self):
        """测试旧记忆的时间衰减"""
        memory_id = self.memory.add_memory("旧记忆")

        # 模拟30天前的记忆
        old_time = datetime.now() - timedelta(days=30)
        self.memory.timestamps[memory_id] = old_time

        time_factor = self.memory._calculate_time_factor(memory_id)
        assert time_factor == 0.1

    def test_very_old_memory_decay(self):
        """测试非常旧记忆的时间衰减"""
        memory_id = self.memory.add_memory("非常旧的记忆")

        # 模拟60天前的记忆
        old_time = datetime.now() - timedelta(days=60)
        self.memory.timestamps[memory_id] = old_time

        time_factor = self.memory._calculate_time_factor(memory_id)
        assert time_factor == 0.1  # 最小值

class TestMemoryCleanup:
    """记忆清理测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()
        self.memory.max_memories = 5  # 设置较小的限制用于测试

    def test_cleanup_old_memories(self):
        """测试清理旧记忆"""
        # 添加超过限制的记忆
        memory_ids = []
        for i in range(10):
            memory_id = self.memory.add_memory(f"记忆{i}")
            memory_ids.append(memory_id)

        # 触发清理
        self.memory._cleanup_old_memories()

        # 检查记忆数量不超过限制
        assert len(self.memory.memories) <= self.memory.max_memories

    def test_cleanup_preserves_important_memories(self):
        """测试清理保留重要记忆"""
        # 添加一些记忆，其中一个是高权重的
        for i in range(8):
            weight = 1.0
            if i == 5:  # 第6个记忆设为高权重
                weight = 10.0
            self.memory.add_memory(f"记忆{i}", weight=weight)

        # 触发清理
        self.memory._cleanup_old_memories()

        # 检查高权重记忆仍然存在
        remaining_contents = [mem["content"] for mem in self.memory.memories.values()]
        assert "记忆5" in remaining_contents

class TestMemoryStats:
    """记忆统计测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()

    def test_empty_memory_stats(self):
        """测试空记忆统计"""
        stats = self.memory.get_memory_stats()

        assert stats["total_memories"] == 0
        assert stats["max_memories"] == 10000
        assert stats["average_weight"] == 0
        assert stats["oldest_memory"] is None
        assert stats["newest_memory"] is None

    def test_memory_stats_with_data(self):
        """测试有数据的记忆统计"""
        # 添加一些记忆
        weights = [1.0, 2.0, 3.0]
        for weight in weights:
            self.memory.add_memory("测试记忆", weight=weight)

        stats = self.memory.get_memory_stats()

        assert stats["total_memories"] == 3
        assert stats["average_weight"] == sum(weights) / len(weights)
        assert stats["oldest_memory"] is not None
        assert stats["newest_memory"] is not None

    def test_memory_stats_single_memory(self):
        """测试单个记忆的统计"""
        self.memory.add_memory("单个记忆")
        stats = self.memory.get_memory_stats()

        assert stats["total_memories"] == 1
        assert stats["average_weight"] == 1.0  # 默认权重

class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        """测试前置设置"""
        self.memory = MemorySystem()

    def test_unicode_memory_content(self):
        """测试Unicode记忆内容"""
        unicode_content = "🌟测试记忆🌟"
        memory_id = self.memory.add_memory(unicode_content)

        assert self.memory.memories[memory_id]["content"] == unicode_content

    def test_special_characters_in_memory(self):
        """测试记忆中的特殊字符"""
        special_content = "测试<>{}[]|\\\"'`~!@#$%^&*()_+-="
        memory_id = self.memory.add_memory(special_content)

        assert self.memory.memories[memory_id]["content"] == special_content

    def test_very_long_memory_content(self):
        """测试很长的记忆内容"""
        long_content = "测试" * 1000  # 4000字符
        memory_id = self.memory.add_memory(long_content)

        assert self.memory.memories[memory_id]["content"] == long_content

    def test_retrieval_with_unicode_query(self):
        """测试Unicode查询"""
        self.memory.add_memory("测试中文记忆")
        results = self.memory.retrieve("中文")

        assert len(results) >= 1
        assert "中文" in results[0]["content"]
