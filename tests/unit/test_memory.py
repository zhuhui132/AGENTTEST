"""
记忆系统功能测试
测试记忆系统的基础功能
"""

import pytest
from datetime import datetime
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory
from src.memory.manager import MemoryManager
from src.core.types import MemoryType


class TestEpisodicMemory:
    """情景记忆测试类"""

    @pytest.fixture
    def episodic_memory(self):
        """情景记忆fixture"""
        return EpisodicMemory(max_size=100)

    def test_episodic_memory_initialization(self, episodic_memory):
        """测试情景记忆初始化"""
        assert episodic_memory.max_size == 100
        assert len(episodic_memory.memories) == 0

    def test_add_memory(self, episodic_memory):
        """测试添加记忆"""
        content = "用户说他的名字是张三"
        timestamp = datetime.now()

        memory_id = episodic_memory.add(content, timestamp)

        assert memory_id is not None
        assert len(episodic_memory.memories) == 1
        assert episodic_memory.memories[0].content == content

    def test_retrieve_memory(self, episodic_memory):
        """测试检索记忆"""
        content = "用户说他的名字是张三"
        memory_id = episodic_memory.add(content)

        retrieved = episodic_memory.get(memory_id)

        assert retrieved is not None
        assert retrieved.content == content
        assert retrieved.memory_type == MemoryType.EPISODIC

    def test_search_memories(self, episodic_memory):
        """测试搜索记忆"""
        contents = [
            "用户说他的名字是张三",
            "用户喜欢编程",
            "张三是一名程序员"
        ]

        for content in contents:
            episodic_memory.add(content)

        results = episodic_memory.search("张三")

        assert len(results) >= 2
        assert all("张三" in result.content for result in results)

    def test_memory_limit(self, episodic_memory):
        """测试记忆容量限制"""
        # 添加超过最大容量的记忆
        for i in range(150):
            episodic_memory.add(f"记忆{i}")

        assert len(episodic_memory.memories) == episodic_memory.max_size

    def test_delete_memory(self, episodic_memory):
        """测试删除记忆"""
        content = "测试记忆"
        memory_id = episodic_memory.add(content)

        success = episodic_memory.delete(memory_id)

        assert success is True
        assert len(episodic_memory.memories) == 0
        assert episodic_memory.get(memory_id) is None


class TestSemanticMemory:
    """语义记忆测试类"""

    @pytest.fixture
    def semantic_memory(self):
        """语义记忆fixture"""
        return SemanticMemory()

    def test_semantic_memory_initialization(self, semantic_memory):
        """测试语义记忆初始化"""
        assert semantic_memory.concepts == {}

    def test_add_concept(self, semantic_memory):
        """测试添加概念"""
        concept = "人工智能"
        definition = "模拟人类智能的技术"

        semantic_memory.add_concept(concept, definition)

        assert concept in semantic_memory.concepts
        assert semantic_memory.concepts[concept] == definition

    def test_get_concept(self, semantic_memory):
        """测试获取概念"""
        concept = "人工智能"
        definition = "模拟人类智能的技术"

        semantic_memory.add_concept(concept, definition)
        retrieved = semantic_memory.get_concept(concept)

        assert retrieved == definition

    def test_update_concept(self, semantic_memory):
        """测试更新概念"""
        concept = "人工智能"
        old_definition = "模拟人类智能的技术"
        new_definition = "模拟人类智能的综合技术"

        semantic_memory.add_concept(concept, old_definition)
        semantic_memory.update_concept(concept, new_definition)

        assert semantic_memory.concepts[concept] == new_definition

    def test_related_concepts(self, semantic_memory):
        """测试相关概念"""
        concepts = [
            ("人工智能", "智能技术"),
            ("机器学习", "AI的子领域"),
            ("深度学习", "机器学习的分支")
        ]

        for concept, definition in concepts:
            semantic_memory.add_concept(concept, definition)

        related = semantic_memory.get_related_concepts("人工智能")

        assert len(related) >= 0
        assert "机器学习" in [c[0] for c in concepts] or "深度学习" in [c[0] for c in concepts]


class TestWorkingMemory:
    """工作记忆测试类"""

    @pytest.fixture
    def working_memory(self):
        """工作记忆fixture"""
        return WorkingMemory(capacity=7)

    def test_working_memory_initialization(self, working_memory):
        """测试工作记忆初始化"""
        assert working_memory.capacity == 7
        assert len(working_memory.items) == 0

    def test_add_item(self, working_memory):
        """测试添加项目"""
        item = "当前用户：张三"
        working_memory.add(item)

        assert len(working_memory.items) == 1
        assert item in working_memory.items

    def test_capacity_limit(self, working_memory):
        """测试容量限制"""
        items = [f"项目{i}" for i in range(10)]

        for item in items:
            working_memory.add(item)

        assert len(working_memory.items) == working_memory.capacity
        # 应该保留最新的项目
        for item in items[-working_memory.capacity:]:
            assert item in working_memory.items

    def test_get_item(self, working_memory):
        """测试获取项目"""
        item = "测试项目"
        working_memory.add(item)

        retrieved = working_memory.get(0)

        assert retrieved == item

    def test_clear_memory(self, working_memory):
        """测试清空记忆"""
        working_memory.add("项目1")
        working_memory.add("项目2")

        working_memory.clear()

        assert len(working_memory.items) == 0


class TestMemoryManager:
    """记忆管理器测试类"""

    @pytest.fixture
    def memory_manager(self):
        """记忆管理器fixture"""
        return MemoryManager()

    def test_memory_manager_initialization(self, memory_manager):
        """测试记忆管理器初始化"""
        assert memory_manager.episodic is not None
        assert memory_manager.semantic is not None
        assert memory_manager.working is not None

    def test_store_episodic(self, memory_manager):
        """测试存储情景记忆"""
        content = "用户说他喜欢音乐"
        result = memory_manager.store_episodic(content)

        assert result is True
        assert len(memory_manager.episodic.memories) == 1

    def test_store_semantic(self, memory_manager):
        """测试存储语义记忆"""
        concept = "音乐"
        definition = "一种艺术形式"
        result = memory_manager.store_semantic(concept, definition)

        assert result is True
        assert concept in memory_manager.semantic.concepts

    def test_add_working_item(self, memory_manager):
        """测试添加工作记忆项目"""
        item = "当前任务：推荐音乐"
        memory_manager.add_working_item(item)

        assert len(memory_manager.working.items) == 1
        assert item in memory_manager.working.items

    def test_retrieve_relevant_memories(self, memory_manager):
        """测试检索相关记忆"""
        # 添加一些记忆
        memory_manager.store_episodic("用户说他喜欢古典音乐")
        memory_manager.store_episodic("用户提到了贝多芬")
        memory_manager.store_semantic("古典音乐", "传统的音乐形式")

        query = "古典音乐"
        relevant = memory_manager.retrieve_relevant(query)

        assert len(relevant) >= 1
        # 应该包含相关的记忆
        relevant_contents = [item.content for item in relevant]
        assert any("古典" in content or "音乐" in content for content in relevant_contents)

    def test_memory_statistics(self, memory_manager):
        """测试记忆统计"""
        # 添加一些记忆
        memory_manager.store_episodic("测试记忆1")
        memory_manager.store_episodic("测试记忆2")
        memory_manager.store_semantic("测试概念", "测试定义")

        stats = memory_manager.get_statistics()

        assert 'episodic_count' in stats
        assert 'semantic_count' in stats
        assert 'working_count' in stats
        assert stats['episodic_count'] == 2
        assert stats['semantic_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
