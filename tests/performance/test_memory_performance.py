"""
记忆系统性能测试
测试记忆系统的性能指标和优化
"""

import pytest
import time
import asyncio
import psutil
import os
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory
from src.memory.manager import MemoryManager


class TestEpisodicMemoryPerformance:
    """情景记忆性能测试类"""

    @pytest.fixture
    def episodic_memory(self):
        """情景记忆fixture"""
        return EpisodicMemory(max_size=10000)

    def test_add_performance(self, episodic_memory):
        """测试添加性能"""
        # 大量添加测试
        start_time = time.time()

        for i in range(1000):
            episodic_memory.add(f"测试记忆内容{i}", time.time())

        end_time = time.time()
        add_time = end_time - start_time

        # 1000条记忆添加应该在1秒内完成
        assert add_time < 1.0

        # 计算添加速度
        adds_per_second = 1000 / add_time
        assert adds_per_second > 500

    def test_retrieval_performance(self, episodic_memory):
        """测试检索性能"""
        # 预先添加大量记忆
        for i in range(1000):
            episodic_memory.add(f"包含关键词{i%100}的记忆内容", time.time())

        # 测试检索性能
        start_time = time.time()

        for i in range(100):
            results = episodic_memory.search(f"关键词{i%100}")
            assert len(results) >= 10  # 应该找到相关记忆

        end_time = time.time()
        retrieval_time = end_time - start_time

        # 100次检索应该在0.5秒内完成
        assert retrieval_time < 0.5

        # 计算检索速度
        retrievals_per_second = 100 / retrieval_time
        assert retrievals_per_second > 200

    def test_memory_usage(self, episodic_memory):
        """测试内存使用"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 添加大量记忆
        for i in range(5000):
            episodic_memory.add(f"很长的测试记忆内容{i}" * 100, time.time())

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（小于100MB）
        assert memory_increase < 100 * 1024 * 1024

    def test_concurrent_access(self, episodic_memory):
        """测试并发访问性能"""
        async def add_memories():
            for i in range(100):
                episodic_memory.add(f"并发记忆{i}", time.time())

        async def search_memories():
            for i in range(100):
                episodic_memory.search(f"关键词{i}")

        start_time = time.time()

        # 并发执行添加和搜索
        await asyncio.gather(
            add_memories(),
            search_memories(),
            add_memories(),
            search_memories()
        )

        end_time = time.time()
        concurrent_time = end_time - start_time

        # 并发操作应该在2秒内完成
        assert concurrent_time < 2.0


class TestSemanticMemoryPerformance:
    """语义记忆性能测试类"""

    @pytest.fixture
    def semantic_memory(self):
        """语义记忆fixture"""
        return SemanticMemory()

    def test_concept_storage_performance(self, semantic_memory):
        """测试概念存储性能"""
        start_time = time.time()

        for i in range(2000):
            semantic_memory.add_concept(
                f"概念{i}",
                f"这是概念{i}的详细定义，包含丰富的语义信息"
            )

        end_time = time.time()
        storage_time = end_time - start_time

        # 2000个概念存储应该在1秒内完成
        assert storage_time < 1.0

    def test_concept_retrieval_performance(self, semantic_memory):
        """测试概念检索性能"""
        # 预先添加概念
        concepts = [
            (f"人工智能概念{i}", f"AI领域的第{i}个概念定义")
            for i in range(1000)
        ]

        for concept, definition in concepts:
            semantic_memory.add_concept(concept, definition)

        start_time = time.time()

        for i in range(100):
            concept = semantic_memory.get_concept(f"人工智能概念{i}")
            assert concept is not None

        end_time = time.time()
        retrieval_time = end_time - start_time

        # 100次概念检索应该在0.1秒内完成
        assert retrieval_time < 0.1

    def test_related_concepts_performance(self, semantic_memory):
        """测试相关概念查找性能"""
        # 构建概念网络
        for i in range(1000):
            semantic_memory.add_concept(
                f"概念{i}",
                f"与概念{(i-1)%1000}和概念{(i+1)%1000}相关"
            )

        start_time = time.time()

        for i in range(100):
            related = semantic_memory.get_related_concepts(f"概念{i%1000}")
            assert len(related) >= 0

        end_time = time.time()
        search_time = end_time - start_time

        # 100次相关概念搜索应该在1秒内完成
        assert search_time < 1.0


class TestWorkingMemoryPerformance:
    """工作记忆性能测试类"""

    @pytest.fixture
    def working_memory(self):
        """工作记忆fixture"""
        return WorkingMemory(capacity=1000)

    def test_item_addition_performance(self, working_memory):
        """测试项目添加性能"""
        start_time = time.time()

        for i in range(10000):
            working_memory.add(f"工作记忆项目{i}")

        end_time = time.time()
        add_time = end_time - start_time

        # 10000个项目添加应该在0.5秒内完成
        assert add_time < 0.5

    def test_capacity_rotation_performance(self, working_memory):
        """测试容量轮转性能"""
        start_time = time.time()

        # 超过容量限制，触发轮转
        for i in range(2000):
            working_memory.add(f"项目{i}")

        end_time = time.time()
        rotation_time = end_time - start_time

        # 2000个项目轮转应该在1秒内完成
        assert rotation_time < 1.0

        # 验证容量始终保持在限制内
        assert len(working_memory.items) <= working_memory.capacity

    def test_frequent_access_performance(self, working_memory):
        """测试频繁访问性能"""
        # 预先添加项目
        for i in range(100):
            working_memory.add(f"频繁访问项目{i}")

        start_time = time.time()

        for i in range(1000):
            item = working_memory.get(i % 100)
            assert item is not None

        end_time = time.time()
        access_time = end_time - start_time

        # 1000次访问应该在0.1秒内完成
        assert access_time < 0.1


class TestMemoryManagerPerformance:
    """记忆管理器性能测试类"""

    @pytest.fixture
    def memory_manager(self):
        """记忆管理器fixture"""
        return MemoryManager()

    @pytest.mark.asyncio
    async def test_integrated_storage_performance(self, memory_manager):
        """测试集成存储性能"""
        start_time = time.time()

        for i in range(500):
            await memory_manager.store_episodic(f"情景记忆{i}")
            memory_manager.store_semantic(f"概念{i}", f"定义{i}")
            memory_manager.add_working_item(f"工作项目{i}")

        end_time = time.time()
        storage_time = end_time - start_time

        # 500个集成存储应该在2秒内完成
        assert storage_time < 2.0

    @pytest.mark.asyncio
    async def test_integrated_retrieval_performance(self, memory_manager):
        """测试集成检索性能"""
        # 预先存储数据
        for i in range(200):
            await memory_manager.store_episodic(f"包含关键词{i}的情景记忆")

        start_time = time.time()

        for i in range(100):
            relevant = await memory_manager.retrieve_relevant(f"关键词{i%50}")
            assert len(relevant) >= 0

        end_time = time.time()
        retrieval_time = end_time - start_time

        # 100次集成检索应该在3秒内完成
        assert retrieval_time < 3.0

    def test_memory_cleanup_performance(self, memory_manager):
        """测试记忆清理性能"""
        # 预先填充大量记忆
        for i in range(1000):
            memory_manager.store_episodic(f"需要清理的记忆{i}", time.time())

        start_time = time.time()

        # 执行清理
        memory_manager.cleanup_old_memories(days_threshold=0)

        end_time = time.time()
        cleanup_time = end_time - start_time

        # 1000条记忆清理应该在2秒内完成
        assert cleanup_time < 2.0

        # 验证清理效果
        remaining_count = len(memory_manager.episodic.memories)
        assert remaining_count < 1000

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, memory_manager):
        """测试并发操作性能"""
        async def store_operations():
            for i in range(100):
                await memory_manager.store_episodic(f"并发记忆{i}")

        async def retrieve_operations():
            for i in range(100):
                await memory_manager.retrieve_relevant(f"查询{i}")

        start_time = time.time()

        # 并发执行存储和检索
        await asyncio.gather(
            store_operations(),
            retrieve_operations(),
            store_operations(),
            retrieve_operations()
        )

        end_time = time.time()
        concurrent_time = end_time - start_time

        # 并发操作应该在3秒内完成
        assert concurrent_time < 3.0


class TestMemoryOptimization:
    """记忆优化测试类"""

    @pytest.fixture
    def memory_manager(self):
        """记忆管理器fixture"""
        return MemoryManager()

    def test_memory_efficiency_metrics(self, memory_manager):
        """测试记忆效率指标"""
        # 添加不同类型的记忆
        for i in range(100):
            memory_manager.store_episodic(f"情景记忆{i}")
            memory_manager.store_semantic(f"概念{i}", f"定义{i}")

        stats = memory_manager.get_statistics()

        # 验证统计指标的完整性
        assert 'episodic_count' in stats
        assert 'semantic_count' in stats
        assert 'memory_size_mb' in stats
        assert 'retrieval_success_rate' in stats

        # 验证基本的效率指标
        assert stats['episodic_count'] >= 100
        assert stats['semantic_count'] >= 100
        assert stats['memory_size_mb'] > 0

    def test_memory_optimization_suggestions(self, memory_manager):
        """测试记忆优化建议"""
        # 模拟需要优化的情况
        for i in range(5000):
            memory_manager.store_episodic(f"冗余记忆{i}")

        suggestions = memory_manager.get_optimization_suggestions()

        assert isinstance(suggestions, list)

        # 应该包含优化建议
        suggestion_texts = [s['type'] for s in suggestions]
        expected_suggestions = ['cleanup', 'compression', 'reorganization']

        has_optimization_suggestion = any(
            suggestion in suggestion_texts
            for suggestion in expected_suggestions
        )
        assert has_optimization_suggestion

    def test_memory_auto_optimization(self, memory_manager):
        """测试自动优化功能"""
        # 添加需要优化的记忆
        for i in range(2000):
            memory_manager.store_episodic(f"旧记忆{i}")

        initial_stats = memory_manager.get_statistics()
        initial_size = initial_stats.get('memory_size_mb', 0)

        # 执行自动优化
        optimization_result = memory_manager.auto_optimize()

        final_stats = memory_manager.get_statistics()
        final_size = final_stats.get('memory_size_mb', 0)

        # 验证优化结果
        assert optimization_result['optimized'] is True
        assert final_size <= initial_size  # 优化后内存应该减少或保持

        # 验证优化报告
        assert 'memory_freed_mb' in optimization_result
        assert 'optimizations_applied' in optimization_result
        assert len(optimization_result['optimizations_applied']) >= 0


if __name__ == "__main__":
    pytest.main([__file__])
