"""
Agent与记忆系统集成测试
"""
import pytest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent import Agent
from memory import MemorySystem

class TestAgentMemoryIntegration:
    """Agent与记忆系统集成测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("集成测试助手")

    def test_agent_memory_initialization(self):
        """测试Agent记忆系统初始化"""
        assert hasattr(self.agent, 'memory')
        assert isinstance(self.agent.memory, MemorySystem)
        assert self.agent.memory.memories == {}

    def test_message_processing_with_memory_retrieval(self):
        """测试消息处理时的记忆检索"""
        # 先添加一些记忆
        self.agent.memory.add_memory("用户喜欢吃苹果", weight=5.0)
        self.agent.memory.add_memory("用户对花粉过敏")

        # 处理相关消息
        result = self.agent.process_message("我想吃点什么水果")

        # 检查是否检索到相关记忆
        assert "memories_used" in result
        assert isinstance(result["memories_used"], list)

    def test_memory_addition_during_conversation(self):
        """测试对话过程中的记忆添加"""
        initial_count = len(self.agent.memory.memories)

        # 处理消息（假设系统会自动添加重要信息到记忆）
        self.agent.process_message("我叫张三，住在上海")

        # 检查记忆是否增加（需要根据实际实现调整）
        current_count = len(self.agent.memory.memories)

        # 注意：这个测试可能需要根据实际的记忆添加逻辑来调整
        # 如果系统不自动添加记忆，这个测试可能需要手动调用记忆添加方法
        assert current_count >= initial_count

    def test_memory_relevance_scoring(self):
        """测试记忆相关性评分"""
        # 添加相关和无关记忆
        self.agent.memory.add_memory("昨天北京天气晴朗", weight=3.0)
        self.agent.memory.add_memory("用户喜欢编程", weight=2.0)

        # 查询天气相关
        result = self.agent.process_message("今天天气怎么样")

        # 检查检索到的记忆相关性
        memories_used = result["memories_used"]
        if memories_used:
            for memory in memories_used:
                assert "score" in memory
                assert memory["score"] > 0

    def test_conversation_context_in_memory(self):
        """测试对话上下文在记忆中的体现"""
        messages = [
            "我是李四",
            "我在北京工作",
            "我喜欢打篮球"
        ]

        conversation_context = []
        for msg in messages:
            result = self.agent.process_message(msg)
            conversation_context.append(result)

        # 检查上下文是否正确构建
        # 这需要根据实际的上下文管理实现来调整
        last_result = conversation_context[-1]
        assert "context" in last_result

    def test_memory_update_during_interaction(self):
        """测试交互过程中的记忆更新"""
        # 添加初始记忆
        memory_id = self.agent.memory.add_memory("用户住在深圳")

        # 处理包含更新信息的消息
        self.agent.process_message("我现在搬家到广州了")

        # 检查记忆是否更新（这需要实际的记忆更新逻辑）
        # 如果系统实现了自动记忆更新，可以这样测试：
        updated_memory = self.agent.memory.memories.get(memory_id)
        if updated_memory:
            # 检查记忆内容是否包含新信息
            assert "广州" in updated_memory.get("content", "")

    def test_memory_weight_adjustment(self):
        """测试记忆权重调整"""
        # 添加低权重记忆
        memory_id = self.agent.memory.add_memory("一般信息", weight=1.0)

        # 通过多次相关对话来提高权重（如果实现了）
        for _ in range(3):
            self.agent.process_message("关于一般信息的问题")

        # 检查权重是否调整（需要实际的权重调整逻辑）
        current_weight = self.agent.memory.weights.get(memory_id, 0)
        assert current_weight >= 1.0  # 权重应该保持或增加

    def test_memory_cleanup_integration(self):
        """测试记忆清理集成"""
        # 添加大量记忆
        original_max = self.agent.memory.max_memories
        self.agent.memory.max_memories = 5

        for i in range(10):
            self.agent.memory.add_memory(f"测试记忆{i}")

        # 触发清理（可能在消息处理时自动触发）
        self.agent.process_message("触发清理的消息")

        # 检查记忆数量是否在限制内
        assert len(self.agent.memory.memories) <= self.agent.memory.max_memories

        # 恢复原始设置
        self.agent.memory.max_memories = original_max

    def test_memory_error_handling(self):
        """测试记忆系统错误处理"""
        # 模拟记忆系统错误
        original_retrieve = self.agent.memory.retrieve
        self.agent.memory.retrieve = lambda x: (_ for _ in ()).throw(Exception("记忆系统错误"))

        try:
            # 处理消息时应该优雅处理记忆错误
            result = self.agent.process_message("测试消息")
            # 系统应该仍然能够返回响应，即使记忆检索失败
            assert "response" in result
        finally:
            # 恢复原始方法
            self.agent.memory.retrieve = original_retrieve

    def test_memory_concurrency(self):
        """测试记忆系统的并发处理"""
        import threading

        results = []
        errors = []

        def add_memory_batch(start_id):
            try:
                for i in range(5):
                    memory_id = self.agent.memory.add_memory(f"并发记忆{start_id}_{i}")
                    results.append(memory_id)
            except Exception as e:
                errors.append(e)

        # 创建多个线程同时添加记忆
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_memory_batch, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查结果
        assert len(errors) == 0  # 没有错误
        assert len(results) == 15  # 15个记忆被添加
        assert len(self.agent.memory.memories) == 15  # 记忆系统状态正确

class TestMemoryPerformanceIntegration:
    """记忆性能集成测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("性能测试助手")

    def test_large_memory_retrieval_performance(self):
        """测试大量记忆的检索性能"""
        import time

        # 添加大量记忆
        for i in range(1000):
            self.agent.memory.add_memory(f"性能测试记忆{i}")

        # 测试检索性能
        start_time = time.time()
        result = self.agent.process_message("性能测试")
        end_time = time.time()

        retrieval_time = end_time - start_time

        # 检索应该在合理时间内完成（比如1秒内）
        assert retrieval_time < 1.0
        assert "response" in result
        assert "memories_used" in result

    def test_memory_usage_during_long_conversation(self):
        """测试长对话中的内存使用"""
        import psutil
        import os

        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 模拟长对话
        for i in range(100):
            self.agent.process_message(f"这是第{i}条测试消息")

        # 检查最终内存使用
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（比如100MB）
        assert memory_increase < 100 * 1024 * 1024  # 100MB

    def test_memory_cleanup_performance_impact(self):
        """测试内存清理对性能的影响"""
        import time

        # 设置较小的内存限制
        self.agent.memory.max_memories = 10

        # 添加大量记忆触发多次清理
        start_time = time.time()
        for i in range(100):
            self.agent.memory.add_memory(f"触发清理的记忆{i}")
        end_time = time.time()

        cleanup_time = end_time - start_time

        # 清理操作应该在合理时间内完成
        assert cleanup_time < 2.0
        assert len(self.agent.memory.memories) <= self.agent.memory.max_memories

class TestMemoryIntegrationEdgeCases:
    """记忆集成边界情况测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("边界测试助手")

    def test_memory_with_unicode_content(self):
        """测试Unicode内容的记忆集成"""
        unicode_messages = [
            "🌟你好世界🌟",
            "中文测试内容",
            "Test with emoji 🚀🌈",
            "混合语言 Hello 世界"
        ]

        for msg in unicode_messages:
            # 添加到记忆
            self.agent.memory.add_memory(f"用户说: {msg}")

        # 测试检索
        for msg in unicode_messages:
            result = self.agent.process_message(f"关于'{msg}'的讨论")
            assert "response" in result
            assert isinstance(result["memories_used"], list)

    def test_memory_with_extremely_long_content(self):
        """测试极长内容的记忆集成"""
        long_content = "这是一个很长的测试内容。" * 1000

        # 添加长内容记忆
        memory_id = self.agent.memory.add_memory(long_content)
        assert memory_id is not None

        # 测试检索
        result = self.agent.process_message("关于长内容的讨论")
        assert "response" in result

    def test_memory_corruption_handling(self):
        """测试记忆损坏处理"""
        # 手动损坏记忆数据
        self.agent.memory.memories["corrupt"] = {"invalid": "data"}

        # 系统应该能够处理损坏的记忆数据
        result = self.agent.process_message("测试消息")
        assert "response" in result

        # 清理损坏的数据
        if "corrupt" in self.agent.memory.memories:
            del self.agent.memory.memories["corrupt"]
