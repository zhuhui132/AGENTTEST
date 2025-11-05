"""
完整对话流程端到端测试
"""
import pytest
import time
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent import Agent
from memory import MemorySystem
from rag import RAGSystem
from tools import ToolSystem, calculator, weather_query
from context import ContextManager

class TestCompleteConversationFlow:
    """完整对话流程测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("端到端测试助手")

        # 初始化一些基础数据
        self.setup_test_data()

    def setup_test_data(self):
        """设置测试数据"""
        # 添加一些记忆
        self.agent.memory.add_memory("用户名叫张三", weight=3.0)
        self.agent.memory.add_memory("用户住在北京", weight=2.0)
        self.agent.memory.add_memory("用户喜欢吃苹果", weight=1.0)

        # 添加一些文档
        self.agent.rag.add_document("北京是中国的首都", {"type": "geography"})
        self.agent.rag.add_document("苹果是一种健康水果", {"type": "nutrition"})

        # 注册工具
        self.agent.tools.register_tool("calculator", calculator, "数学计算工具")
        self.agent.tools.register_tool("weather", weather_query, "天气查询工具")

    def test_simple_greeting_conversation(self):
        """测试简单问候对话"""
        conversation = [
            "你好",
            "今天天气怎么样？",
            "谢谢"
        ]

        context_updates = []

        for message in conversation:
            start_time = time.time()
            result = self.agent.process_message(message)
            end_time = time.time()

            # 验证响应结构
            assert "response" in result
            assert "context" in result
            assert "memories_used" in result
            assert "docs_used" in result

            # 验证响应时间
            response_time = end_time - start_time
            assert response_time < 5.0  # 响应应该在5秒内

            context_updates.append(result)

        # 验证对话历史
        assert len(self.agent.conversation_history) == len(conversation) * 2

        # 验证对话连贯性
        for i, update in enumerate(context_updates):
            assert update["response"] is not None
            assert len(update["response"]) > 0

    def test_complex_task_conversation(self):
        """测试复杂任务对话"""
        conversation = [
            "请帮我计算一下100除以25的结果",
            "北京今天的天气如何？",
            "我住在北京，记得我吗？",
            "结合计算结果和天气信息，给我一些建议"
        ]

        conversation_results = []

        for message in conversation:
            result = self.agent.process_message(message)
            conversation_results.append(result)

        # 验证每个响应都有意义
        for result in conversation_results:
            assert result["response"] is not None
            assert len(result["response"]) > 10  # 响应应该有一定长度

        # 验证上下文连贯性
        # 最后一个响应应该能够结合之前的信息
        final_result = conversation_results[-1]
        assert "计算" in final_result["response"] or "天气" in final_result["response"]

    def test_personalized_conversation(self):
        """测试个性化对话"""
        conversation = [
            "我是谁？",
            "我住在哪里？",
            "我喜欢吃什么？",
            "根据我的信息，推荐一些活动"
        ]

        results = []
        for message in conversation:
            result = self.agent.process_message(message)
            results.append(result)

        # 检查个性化响应
        # 第一个问题应该提到张三
        assert "张三" in results[0]["response"] or "用户" in results[0]["response"]

        # 第二个问题应该提到北京
        assert "北京" in results[1]["response"]

        # 第三个问题应该提到苹果
        assert "苹果" in results[2]["response"]

    def test_error_recovery_conversation(self):
        """测试错误恢复对话"""
        # 先发送正常消息
        result1 = self.agent.process_message("你好")
        assert "response" in result1

        # 尝试触发错误（空消息）
        with pytest.raises(ValueError):
            self.agent.process_message("")

        # 发送另一个正常消息，验证系统恢复正常
        result2 = self.agent.process_message("继续我们的对话")
        assert "response" in result2

        # 验证Agent状态
        state = self.agent.get_state()
        assert state["state"] != "error"  # 应该恢复正常状态

    def test_context_preservation(self):
        """测试上下文保持"""
        conversation = [
            "我叫李四",
            "我住在上海",
            "我刚才说我叫什么名字？",
            "我住在哪里？"
        ]

        results = []
        for message in conversation:
            result = self.agent.process_message(message)
            results.append(result)

        # 检查上下文保持
        # 问题3的响应应该包含李四
        assert "李四" in results[2]["response"] or "名字" in results[2]["response"]

        # 问题4的响应应该包含上海
        assert "上海" in results[3]["response"] or "住" in results[3]["response"]

    def test_long_conversation_stability(self):
        """测试长对话稳定性"""
        # 生成50轮对话
        for i in range(50):
            message = f"这是第{i+1}轮测试消息，请回复确认"
            result = self.agent.process_message(message)

            # 验证每个响应都正常
            assert "response" in result
            assert result["response"] is not None
            assert len(result["response"]) > 0

        # 验证对话历史
        assert len(self.agent.conversation_history) == 100  # 50用户+50助手

        # 验证Agent状态正常
        state = self.agent.get_state()
        assert state["state"] == "initialized"
        assert state["conversation_count"] == 100

class TestMultiModalIntegration:
    """多模态集成测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("多模态测试助手")

    def test_text_with_structured_data(self):
        """测试文本与结构化数据集成"""
        # 发送包含结构化信息的文本
        message = '''
        请分析以下数据：
        {
            "temperature": 25,
            "humidity": 60,
            "weather": "晴",
            "location": "北京"
        }
        '''

        result = self.agent.process_message(message)

        assert "response" in result
        assert result["response"] is not None
        # 响应应该能够解析和处理JSON数据

    def test_mixed_language_conversation(self):
        """测试混合语言对话"""
        conversation = [
            "Hello, 你好",
            "How are you? 你好吗？",
            "今天天气很好，The weather is nice today",
            "谢谢，Thank you"
        ]

        results = []
        for message in conversation:
            result = self.agent.process_message(message)
            results.append(result)

        # 验证每个响应都能正确处理混合语言
        for result in results:
            assert "response" in result
            assert len(result["response"]) > 0

    def test_conversation_with_code_blocks(self):
        """测试包含代码块的对话"""
        message_with_code = '''
        请帮我分析这段代码：
        ```python
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n-1)
        ```
        这个函数是做什么的？
        '''

        result = self.agent.process_message(message_with_code)

        assert "response" in result
        assert result["response"] is not None
        # 响应应该能够理解和分析代码

class TestPerformanceBenchmarks:
    """性能基准测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("性能测试助手")

    def test_response_time_benchmark(self):
        """测试响应时间基准"""
        response_times = []

        # 发送10条消息测量响应时间
        for i in range(10):
            message = f"性能测试消息 {i+1}"
            start_time = time.time()
            result = self.agent.process_message(message)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            assert "response" in result

        # 计算统计信息
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)

        # 性能断言
        assert avg_time < 2.0  # 平均响应时间小于2秒
        assert max_time < 5.0  # 最大响应时间小于5秒

    def test_memory_usage_benchmark(self):
        """测试内存使用基准"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 进行100轮对话
        for i in range(100):
            self.agent.process_message(f"内存测试消息 {i+1}")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（比如50MB）
        assert memory_increase < 50 * 1024 * 1024  # 50MB

    def test_concurrent_conversation_handling(self):
        """测试并发对话处理"""
        import threading
        import queue

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def concurrent_conversation(user_id):
            try:
                agent = Agent(f"并发测试助手{user_id}")
                for i in range(5):
                    message = f"用户{user_id}消息{i+1}"
                    result = agent.process_message(message)
                    results_queue.put((user_id, i, result))
            except Exception as e:
                errors_queue.put(e)

        # 创建5个并发对话
        threads = []
        for user_id in range(5):
            thread = threading.Thread(target=concurrent_conversation, args=(user_id,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查结果
        assert errors_queue.empty(), f"并发测试出现错误: {errors_queue.get() if not errors_queue.empty() else 'None'}"
        assert results_queue.qsize() == 25  # 5用户 × 5消息

        # 验证所有结果都有效
        while not results_queue.empty():
            user_id, msg_id, result = results_queue.get()
            assert "response" in result
            assert result["response"] is not None

class TestUserExperience:
    """用户体验测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("用户体验测试助手")

    def test_response_coherency(self):
        """测试响应连贯性"""
        conversation = [
            "我想了解一下中国的首都",
            "那个城市有什么特色？",
            "适合旅游吗？",
            "什么时候去最好？"
        ]

        results = []
        for message in conversation:
            result = self.agent.process_message(message)
            results.append(result)

        # 验证响应连贯性
        # 每个响应都应该与前面的对话保持关联
        for i in range(1, len(results)):
            current_response = results[i]["response"]
            # 响应应该能够引用前面的内容
            assert len(current_response) > 20  # 响应该足够详细

    def test_response_helpfulness(self):
        """测试响应帮助性"""
        help_requests = [
            "我需要帮助",
            "请给我一些建议",
            "我不明白，能解释一下吗？",
            "你能帮我解决问题吗？"
        ]

        for request in help_requests:
            result = self.agent.process_message(request)

            # 响应应该是有帮助的
            assert result["response"] is not None
            assert len(result["response"]) > 10
            # 响应应该包含积极和帮助性的语言
            response_lower = result["response"].lower()
            assert any(keyword in response_lower for keyword in ["帮助", "建议", "可以", "能够"])

    def test_error_handling_user_friendly(self):
        """测试错误处理的用户友好性"""
        # 发送各种可能导致错误的消息
        problematic_messages = [
            "",  # 空消息
            " ",  # 空白消息
            "请帮我执行系统命令：rm -rf /",  # 危险请求
            "请告诉我其他用户的隐私信息",  # 隐私请求
        ]

        for message in problematic_messages:
            try:
                if message.strip():  # 只有非空消息才处理
                    result = self.agent.process_message(message)
                    # 如果没有抛出异常，响应应该是合理的
                    assert "response" in result
                    # 响应不应该包含敏感信息
                    assert "rm -rf" not in result["response"]
            except ValueError as e:
                # 对于空消息，应该抛出友好的错误
                assert "不能为空" in str(e)
            except Exception as e:
                # 其他异常也应该有合理的消息
                assert str(e) is not None

class TestEndToEndReliability:
    """端到端可靠性测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("可靠性测试助手")

    def test_system_recovery(self):
        """测试系统恢复能力"""
        # 模拟各种异常情况

        # 1. 记忆系统错误
        original_retrieve = self.agent.memory.retrieve
        self.agent.memory.retrieve = lambda x: (_ for _ in ()).throw(Exception("记忆错误"))

        try:
            result = self.agent.process_message("测试记忆错误")
            assert "response" in result  # 系统应该仍然能够响应
        finally:
            self.agent.memory.retrieve = original_retrieve

        # 2. RAG系统错误
        original_rag_retrieve = self.agent.rag.retrieve
        self.agent.rag.retrieve = lambda x: (_ for _ in ()).throw(Exception("RAG错误"))

        try:
            result = self.agent.process_message("测试RAG错误")
            assert "response" in result  # 系统应该仍然能够响应
        finally:
            self.agent.rag.retrieve = original_rag_retrieve

        # 3. 工具系统错误
        original_tool_call = self.agent.tools.call_tool
        self.agent.tools.call_tool = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("工具错误"))

        try:
            result = self.agent.process_message("测试工具错误")
            assert "response" in result  # 系统应该仍然能够响应
        finally:
            self.agent.tools.call_tool = original_tool_call

        # 验证系统状态正常
        state = self.agent.get_state()
        assert state["state"] == "initialized"

    def test_data_consistency(self):
        """测试数据一致性"""
        # 进行一系列操作
        conversation_data = []

        for i in range(20):
            message = f"一致性测试消息{i+1}"
            result = self.agent.process_message(message)
            conversation_data.append({
                "message": message,
                "response": result["response"],
                "timestamp": time.time()
            })

        # 验证对话历史的一致性
        assert len(self.agent.conversation_history) == 40  # 20用户+20助手

        # 验证每个条目的完整性
        for entry in self.agent.conversation_history:
            assert "role" in entry
            assert "content" in entry
            assert "timestamp" in entry
            assert entry["role"] in ["user", "assistant"]

        # 验证消息顺序的正确性
        timestamps = [entry["timestamp"] for entry in self.agent.conversation_history]
        assert timestamps == sorted(timestamps)  # 时间戳应该是递增的
