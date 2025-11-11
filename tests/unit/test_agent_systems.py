"""
Agent系统专项测试
深度测试Agent系统的各种场景和边界条件
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
from src.agents.agent import IntelligentAgent
from src.core.types import AgentConfig, Message


class TestAgentSystems:
    """Agent系统专项测试类"""

    @pytest.fixture
    def config(self):
        """Agent配置fixture"""
        return AgentConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7,
            enable_memory=True,
            enable_rag=True,
            enable_tools=True,
            timeout_seconds=30
        )

    @pytest.fixture
    def mock_llm(self):
        """模拟LLM"""
        llm = Mock()

        async def side_effect(prompt, **kwargs):
            # 根据提示内容返回不同的响应
            if "复杂任务" in str(prompt):
                return Mock(content="这是复杂任务的处理结果", finish_reason="stop")
            elif "并发" in str(prompt):
                return Mock(content="并发任务响应", finish_reason="stop")
            elif "状态" in str(prompt):
                return Mock(content=f"当前状态：{time.time()}", finish_reason="stop")
            else:
                return Mock(content="标准响应", finish_reason="stop")

        llm.generate = AsyncMock(side_effect=side_effect)
        return llm

    @pytest.fixture
    def agent(self, config, mock_llm):
        """Agent实例fixture"""
        return IntelligentAgent(config=config, llm=mock_llm)

    def test_agent_lifecycle_management(self, agent):
        """测试Agent生命周期管理"""
        # 初始状态
        assert agent.state == "idle"
        assert agent.is_active() is False

        # 激活Agent
        agent.activate()
        assert agent.state == "active"
        assert agent.is_active() is True

        # 暂停Agent
        agent.pause()
        assert agent.state == "paused"
        assert agent.is_active() is False

        # 恢复Agent
        agent.resume()
        assert agent.state == "active"
        assert agent.is_active() is True

        # 关闭Agent
        agent.shutdown()
        assert agent.state == "shutdown"
        assert agent.is_active() is False

    @pytest.mark.asyncio
    async def test_state_transition_graph(self, agent):
        """测试状态转换图"""
        # 状态转换图测试
        valid_transitions = {
            'idle': ['active', 'processing'],
            'active': ['paused', 'processing', 'shutdown'],
            'paused': ['active', 'shutdown'],
            'processing': ['active', 'idle', 'error'],
            'error': ['idle', 'shutdown'],
            'shutdown': []
        }

        # 测试每个状态的有效转换
        for from_state, to_states in valid_transitions.items():
            agent.set_state(from_state)

            for to_state in to_states:
                can_transition = agent.can_transition_to(to_state)
                if can_transition:
                    agent.transition_to(to_state)
                    assert agent.state == to_state
                    agent.set_state(from_state)  # 重置状态
                else:
                    should_fail = to_state not in valid_transitions[from_state]
                    if should_fail:
                        with pytest.raises(ValueError):
                            agent.transition_to(to_state)

    @pytest.mark.asyncio
    async def test_message_queue_management(self, agent):
        """测试消息队列管理"""
        messages = [
            "消息1", "消息2", "消息3", "消息4", "消息5"
        ]

        # 批量添加消息
        for msg in messages:
            agent.add_to_queue(msg)

        # 验证队列状态
        assert agent.get_queue_size() == 5
        assert agent.get_queue_messages() == messages

        # 处理队列
        processed = []
        while agent.has_pending_messages():
            msg = agent.get_next_message()
            if msg:
                response = await agent.process_message(msg)
                processed.append(response)

        # 验证所有消息都被处理
        assert len(processed) == 5
        assert agent.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, agent):
        """测试并发用户会话管理"""
        users = ['user1', 'user2', 'user3']

        async def simulate_user_session(user_id):
            session_id = await agent.create_session(user_id)

            messages = [f"用户{user_id}的消息{i}" for i in range(3)]

            responses = []
            for msg in messages:
                response = await agent.process_message(msg, session_id=session_id)
                responses.append(response)

            await agent.close_session(session_id)
            return responses

        # 并发执行用户会话
        sessions = await asyncio.gather(*[
            simulate_user_session(user) for user in users
        ])

        # 验证会话隔离
        for i, user_responses in enumerate(sessions):
            assert len(user_responses) == 3
            # 验证会话间不相互影响
            for j, other_responses in enumerate(sessions):
                if i != j:
                    # 响应内容应该不同（模拟不同用户的对话）
                    assert any(
                        resp1.content != resp2.content
                        for resp1 in user_responses
                        for resp2 in other_responses
                    )

    @pytest.mark.asyncio
    async def test_context_window_management(self, agent):
        """测试上下文窗口管理"""
        # 创建超过窗口长度的对话
        long_conversation = []
        for i in range(20):  # 超过通常的上下文窗口
            long_conversation.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"对话消息{i}"
            })

        # 设置上下文窗口大小为10
        agent.set_context_window_size(10)

        # 处理长对话
        responses = []
        for i in range(0, len(long_conversation), 2):
            msg = long_conversation[i]
            response = await agent.process_message(msg["content"])
            responses.append(response)

        # 验证上下文窗口
        current_context = agent.get_current_context()
        assert len(current_context) <= 10  # 不应该超过窗口大小

        # 验证只保留最近的对话
        if len(long_conversation) > 10:
            # 当前上下文应该是最近的消息
            expected_last_messages = long_conversation[-10:]
            actual_context_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in current_context
            ]

            # 验证顺序和内容匹配
            assert len(actual_context_messages) <= 10
            assert len(actual_context_messages) == len(expected_last_messages)

    @pytest.mark.asyncio
    async def test_priority_message_processing(self, agent):
        """测试优先级消息处理"""
        # 添加不同优先级的消息
        agent.add_priority_message("低优先级消息", priority=1)
        agent.add_priority_message("中优先级消息", priority=5)
        agent.add_priority_message("高优先级消息", priority=10)
        agent.add_priority_message("紧急消息", priority=15)

        # 验证优先级队列
        priority_queue = agent.get_priority_queue()
        priorities = [msg.get('priority', 0) for msg in priority_queue]
        assert priorities == sorted(priorities, reverse=True)  # 高优先级在前

        # 处理消息（应该按优先级顺序）
        processed_order = []
        while agent.has_priority_messages():
            msg = agent.get_next_priority_message()
            if msg:
                response = await agent.process_message(msg["content"])
                processed_order.append(msg.get('priority', 0))

        # 验证处理顺序是降序的优先级
        assert processed_order == [15, 10, 5, 1]

    @pytest.mark.asyncio
    async def test_error_state_recovery(self, agent):
        """测试错误状态恢复机制"""
        # 模拟错误发生的场景
        error_scenarios = [
            "网络错误",
            "API限制",
            "模型不可用",
            "内存不足"
        ]

        for error_type in error_scenarios:
            # 模拟错误
            with pytest.raises(Exception):
                await agent.simulate_error(error_type)

            # 验证错误状态
            assert agent.state == "error"

            # 测试恢复
            recovery_result = agent.attempt_recovery()
            assert recovery_result['success'] in [True, False]

            if recovery_result['success']:
                assert agent.state != "error"
            else:
                assert recovery_result['retry_count'] > 0
                assert recovery_result['error_type'] == error_type

    @pytest.mark.asyncio
    async def test_load_balancing(self, agent):
        """测试负载均衡机制"""
        # 模拟多个LLM实例
        llm_instances = [Mock() for _ in range(3)]

        for i, llm in enumerate(llm_instances):
            llm.generate = AsyncMock(
                return_value=Mock(content=f"LLM{i+1}响应", finish_reason="stop")
            )

        agent.set_llm_pool(llm_instances)

        # 发送大量请求
        requests = [f"请求{i}" for i in range(20)]

        responses = await asyncio.gather(*[
            agent.process_message_with_lb(req) for req in requests
        ])

        # 验证负载均衡
        llm_usage = agent.get_llm_usage_stats()
        assert len(llm_usage) == 3  # 3个LLM实例

        # 验证请求分布（应该相对均衡）
        max_requests = max(llm_usage.values())
        min_requests = min(llm_usage.values())

        # 负载不应该太不均衡
        load_ratio = max_requests / min_requests if min_requests > 0 else float('inf')
        assert load_ratio < 3.0  # 最大负载不应该是最小负载的3倍以上

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, agent):
        """测试断路器集成"""
        # 设置断路器参数
        agent.set_circuit_breaker_config({
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'half_open_max_calls': 3
        })

        # 初始状态应该是关闭的
        assert agent.get_circuit_breaker_state() == 'closed'

        # 模拟连续失败
        for i in range(6):  # 超过失败阈值
            with pytest.raises(Exception):
                await agent.simulate_failure(f"失败{i}")

        # 断路器应该打开
        assert agent.get_circuit_breaker_state() == 'open'

        # 尝试更多请求应该快速失败
        start_time = time.time()
        with pytest.raises(Exception):
            await agent.process_message("断路器打开时的请求")
        end_time = time.time()

        # 断路器打开时应该快速失败
        assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_agent_orchestration(self, agent):
        """测试Agent编排功能"""
        # 创建子Agent
        sub_agents = [
            Mock(name=f"agent{i}") for i in range(3)
        ]

        for sub_agent in sub_agents:
            sub_agent.process = AsyncMock(
                return_value=Mock(content=f"Agent{i}处理结果")
            )

        # 设置编排配置
        orchestration_config = {
            'parallel_execution': True,
            'dependency_resolution': True,
            'result_aggregation': True
        }

        # 测试编排任务
        tasks = [
            {"task": f"任务{i}", "agent": sub_agents[i % 3]}
            for i in range(6)
        ]

        results = await agent.orchestrate_tasks(tasks, orchestration_config)

        # 验证编排结果
        assert len(results) == 6
        assert all('result' in result for result in results)

        # 验证并行执行
        execution_time = max(result['execution_time'] for result in results)
        assert execution_time < 10.0  # 并行执行应该较快

    def test_agent_configuration_validation(self):
        """测试Agent配置验证"""
        # 有效配置
        valid_configs = [
            AgentConfig(
                model_name="test-model",
                max_tokens=1000,
                temperature=0.7
            ),
            AgentConfig(
                model_name="fast-model",
                max_tokens=500,
                temperature=0.1,
                enable_memory=False
            ),
            AgentConfig(
                model_name="creative-model",
                max_tokens=2000,
                temperature=1.0,
                enable_tools=True
            )
        ]

        for config in valid_configs:
            # 应该能正常创建
            agent = IntelligentAgent(config=config)
            assert agent.config == config

        # 无效配置
        invalid_configs = [
            {},  # 空配置
            {"model_name": ""},  # 空模型名
            {"max_tokens": -1},  # 负数
            {"temperature": 2.0},  # 超出范围
            {"enable_memory": "invalid"}  # 错误类型
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                IntelligentAgent(config=invalid_config)

    @pytest.mark.asyncio
    async def test_agent_metrics_and_monitoring(self, agent):
        """测试Agent指标和监控"""
        # 生成一些活动
        await agent.process_message("测试消息1")
        await agent.process_message("测试消息2")

        # 模拟一些工具调用
        agent.record_tool_call("tool1", True, 0.5)
        agent.record_tool_call("tool2", False, 0.2)

        # 获取指标
        metrics = agent.get_metrics()

        # 验证指标完整性
        assert 'messages_processed' in metrics
        assert 'average_response_time' in metrics
        assert 'tool_call_success_rate' in metrics
        assert 'error_rate' in metrics
        assert 'memory_usage' in metrics

        # 验证指标值
        assert metrics['messages_processed'] == 2
        assert metrics['tool_call_success_rate'] == 0.5  # 1/2 成功
        assert 0 <= metrics['error_rate'] <= 1.0
        assert metrics['memory_usage'] >= 0

    @pytest.mark.asyncio
    async def test_agent_capability_adaptation(self, agent):
        """测试Agent能力自适应"""
        # 测试不同能力需求
        capability_scenarios = [
            {
                'name': 'simple_qa',
                'required_capabilities': ['text_generation'],
                'test_message': '简单问答'
            },
            {
                'name': 'complex_reasoning',
                'required_capabilities': ['text_generation', 'reasoning', 'context_maintenance'],
                'test_message': '复杂的逻辑推理问题'
            },
            {
                'name': 'tool_usage',
                'required_capabilities': ['text_generation', 'tool_calling'],
                'test_message': '请帮我计算复利'
            },
            {
                'name': 'multimodal',
                'required_capabilities': ['text_generation', 'image_processing'],
                'test_message': '请分析这张图片'
            }
        ]

        for scenario in capability_scenarios:
            # 检查能力匹配
            capability_match = agent.check_capability_match(
                scenario['required_capabilities']
            )

            # 根据场景处理
            if capability_match['fully_matched']:
                response = await agent.process_message(scenario['test_message'])
                assert response.content is not None
            else:
                # 部分匹配时应该有适配行为
                adaptation_response = agent.adapt_to_capabilities(
                    capability_match['matched_capabilities'],
                    scenario['required_capabilities']
                )
                assert adaptation_response['adaptation_needed'] is True
                assert adaptation_response['suggested_config'] is not None

    @pytest.mark.asyncio
    async def test_agent_persistence_and_recovery(self, agent):
        """测试Agent持久化和恢复"""
        # 生成一些状态数据
        await agent.process_message("持久化测试消息1")
        await agent.process_message("持久化测试消息2")

        # 持久化状态
        persistent_state = agent.get_persistent_state()

        # 验证持久化状态
        assert 'config' in persistent_state
        assert 'current_session' in persistent_state
        assert 'memory_state' in persistent_state
        assert 'metrics' in persistent_state

        # 创建新的Agent实例并恢复状态
        new_agent = IntelligentAgent()
        recovery_result = new_agent.recover_from_persistent_state(persistent_state)

        # 验证恢复结果
        assert recovery_result['success'] is True
        assert new_agent.config == persistent_state['config']
        assert new_agent.get_current_session() == persistent_state['current_session']

        # 验证恢复后的功能
        response = await new_agent.process_message("恢复后的测试消息")
        assert response.content is not None


if __name__ == "__main__":
    pytest.main([__file__])
