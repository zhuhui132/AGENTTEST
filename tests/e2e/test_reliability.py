"""
可靠性测试端到端测试
测试系统在异常和负载下的可靠性
"""

import pytest
import asyncio
import time
import random
from unittest.mock import Mock, AsyncMock
from src.agents.agent import IntelligentAgent
from src.core.types import AgentConfig
from src.core.exceptions import AgentError


class TestReliabilityFeatures:
    """可靠性功能测试类"""

    @pytest.fixture
    def reliability_config(self):
        """可靠性配置fixture"""
        return AgentConfig(
            model_name="test-model",
            enable_circuit_breaker=True,
            enable_retry_mechanism=True,
            enable_health_checks=True,
            timeout_seconds=30,
            max_retries=3,
            error_threshold=0.1,
            health_check_interval=60
        )

    @pytest.fixture
    def failing_llm(self):
        """间歇性失败的LLM"""
        def side_effect(*args, **kwargs):
            # 30%的概率失败
            if random.random() < 0.3:
                raise Exception("LLM服务暂时不可用")
            else:
                return Mock(content="正常响应", finish_reason="stop")

        llm = Mock()
        llm.generate = AsyncMock(side_effect=side_effect)
        return llm

    @pytest.fixture
    def slow_llm(self):
        """响应缓慢的LLM"""
        llm = Mock()

        async def slow_response(*args, **kwargs):
            # 随机延迟1-5秒
            delay = random.uniform(1, 5)
            await asyncio.sleep(delay)
            return Mock(content="延迟响应", finish_reason="stop")

        llm.generate = slow_response
        return llm

    @pytest.fixture
    def agent_with_failing_llm(self, reliability_config, failing_llm):
        """带失败LLM的Agent fixture"""
        return IntelligentAgent(config=reliability_config, llm=failing_llm)

    @pytest.fixture
    def agent_with_slow_llm(self, reliability_config, slow_llm):
        """带慢LLM的Agent fixture"""
        return IntelligentAgent(config=reliability_config, llm=slow_llm)

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, agent_with_failing_llm):
        """测试断路器功能"""
        # 发送大量请求触发断路器
        responses = []

        for i in range(20):
            try:
                response = await agent_with_failing_llm.process_message(f"测试消息{i}")
                responses.append(response)
            except Exception as e:
                responses.append({"error": str(e)})

        # 验证断路器是否触发
        # 在足够多的失败后，断路器应该打开
        error_responses = [r for r in responses if isinstance(r, dict) and "error" in r]
        success_responses = [r for r in responses if hasattr(r, 'finish_reason')]

        # 应该有错误响应
        assert len(error_responses) > 0

        # 断路器打开后，后续请求应该快速失败
        if len(error_responses) >= 5:
            # 检查是否有快速失败
            recent_responses = responses[-5:]
            quick_failures = [r for r in recent_responses
                           if isinstance(r, dict) and "circuit_breaker" in str(r)]
            assert len(quick_failures) > 0

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, agent_with_failing_llm):
        """测试重试机制"""
        retry_success_count = 0
        retry_failure_count = 0

        for i in range(10):
            try:
                response = await agent_with_failing_llm.process_message(f"重试测试{i}")

                if hasattr(response, 'finish_reason') and response.finish_reason == "stop":
                    retry_success_count += 1
                else:
                    retry_failure_count += 1

            except AgentError as e:
                if "retry" in str(e).lower():
                    retry_failure_count += 1
                else:
                    raise

        # 验证重试机制是否工作
        # 应该有一些成功的重试
        total_requests = retry_success_count + retry_failure_count
        success_rate = retry_success_count / total_requests if total_requests > 0 else 0

        # 在30%失败率下，成功率应该大于50%
        assert success_rate > 0.5
        assert retry_success_count > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, agent_with_slow_llm):
        """测试超时处理"""
        timeout_count = 0
        normal_count = 0

        for i in range(20):
            start_time = time.time()

            try:
                response = await agent_with_slow_llm.process_message(f"超时测试{i}")
                end_time = time.time()

                if end_time - start_time >= 30:  # 30秒超时
                    timeout_count += 1
                else:
                    normal_count += 1

            except TimeoutError:
                timeout_count += 1
            except Exception:
                pass  # 其他异常

        # 验证超时处理
        # 由于有随机延迟1-5秒，大多数请求应该在30秒内完成
        assert normal_count > timeout_count
        assert normal_count >= 10  # 至少一半正常完成
        assert timeout_count >= 1   # 应该有一些超时

    @pytest.mark.asyncio
    async def test_health_check_mechanism(self, reliability_config, slow_llm):
        """测试健康检查机制"""
        agent = IntelligentAgent(config=reliability_config, llm=slow_llm)

        # 等待健康检查
        await asyncio.sleep(2)

        # 执行健康检查
        health_status = await agent.health_check()

        # 验证健康检查结果
        assert 'overall_health' in health_status
        assert 'llm_health' in health_status
        assert 'memory_health' in health_status
        assert 'last_check_time' in health_status

        # 健康状态应该是可用的或降级
        assert health_status['overall_health'] in ['healthy', 'degraded', 'unhealthy']

        # 由于使用慢LLM，LLM健康可能不是最优
        assert health_status['llm_health'] in ['healthy', 'degraded']

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, agent_with_failing_llm):
        """测试优雅降级功能"""
        responses = []

        # 发送请求观察降级行为
        for i in range(15):
            try:
                response = await agent_with_failing_llm.process_message(f"降级测试{i}")
                responses.append(response)
            except Exception:
                pass

        # 分析响应模式
        successful_responses = [r for r in responses if hasattr(r, 'finish_reason')]
        error_responses = [r for r in responses if isinstance(r, dict) and "error" in r]

        # 应该有响应，但可能质量下降
        assert len(successful_responses) > 0

        # 检查是否有降级指示
        degraded_responses = [r for r in successful_responses
                            if hasattr(r, 'finish_reason') and
                               r.finish_reason in ["degraded", "limited"]]

        # 系统应该在失败情况下提供降级服务
        assert len(degraded_responses) > 0 or len(error_responses) > 0

    @pytest.mark.asyncio
    async def test_load_shedding(self, reliability_config, failing_llm):
        """测试负载卸载功能"""
        # 创建多个Agent实例模拟分布式环境
        agents = []
        for i in range(5):
            agent = IntelligentAgent(config=reliability_config, llm=failing_llm)
            agents.append(agent)

        # 并发发送大量请求
        tasks = []
        for i, agent in enumerate(agents):
            for j in range(10):
                task = agent.process_message(f"负载测试{i}_{j}")
                tasks.append(task)

        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # 分析响应
        successful = [r for r in responses if hasattr(r, 'finish_reason')]
        rejected = [r for r in responses if isinstance(r, Exception)]
        rate_limited = [r for r in successful if
                        hasattr(r, 'finish_reason') and r.finish_reason == "rate_limited"]

        total_time = end_time - start_time
        throughput = len(successful) / total_time

        # 验证负载卸载行为
        # 在高负载下，系统应该拒绝一些请求而不是全部崩溃
        assert len(rate_limited) > 0  # 应该有速率限制
        assert len(rejected) < len(tasks)  # 不是所有请求都被拒绝
        assert throughput > 0  # 应该有处理能力

    @pytest.mark.asyncio
    async def test_error_recovery(self, agent_with_failing_llm):
        """测试错误恢复功能"""
        recovery_tests = []

        for i in range(10):
            try:
                response = await agent_with_failing_llm.process_message(f"恢复测试{i}")
                recovery_tests.append({
                    "attempt": i + 1,
                    "success": True,
                    "response": response
                })
            except Exception as e:
                recovery_tests.append({
                    "attempt": i + 1,
                    "success": False,
                    "error": str(e)
                })

        # 分析恢复模式
        successful_attempts = [t for t in recovery_tests if t["success"]]
        failed_attempts = [t for t in recovery_tests if not t["success"]]

        # 验证恢复能力
        # 即使有30%失败率，也应该有一些成功
        success_rate = len(successful_attempts) / len(recovery_tests)
        assert success_rate > 0.4

        # 检查是否有从失败中恢复的案例
        # 即失败后成功的案例
        recovery_cases = []
        for i in range(1, len(recovery_tests)):
            if not recovery_tests[i-1]["success"] and recovery_tests[i]["success"]:
                recovery_cases.append(i)

        # 应该有从失败中恢复的情况
        assert len(recovery_cases) > 0

    @pytest.mark.asyncio
    async def test_consistency_under_load(self, agent_with_slow_llm):
        """测试负载下的一致性"""
        consistent_requests = ["请回答1+1等于几？"] * 20

        responses = await asyncio.gather(*[
            agent_with_slow_llm.process_message(req) for req in consistent_requests
        ])

        # 分析响应一致性
        valid_responses = [r for r in responses if hasattr(r, 'content')]
        valid_contents = [r.content for r in valid_responses]

        # 检查数字答案的一致性
        number_answers = []
        for content in valid_contents:
            # 提取数字答案
            import re
            numbers = re.findall(r'\d+', content)
            if numbers:
                number_answers.append(int(numbers[0]))

        if number_answers:
            # 大多数答案应该是2
            most_common_answer = max(set(number_answers), key=number_answers.count)
            answer_consistency = number_answers.count(most_common_answer) / len(number_answers)

            # 即使在负载下，答案也应该相对一致
            assert answer_consistency > 0.5  # 至少50%一致

    @pytest.mark.asyncio
    async def test_memory_reliability(self, agent_with_failing_llm):
        """测试记忆系统可靠性"""
        # 发送记忆相关消息
        memory_messages = [
            "我的名字是张三",
            "我住在北京市",
            "我的工作程序员",
            "我今年30岁"
        ]

        for msg in memory_messages:
            try:
                await agent_with_failing_llm.process_message(msg)
            except Exception:
                pass  # 忽略LLM错误，专注记忆

        # 测试记忆检索
        try:
            response = await agent_with_failing_llm.process_message("我叫什么名字？")

            # 即使在错误情况下，记忆应该仍然可用
            assert response.content is not None
            # 可能不完整，但应该有响应
            assert len(response.content) > 0

        except Exception as e:
            # 记忆系统应该独立于LLM
            pytest.fail(f"记忆系统不应该受LLM失败影响: {e}")

    def test_configuration_validation(self):
        """测试可靠性配置验证"""
        # 有效配置
        valid_config = AgentConfig(
            enable_circuit_breaker=True,
            max_retries=3,
            timeout_seconds=30
        )
        assert valid_config.enable_circuit_breaker is True
        assert valid_config.max_retries == 3

        # 无效配置
        invalid_configs = [
            {"max_retries": -1},  # 负数重试
            {"timeout_seconds": 0},   # 零超时
            {"error_threshold": -0.1}  # 负数阈值
        ]

        for config in invalid_configs:
            with pytest.raises((ValueError, AgentError)):
                AgentConfig(**config)

    @pytest.mark.asyncio
    async def test_monitoring_and_alerts(self, agent_with_failing_llm):
        """测试监控和警报功能"""
        # 监控指标
        initial_metrics = agent_with_failing_llm.get_reliability_metrics()

        # 发送请求产生指标变化
        for i in range(10):
            try:
                await agent_with_failing_llm.process_message(f"监控测试{i}")
            except Exception:
                pass

        # 获取更新后的指标
        updated_metrics = agent_with_failing_llm.get_reliability_metrics()

        # 验证指标更新
        assert 'error_rate' in updated_metrics
        assert 'success_rate' in updated_metrics
        assert 'avg_response_time' in updated_metrics
        assert 'circuit_breaker_status' in updated_metrics

        # 在有错误的情况下，错误率应该大于0
        assert updated_metrics['error_rate'] > 0
        assert updated_metrics['success_rate'] < 1.0

        # 检查警报条件
        alerts = agent_with_failing_llm.check_alert_conditions(updated_metrics)

        # 高错误率应该触发警报
        if updated_metrics['error_rate'] > 0.2:  # 20%错误率
            assert len(alerts) > 0
            assert any('high_error_rate' in alert['type'] for alert in alerts)


if __name__ == "__main__":
    pytest.main([__file__])
