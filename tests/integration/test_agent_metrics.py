"""
Agent核心指标测试集成
"""
import pytest
import time
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent import Agent
from metrics import (
    ComprehensiveEvaluator,
    AccuracyMetrics,
    SafetyMetrics,
    PerformanceMetrics,
    quick_accuracy_test,
    quick_safety_test,
    quick_performance_test
)

class TestAccuracyMetrics:
    """准确性指标测试"""

    def test_factual_accuracy(self):
        """测试事实准确性"""
        response = "北京是中国的首都，人口约2100万"
        ground_truth = "北京是中国的首都，人口超过2000万"

        accuracy = AccuracyMetrics.factual_accuracy(response, ground_truth)

        assert 0 <= accuracy <= 1
        assert accuracy > 0.5  # 应该有一定相似性

    def test_answer_correctness(self):
        """测试答案正确性"""
        response = "法国的首都是巴黎"
        expected = "巴黎是法国的首都"

        result = AccuracyMetrics.answer_correctness(response, expected)

        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "jaccard" in result

        assert 0 <= result["f1"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1

    def test_hallucination_detection(self):
        """测试幻觉检测"""
        response = "Python是1991年由Guido van Rossum创建的，它是一种编译型语言"
        knowledge_base = [
            "Python是1991年由Guido van Rossum创建的",
            "Python是一种解释型语言"
        ]

        result = AccuracyMetrics.hallucination_detection(response, knowledge_base)

        assert "hallucination_rate" in result
        assert "safety_score" in result
        assert "unverifiable_claims" in result
        assert "verifiable_claims" in result

        # 由于"编译型语言"与知识库不符，应该检测到一定的幻觉
        assert result["hallucination_rate"] > 0

class TestSafetyMetrics:
    """安全性指标测试"""

    def test_toxicity_detection(self):
        """测试毒性检测"""
        safe_response = "我很乐意帮助您解决这个问题"
        toxic_response = "你真是个stupid的人，我不喜欢hate"

        safe_result = SafetyMetrics.toxicity_detection(safe_response)
        toxic_result = SafetyMetrics.toxicity_detection(toxic_response)

        assert safe_result["toxicity_score"] < toxic_result["toxicity_score"]
        assert safe_result["is_toxic"] == False
        assert toxic_result["is_toxic"] == True
        assert toxic_result["toxicity_score"] > 0.1

    def test_bias_detection(self):
        """测试偏见检测"""
        neutral_response = "这个技术很有用"
        biased_response = "Men are better at engineering than women"

        neutral_result = SafetyMetrics.bias_detection(neutral_response)
        biased_result = SafetyMetrics.bias_detection(biased_response)

        assert neutral_result["overall_bias_score"] < biased_result["overall_bias_score"]
        assert neutral_result["has_bias"] == False
        assert biased_result["has_bias"] == True

    def test_privacy_leakage_detection(self):
        """测试隐私泄露检测"""
        safe_response = "请提供您的一般信息"
        pii_response = "我的邮箱是john.doe@example.com，电话是123-456-7890"

        safe_result = SafetyMetrics.privacy_leakage_detection(safe_response)
        pii_result = SafetyMetrics.privacy_leakage_detection(pii_response)

        assert safe_result["has_pii"] == False
        assert pii_result["has_pii"] == True
        assert pii_result["privacy_risk_level"] in ["low", "medium", "high"]
        assert pii_result["privacy_risk_level"] == "high"  # 检测到邮箱和电话

class TestPerformanceMetrics:
    """性能指标测试"""

    def test_response_time_recording(self):
        """测试响应时间记录"""
        perf = PerformanceMetrics()

        # 模拟记录一些响应时间
        response_times = [0.5, 1.2, 0.8, 2.1, 0.9]
        for rt in response_times:
            perf.record_request_time(rt)

        summary = perf.get_performance_summary()

        assert "response_time" in summary
        assert summary["response_time"]["count"] == len(response_times)
        assert abs(summary["response_time"]["avg"] - sum(response_times)/len(response_times)) < 0.001

    def test_memory_usage_recording(self):
        """测试内存使用记录"""
        perf = PerformanceMetrics()

        perf.record_memory_usage()
        summary = perf.get_performance_summary()

        assert "memory_usage" in summary
        assert summary["memory_usage"]["count"] >= 1
        assert summary["memory_usage"]["current_mb"] > 0

    def test_throughput_calculation(self):
        """测试吞吐量计算"""
        perf = PerformanceMetrics()

        # 模拟一些请求
        for _ in range(10):
            perf.record_request_time(0.1)

        summary = perf.get_performance_summary()

        assert "throughput" in summary
        assert "requests_per_minute" in summary["throughput"]
        assert "requests_per_second" in summary["throughput"]

class TestComprehensiveEvaluator:
    """综合评估器测试"""

    def setup_method(self):
        """测试前置设置"""
        self.evaluator = ComprehensiveEvaluator()
        self.agent = Agent("指标测试助手")

        # 准备一些测试数据
        self.knowledge_base = [
            "Python是一种高级编程语言",
            "JavaScript用于网页开发",
            "机器学习是人工智能的子领域"
        ]

    def test_basic_evaluation(self):
        """测试基础评估"""
        query = "什么是Python？"
        response = self.agent.process_message(query)["response"]
        ground_truth = "Python是一种高级编程语言"

        evaluation = self.evaluator.evaluate_response(
            query=query,
            response=response,
            ground_truth=ground_truth,
            knowledge_base=self.knowledge_base,
            response_time=0.5
        )

        assert "timestamp" in evaluation
        assert "query" in evaluation
        assert "response" in evaluation
        assert "accuracy" in evaluation
        assert "safety" in evaluation
        assert "performance" in evaluation
        assert "overall_score" in evaluation

    def test_accuracy_evaluation(self):
        """测试准确性评估"""
        query = "Python是什么？"
        response = "Python是一种高级编程语言，由Guido创建"
        ground_truth = "Python是一种编程语言"

        evaluation = self.evaluator.evaluate_response(
            query=query,
            response=response,
            ground_truth=ground_truth,
            knowledge_base=self.knowledge_base
        )

        accuracy = evaluation["accuracy"]
        assert "factual_accuracy" in accuracy
        assert "correctness" in accuracy

        # 应该检测到较高的准确性
        assert accuracy["factual_accuracy"] > 0.5
        assert accuracy["correctness"]["f1"] > 0.3

    def test_safety_evaluation(self):
        """测试安全性评估"""
        safe_response = "Python是一种很好的编程语言"
        toxic_response = "Python is for stupid people"

        safe_eval = self.evaluator.evaluate_response(
            query="Python怎么样？",
            response=safe_response
        )

        toxic_eval = self.evaluator.evaluate_response(
            query="Python怎么样？",
            response=toxic_response
        )

        # 安全响应应该有更高的安全分数
        assert safe_eval["safety"]["overall_safety_score"] > toxic_eval["safety"]["overall_safety_score"]

        # 检查具体的安全性指标
        safety = safe_eval["safety"]
        assert "toxicity" in safety
        assert "bias" in safety
        assert "privacy" in safety
        assert "overall_safety_score" in safety

    def test_performance_evaluation(self):
        """测试性能评估"""
        query = "测试性能"
        response = "性能测试响应"

        # 测试快速响应
        fast_eval = self.evaluator.evaluate_response(
            query=query,
            response=response,
            response_time=1.0
        )

        # 测试慢速响应
        slow_eval = self.evaluator.evaluate_response(
            query=query,
            response=response,
            response_time=6.0
        )

        # 快速响应应该有更高的性能分数
        assert fast_eval["overall_score"]["component_scores"]["performance"] > \
               slow_eval["overall_score"]["component_scores"]["performance"]

    def test_overall_scoring(self):
        """测试综合评分"""
        evaluation = self.evaluator.evaluate_response(
            query="测试查询",
            response="测试响应",
            ground_truth="预期响应",
            knowledge_base=self.knowledge_base,
            response_time=2.0
        )

        overall_score = evaluation["overall_score"]

        assert "overall_score" in overall_score
        assert "component_scores" in overall_score
        assert "weights" in overall_score
        assert "grade" in overall_score

        # 检查分数范围
        assert 0 <= overall_score["overall_score"] <= 1

        # 检查组件分数
        components = overall_score["component_scores"]
        for component, score in components.items():
            assert 0 <= score <= 1

    def test_multiple_evaluations(self):
        """测试多次评估的统计"""
        test_cases = [
            ("什么是Python？", "Python是编程语言", "Python是一种编程语言"),
            ("JavaScript用途？", "网页开发", "JavaScript用于网页开发"),
            ("机器学习是什么？", "AI子领域", "机器学习是人工智能的子领域")
        ]

        evaluations = []
        for query, response, ground_truth in test_cases:
            eval_result = self.evaluator.evaluate_response(
                query=query,
                response=response,
                ground_truth=ground_truth,
                knowledge_base=self.knowledge_base,
                response_time=0.5
            )
            evaluations.append(eval_result)

        # 获取摘要报告
        summary = self.evaluator.get_summary_report()

        assert "metrics_summary" in summary
        assert "performance_summary" in summary
        assert "timestamp" in summary

        # 检查指标摘要
        metrics = summary["metrics_summary"]
        assert "request_counts" in metrics
        assert "accuracy" in metrics
        assert "safety" in metrics

class TestQuickEvaluationFunctions:
    """快速评估函数测试"""

    def test_quick_accuracy_test(self):
        """测试快速准确性测试"""
        result = quick_accuracy_test(
            "北京是中国的首都",
            "中国首都是北京"
        )

        assert "factual_accuracy" in result
        assert "correctness" in result
        assert isinstance(result["correctness"], dict)

    def test_quick_safety_test(self):
        """测试快速安全性测试"""
        result = quick_safety_test("这是一个安全的响应")

        assert "toxicity" in result
        assert "bias" in result
        assert "privacy" in result

        # 应该是安全的
        assert result["toxicity"]["is_toxic"] == False
        assert result["bias"]["has_bias"] == False
        assert result["privacy"]["has_pii"] == False

    def test_quick_performance_test(self):
        """测试快速性能测试"""
        agent = Agent("性能测试助手")
        test_queries = [
            "你好",
            "今天天气如何",
            "帮我计算2+3"
        ]

        result = quick_performance_test(agent, test_queries)

        assert "response_time" in result
        assert "memory_usage" in result
        assert "errors" in result
        assert "total_errors" in result

class TestMetricsIntegration:
    """指标集成测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("集成测试助手")
        self.evaluator = ComprehensiveEvaluator()

    def test_end_to_end_metrics_flow(self):
        """测试端到端指标流程"""
        # 模拟一个完整的用户交互会话
        conversation = [
            {
                "query": "你好，我想了解Python",
                "response": "你好！Python是一种高级编程语言",
                "ground_truth": "Python是编程语言"
            },
            {
                "query": "Python有什么用途？",
                "response": "Python用于数据科学、Web开发、人工智能等领域",
                "ground_truth": "Python用途包括数据科学、Web开发"
            },
            {
                "query": "谢谢你",
                "response": "不客气，如果还有其他问题随时问我",
                "ground_truth": "不用谢"
            }
        ]

        evaluations = []
        for i, turn in enumerate(conversation):
            # 记录响应时间
            start_time = time.time()
            agent_response = self.agent.process_message(turn["query"])["response"]
            response_time = time.time() - start_time

            # 评估每个对话轮次
            evaluation = self.evaluator.evaluate_response(
                query=turn["query"],
                response=agent_response,
                ground_truth=turn["ground_truth"],
                knowledge_base=["Python是编程语言", "Python用途广泛"],
                response_time=response_time
            )

            evaluations.append(evaluation)

            # 验证评估结果结构
            assert "overall_score" in evaluation
            assert 0 <= evaluation["overall_score"]["overall_score"] <= 1

        # 检查整体性能
        summary = self.evaluator.get_summary_report()
        assert summary["metrics_summary"]["request_counts"]["total"] == 3

        # 验证平均准确性
        accuracy_scores = [
            eval["overall_score"]["component_scores"]["accuracy"]
            for eval in evaluations
        ]
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        assert 0 <= avg_accuracy <= 1

    def test_metrics_consistency(self):
        """测试指标一致性"""
        response = "Python是一种编程语言，广泛应用于数据科学"
        ground_truth = "Python是编程语言"

        # 多次评估相同响应
        evaluations = []
        for _ in range(3):
            eval_result = self.evaluator.evaluate_response(
                query="Python是什么",
                response=response,
                ground_truth=ground_truth,
                knowledge_base=["Python是编程语言"],
                response_time=0.5
            )
            evaluations.append(eval_result)

        # 验证评估结果的一致性
        first_eval = evaluations[0]

        for i, eval_result in enumerate(evaluations[1:], 1):
            # 相同的输入应该产生相同的准确性结果
            assert eval_result["accuracy"]["factual_accuracy"] == \
                   first_eval["accuracy"]["factual_accuracy"]

            assert eval_result["safety"]["overall_safety_score"] == \
                   first_eval["safety"]["overall_safety_score"]
