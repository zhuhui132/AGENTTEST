"""Metrics 模块单元测试"""
from __future__ import annotations

import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.metrics import (  # noqa: E402
    MetricsCollector,
    AccuracyMetrics,
    SafetyMetrics,
    PerformanceMetrics as PerfMetrics,
    ComprehensiveEvaluator,
)


def test_metrics_collector_records_and_alerts():
    collector = MetricsCollector(window_size=10)
    for i in range(5):
        collector.record_request(response_time=0.1 * (i + 1))
    collector.record_request(response_time=6.0, success=True)
    collector.record_request(response_time=0.1, success=False, error="TimeoutError")

    summary = collector.get_metrics_summary()
    assert summary["request_counts"]["total"] == 7
    assert summary["response_time"]["max"] >= 6.0
    assert summary["alerts"], "应产生性能或失败告警"


def test_accuracy_metrics_scores():
    accuracy = AccuracyMetrics()
    factual = accuracy.factual_accuracy("北京是中国首都", "中国的首都是北京")
    assert factual > 0.3

    correctness = accuracy.answer_correctness("答案是42", "答案是42")
    assert correctness["precision"] == pytest.approx(1.0)
    assert correctness["f1"] == pytest.approx(1.0)


def test_safety_metrics_detection():
    safety = SafetyMetrics()
    toxic = safety.toxicity_detection("你真蠢")
    assert toxic["is_toxic"] is True

    pii = safety.privacy_leakage_detection("我的邮件 test@example.com")
    assert pii["has_pii"] is True


def test_performance_metrics_summary():
    metrics = PerfMetrics()
    request_times = [0.2, 0.3, 0.5, 0.7]
    for rt in request_times:
        metrics.record_request_time(rt)
    metrics.record_memory_usage(current_mb=100, peak_mb=150)

    summary = metrics.get_summary()
    assert summary["response_time"]["p95"] >= max(request_times)
    assert summary["memory_usage"]["peak_mb"] == 150


def test_comprehensive_evaluator():
    evaluator = ComprehensiveEvaluator()
    dataset = [
        {
            "query": "什么是Python",
            "response": "Python是一种编程语言",
            "ground_truth": "Python是一种编程语言",
            "reference_documents": ["Python 官方文档"],
            "tool_results": [],
            "response_time": 0.4,
        },
        {
            "query": "请介绍北京",
            "response": "北京是中国的首都",
            "ground_truth": "北京是中国的首都",
            "reference_documents": ["北京旅游手册"],
            "tool_results": [{"tool_name": "wiki", "success": True}],
            "response_time": 0.6,
        },
    ]

    report = evaluator.evaluate(dataset)
    assert report["average_accuracy"] >= 0.8
    assert report["average_latency"] < 1.0
    assert report["tool_success_rate"] == 100.0

