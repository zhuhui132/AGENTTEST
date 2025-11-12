"""
大语言模型时代测试 (2020年至今)

该测试模块验证大语言模型时代的关键技术，
包括少样本学习、思维链推理、涌现能力等。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import os

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from src.llm.evolution.large_language_models import (
    FewShotLearning,
    ChainOfThought,
    LargeLanguageModel,
    EmergentAbilities
)


class TestFewShotLearning:
    """少样本学习测试"""

    def setup_method(self):
        """测试前设置"""
        # 创建简单的模拟模型
        self.mock_model = MockLLM()
        self.few_shot = FewShotLearning(self.mock_model)

    def test_add_few_shot_example(self):
        """测试添加少样本示例"""
        task_name = "translation"
        examples = [
            ("hello", "你好"),
            ("goodbye", "再见"),
            ("thank you", "谢谢")
        ]

        # 添加示例
        self.few_shot.add_few_shot_example(task_name, examples)

        # 验证示例存储
        assert task_name in self.few_shot.few_shot_examples
        assert len(self.few_shot.few_shot_examples[task_name]) == 3
        assert self.few_shot.few_shot_examples[task_name] == examples

    def test_few_shot_inference_success(self):
        """测试少样本推理成功"""
        task_name = "translation"
        examples = [
            ("hello", "你好"),
            ("goodbye", "再见")
        ]

        self.few_shot.add_few_shot_example(task_name, examples)

        # 测试推理
        result = self.few_shot.few_shot_inference(task_name, "thank you")

        # 验证结果结构
        assert result['success'] is True
        assert result['task_name'] == task_name
        assert result['test_input'] == "thank you"
        assert 'prompt' in result
        assert 'response' in result
        assert result['examples_used'] == 2

    def test_few_shot_inference_failure(self):
        """测试少样本推理失败"""
        # 测试不存在的任务
        result = self.few_shot.few_shot_inference("nonexistent_task", "test input")

        # 验证失败结果
        assert result['success'] is False
        assert 'error' in result
        assert 'nonexistent_task' in result['error']

    def test_build_few_shot_prompt(self):
        """测试构建少样本学习prompt"""
        examples = [
            ("cat", "猫"),
            ("dog", "狗")
        ]
        test_input = "bird"

        # 直接测试私有方法（在实际情况下不建议）
        prompt = self.few_shot._build_few_shot_prompt(examples, test_input)

        # 验证prompt结构
        assert "根据以下示例，完成最后一个任务" in prompt
        assert "示例1：" in prompt
        assert "示例2：" in prompt
        assert "输入: cat" in prompt
        assert "输出: 猫" in prompt
        assert "输入: dog" in prompt
        assert "输出: 狗" in prompt
        assert "输入: bird" in prompt
        assert "输出: " in prompt

    def test_simulate_llm_response(self):
        """测试模拟LLM响应"""
        prompt = "请将以下文本翻译成中文：hello"

        response = self.few_shot._simulate_llm_response(prompt)

        # 验证响应
        assert isinstance(response, str)
        assert len(response) > 0
        assert "翻译" in response or "结果" in response

    def test_evaluate_few_shot_performance(self):
        """测试少样本学习性能评估"""
        task_name = "classification"
        examples = [
            ([1, 1, 1], "positive"),
            ([0, 0, 0], "negative")
        ]

        self.few_shot.add_few_shot_example(task_name, examples)

        test_cases = [
            {'task_name': task_name, 'test_input': [1, 1, 0], 'expected_output': "positive"},
            {'task_name': task_name, 'test_input': [0, 0, 1], 'expected_output': "positive"}
        ]

        result = self.few_shot.evaluate_few_shot_performance(test_cases)

        # 验证评估结果
        assert result['total_cases'] == 2
        assert 'correct_predictions' in result
        assert 'accuracy' in result
        assert 'success' in result
        assert 0 <= result['accuracy'] <= 1

    def test_few_shot_different_tasks(self):
        """测试不同任务的少样本学习"""
        # 测试多个不同任务
        tasks = {
            "translation": [("hello", "你好"), ("world", "世界")],
            "classification": ([1,0], "A"), ([0,1], "B"),
            "generation": ("标题", "文章标题"), ("正文", "文章内容")
        }

        for task_name, examples in tasks.items():
            self.few_shot.add_few_shot_example(task_name, examples)

            # 测试每个任务的推理
            if task_name == "translation":
                result = self.few_shot.few_shot_inference(task_name, "test")
            elif task_name == "classification":
                result = self.few_shot.few_shot_inference(task_name, [1,1])
            elif task_name == "generation":
                result = self.few_shot.few_shot_inference(task_name, "摘要")

            # 验证每个任务都能正常推理
            assert result['success'] is True
            assert result['task_name'] == task_name
            assert result['examples_used'] == len(examples)


class TestChainOfThought:
    """思维链推理测试"""

    def setup_method(self):
        """测试前设置"""
        self.mock_model = MockLLM()
        self.cot = ChainOfThought(self.mock_model)

    def test_cot_reasoning_basic(self):
        """测试基础思维链推理"""
        problem = "123 + 456 = ?"

        result = self.cot.cot_reasoning(problem)

        # 验证推理结果结构
        assert result['success'] is True
        assert result['problem'] == problem
        assert 'understanding' in result
        assert 'steps' in result
        assert 'reasoning_process' in result
        assert 'final_answer' in result

        # 验证步骤数量
        assert len(result['steps']) >= 1

        # 验证推理过程
        assert len(result['reasoning_process']) >= 2  # 步骤 + 最终答案

    def test_cot_different_problem_types(self):
        """测试不同问题类型的思维链"""
        problems = [
            "如果A > B，B > C，那么A和C的关系？",
            "设计一个算法来找到数组中的最大值",
            "如果今天是星期三，那么100天后是星期几？"
        ]

        results = []
        for problem in problems:
            result = self.cot.cot_reasoning(problem)
            results.append(result)

        # 验证所有问题都能处理
        assert len(results) == len(problems)

        for i, result in enumerate(results):
            assert result['success'] is True
            assert result['problem'] == problems[i]
            assert len(result['steps']) >= 1
            assert len(result['reasoning_process']) >= 2

    def test_understand_problem(self):
        """测试问题理解"""
        problems = [
            "计算 100 + 200 + 300 = ?",
            "判断这个逻辑是否正确：如果P→Q，Q→R，那么P→R",
            "总结这篇文章的主要观点"
        ]

        for problem in problems:
            understanding = self.cot._understand_problem(problem)

            # 验证理解结果
            assert isinstance(understanding, str)
            assert len(understanding) > 0
            assert understanding in [
                "数学计算问题", "逻辑推理问题", "翻译问题",
                "数学计算问题", "通用问题"
            ]

    def test_decompose_steps(self):
        """测试步骤分解"""
        test_cases = [
            ("计算 123 + 456 = ?", "数学计算问题"),
            ("如果A>B，B>C，A和C关系", "逻辑推理问题"),
            ("总结文章观点", "通用问题")
        ]

        for problem, understanding in test_cases:
            steps = self.cot._decompose_steps(problem, understanding)

            # 验证步骤结构
            assert isinstance(steps, list)
            assert len(steps) >= 1

            # 验证每个步骤
            for step in steps:
                assert isinstance(step, dict)
                assert 'step' in step
                assert 'method' in step
                assert isinstance(step['step'], str)
                assert isinstance(step['method'], str)
                assert len(step['step']) > 0
                assert len(step['method']) > 0

    def test_execute_reasoning_step(self):
        """测试推理步骤执行"""
        step = {"step": "识别数字", "method": "extract_numbers_operators"}
        problem = "计算 123 + 456 = ?"

        result = self.cot._execute_reasoning_step(step, problem)

        # 验证步骤执行结果
        assert isinstance(result, str)
        assert len(result) > 0
        assert "找到数字" in result

    def test_reasoning_process_quality(self):
        """测试推理过程质量"""
        problems = [
            "计算 100 * 2 + 50 = ?",
            "如果今天是星期一，7天后是星期几？",
            "分析这个句子的语法结构"
        ]

        quality_results = []
        for problem in problems:
            result = self.cot.cot_reasoning(problem)
            quality_results.append(result)

        # 评估质量
        quality_analysis = self.cot.evaluate_cot_quality(problems)

        # 验证质量分析结果
        assert quality_analysis['evaluated_problems'] == len(problems)
        assert 'average_quality_score' in quality_analysis
        assert 'quality_threshold_met' in quality_analysis
        assert 'detailed_results' in quality_analysis

        # 验证质量指标
        avg_quality = quality_analysis['average_quality_score']
        assert 0 <= avg_quality <= 1

    def test_cot_history_tracking(self):
        """测试思维链历史跟踪"""
        problems = [
            "计算 10 + 20 = ?",
            "计算 30 + 40 = ?",
            "计算 50 + 60 = ?"
        ]

        # 执行多个推理
        for problem in problems:
            self.cot.cot_reasoning(problem)

        # 验证历史记录
        assert len(self.cot.reasoning_history) == len(problems)

        for i, history_item in enumerate(self.cot.reasoning_history):
            assert history_item['problem'] == problems[i]
            assert 'steps' in history_item
            assert 'reasoning_process' in history_item
            assert 'final_answer' in history_item


class TestLargeLanguageModel:
    """大语言模型测试"""

    def setup_method(self):
        """测试前设置"""
        self.vocab_size = 1000
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 2  # 小规模测试

        self.llm = LargeLanguageModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        self.batch_size = 2
        self.seq_len = 10

    def test_model_initialization(self):
        """测试模型初始化"""
        # 验证基本属性
        assert self.llm.vocab_size == self.vocab_size
        assert self.llm.d_model == self.d_model
        assert self.llm.num_heads == self.num_heads
        assert self.llm.num_layers == self.num_layers

        # 验证组件存在
        assert hasattr(self.llm, 'embedding')
        assert hasattr(self.llm, 'decoder_layers')
        assert hasattr(self.llm, 'lm_head')

        # 验证参数数量
        total_params = sum(p.numel() for p in self.llm.parameters())
        assert total_params > 0
        assert total_params > 100000  # 至少十万级别

    def test_forward_pass(self):
        """测试前向传播"""
        # 创建输入
        input_ids = torch.randint(0, self.vocab_size,
                              (self.batch_size, self.seq_len))

        # 前向传播
        output = self.llm.forward(input_ids)

        # 验证输出形状
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        assert output.shape == expected_shape

        # 验证输出类型
        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32

        # 验证输出数值有效性
        assert torch.isfinite(output).all()

    def test_generation(self):
        """测试文本生成"""
        # 创建输入
        input_ids = torch.randint(0, self.vocab_size,
                              (1, 5))  # batch_size=1, seq_len=5

        # 生成文本
        generated = self.llm.generate(
            input_ids,
            max_length=15,
            temperature=1.0,
            do_sample=True
        )

        # 验证生成结果
        assert generated.shape[0] == 1  # batch_size
        assert generated.shape[1] >= 5   # 至少包含原始输入
        assert generated.shape[1] <= 15  # 不超过最大长度

        # 验证生成值的有效性
        assert torch.all(generated >= 0)
        assert torch.all(generated < self.vocab_size)
        assert torch.isfinite(generated).all()

    def test_generation_different_settings(self):
        """测试不同生成设置"""
        input_ids = torch.randint(0, self.vocab_size, (1, 3))

        # 测试不同温度
        temperatures = [0.1, 0.5, 1.0, 2.0]
        for temp in temperatures:
            generated = self.llm.generate(
                input_ids,
                max_length=10,
                temperature=temp,
                do_sample=True
            )

            assert generated.shape[0] == 1
            assert generated.shape[1] <= 10
            assert torch.isfinite(generated).all()

        # 测试采样vs贪心
        generated_greedy = self.llm.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            do_sample=False
        )

        generated_sampled = self.llm.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            do_sample=True
        )

        # 两者应该都是有效的生成
        assert generated_greedy.shape == generated_sampled.shape
        assert torch.isfinite(generated_greedy).all()
        assert torch.isfinite(generated_sampled).all()

    def test_model_with_different_configurations(self):
        """测试不同模型配置"""
        configs = [
            {'vocab_size': 100, 'd_model': 128, 'num_heads': 4, 'num_layers': 2},
            {'vocab_size': 500, 'd_model': 256, 'num_heads': 8, 'num_layers': 4},
            {'vocab_size': 1000, 'd_model': 512, 'num_heads': 8, 'num_layers': 6}
        ]

        for config in configs:
            model = LargeLanguageModel(**config)

            # 验证配置
            assert model.vocab_size == config['vocab_size']
            assert model.d_model == config['d_model']
            assert model.num_heads == config['num_heads']
            assert model.num_layers == config['num_layers']

            # 简单前向传播测试
            input_ids = torch.randint(0, config['vocab_size'], (1, 5))
            output = model.forward(input_ids)

            expected_shape = (1, 5, config['vocab_size'])
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()

    def test_weight_initialization(self):
        """测试权重初始化"""
        # 检查嵌入层初始化
        assert self.llm.embedding.weight.requires_grad is True

        # 检查解码器层初始化
        for layer in self.llm.decoder_layers:
            # 每个层应该有权重
            total_layer_params = sum(p.numel() for p in layer.parameters())
            assert total_layer_params > 0

        # 检查输出层初始化
        assert self.llm.lm_head.weight.requires_grad is True
        assert self.llm.lm_head.weight.shape == (self.d_model, self.vocab_size)

    def test_model_inference_speed(self):
        """测试模型推理速度"""
        input_ids = torch.randint(0, self.vocab_size, (4, 10))

        import time
        start_time = time.time()

        # 多次前向传播
        for _ in range(10):
            output = self.llm.forward(input_ids)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 验证推理时间合理（这里只做基本检查）
        assert elapsed_time > 0
        assert elapsed_time < 10  # 应该在10秒内完成

        # 验证所有推理都成功
        assert torch.isfinite(output).all()

    def test_model_memory_usage(self):
        """测试模型内存使用"""
        input_ids = torch.randint(0, self.vocab_size, (2, 10))

        # 前向传播
        output = self.llm.forward(input_ids)

        # 检查内存使用（这里做基本检查）
        assert torch.isfinite(output).all()

        # 检查梯度内存
        loss = torch.sum(output)
        loss.backward()

        # 验证梯度存在
        grad_params = [p.grad is not None for p in self.llm.parameters()
                     if p.requires_grad]
        assert any(grad_params)  # 至少有些参数有梯度


class TestEmergentAbilities:
    """涌现能力测试"""

    def setup_method(self):
        """测试前设置"""
        self.mock_llm = MockLLM()
        self.emergent = EmergentAbilities(self.mock_llm)

    def test_arithmetic_reasoning(self):
        """测试算术推理能力"""
        test_cases = [
            {"problem": "100 + 200 = ?", "expected": 300},
            {"problem": "50 * 4 = ?", "expected": 200},
            {"problem": "1000 / 25 = ?", "expected": 40}
        ]

        result = self.emergent.test_arithmetic_reasoning(test_cases)

        # 验证测试结果结构
        assert result['ability'] == 'arithmetic_reasoning'
        assert result['test_cases'] == len(test_cases)
        assert 'correct_count' in result
        assert 'accuracy' in result
        assert 'results' in result
        assert 'emergence_detected' in result
        assert 'success' in result

        # 验证测试案例
        for case_result in result['results']:
            assert 'problem' in case_result
            assert 'expected' in case_result
            assert 'model_response' in case_result
            assert 'extracted_answer' in case_result
            assert 'is_correct' in case_result

    def test_code_generation(self):
        """测试代码生成能力"""
        programming_tasks = [
            {"description": "计算两个数字的和", "language": "python"},
            {"description": "验证邮箱格式", "language": "python"},
            {"description": "创建HTML页面", "language": "javascript"}
        ]

        result = self.emergent.test_code_generation(programming_tasks)

        # 验证测试结果
        assert result['ability'] == 'code_generation'
        assert result['test_cases'] == len(programming_tasks)
        assert 'average_quality' in result
        assert 'results' in result
        assert 'emergence_detected' in result
        assert 'success' in result

        # 验证每个代码生成结果
        for code_result in result['results']:
            assert 'description' in code_result
            assert 'language' in code_result
            assert 'generated_code' in code_result
            assert 'quality_score' in code_result
            assert 'has_function' in code_result
            assert 'has_comments' in code_result
            assert 'indentation_correct' in code_result

    def test_translation_capability(self):
        """测试翻译能力"""
        translation_tasks = [
            {
                "source_text": "Hello world",
                "source_lang": "English",
                "target_lang": "Chinese",
                "expected_translation": "你好世界"
            },
            {
                "source_text": "How are you",
                "source_lang": "English",
                "target_lang": "Chinese",
                "expected_translation": "你好吗"
            }
        ]

        result = self.emergent.test_translation_capability(translation_tasks)

        # 验证测试结果
        assert result['ability'] == 'translation'
        assert result['test_cases'] == len(translation_tasks)
        assert 'average_bleu' in result
        assert 'results' in result
        assert 'emergence_detected' in result
        assert 'success' in result

        # 验证每个翻译结果
        for trans_result in result['results']:
            assert 'source_text' in trans_result
            assert 'source_lang' in trans_result
            assert 'target_lang' in trans_result
            assert 'expected_translation' in trans_result
            assert 'model_translation' in trans_result
            assert 'bleu_score' in trans_result

    def test_simulate_arithmetic_response(self):
        """测试算术问题回答模拟"""
        problems = ["100 + 200 = ?", "50 * 4 = ?", "1000 / 25 = ?"]

        for problem in problems:
            response = self.emergent._simulate_arithmetic_response(problem)

            # 验证响应格式
            assert isinstance(response, str)
            assert len(response) > 0
            assert "计算结果是" in response or "需要更多信息" in response

    def test_simulate_code_generation(self):
        """测试代码生成模拟"""
        descriptions = [
            ("计算两个数字的和", "python"),
            ("验证邮箱格式", "python"),
            ("创建HTML页面", "javascript")
        ]

        for description, language in descriptions:
            code = self.emergent._simulate_code_generation(description, language)

            # 验证代码结构
            assert isinstance(code, str)
            assert len(code) > 0

            if language == 'python':
                assert 'def ' in code
            elif language == 'javascript':
                assert 'function' in code

    def test_extract_numerical_answer(self):
        """测试数字答案提取"""
        responses = [
            "计算结果是：300",
            "答案是45.5",
            "最终答案是-200",
            "无法计算这个问题"
        ]

        for response in responses:
            extracted = self.emergent._extract_numerical_answer(response)

            # 验证提取的数字
            assert isinstance(extracted, float)
            if "计算结果是" in response:
                assert extracted != 0.0
            else:
                # 对于无法计算的情况，可能返回0
                assert extracted >= 0.0

    def test_evaluate_code_quality(self):
        """测试代码质量评估"""
        code_samples = [
            "def add(a, b): return a + b",
            '''
def sum_list(numbers):
    """计算列表中所有数字的和"""
    total = 0
    for num in numbers:
        total += num
    return total
            ''',
            "function add(a, b) { return a + b; }"
        ]

        languages = ['python', 'python', 'javascript']

        for code, language in zip(code_samples, languages):
            quality = self.emergent._evaluate_code_quality(code, language)

            # 验证质量评分
            assert isinstance(quality, float)
            assert 0.0 <= quality <= 1.0

            # 验证代码结构检查
            if language == 'python':
                if 'def ' in code:
                    assert quality >= 0.3  # 有函数定义
                if '"""' in code or '#' in code:
                    assert quality >= 0.5  # 有注释
                if len(code.split('\n')) > 3:
                    assert quality >= 0.1  # 有多行
            elif language == 'javascript':
                if 'function' in code:
                    assert quality >= 0.3  # 有函数定义

    def test_generate_emergent_abilities_report(self):
        """测试涌现能力报告生成"""
        # 先测试一些能力
        arithmetic_cases = [
            {"problem": "100 + 200 = ?", "expected": 300}
        ]

        code_tasks = [
            {"description": "简单的函数", "language": "python"}
        ]

        # 执行测试
        self.emergent.test_arithmetic_reasoning(arithmetic_cases)
        self.emergent.test_code_generation(code_tasks)

        # 生成报告
        report = self.emergent.generate_emergent_abilities_report()

        # 验证报告结构
        assert 'total_abilities_tested' in report
        assert 'emergence_rate' in report
        assert 'successful_emergences' in report
        assert 'abilities_summary' in report
        assert 'test_results' in report

        # 验证能力汇总
        summary = report['abilities_summary']
        assert 'arithmetic' in summary
        assert 'code' in summary

        for ability, info in summary.items():
            assert 'demonstrated' in info
            assert 'performance_score' in info
            assert 'emergence_confidence' in info


class TestLargeLanguageModelIntegration:
    """大语言模型集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.llm = LargeLanguageModel(
            vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_layers=2
        )

    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建模拟数据
        input_text = "The weather is"
        input_ids = torch.randint(0, 1000, (1, 4))  # 模拟tokenized输入

        # 前向传播
        output = self.llm.forward(input_ids)

        # 生成新token
        generated = self.llm.generate(input_ids, max_length=8)

        # 验证端到端流程
        assert output.shape == (1, 4, 1000)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4  # 至少包含原始输入
        assert torch.isfinite(output).all()
        assert torch.isfinite(generated).all()

    def test_few_shot_with_llm(self):
        """测试LLM与少样本学习集成"""
        few_shot = FewShotLearning(self.llm)

        # 添加示例
        examples = [
            ("cat", "猫"),
            ("dog", "狗")
        ]

        few_shot.add_few_shot_example("translation", examples)

        # 测试推理
        result = few_shot.few_shot_inference("translation", "bird")

        # 验证集成功能
        assert result['success'] is True
        assert result['examples_used'] == 2

    def test_cot_with_llm(self):
        """测试LLM与思维链推理集成"""
        cot = ChainOfThought(self.llm)

        problem = "计算 100 + 200 = ?"
        result = cot.cot_reasoning(problem)

        # 验证集成功能
        assert result['success'] is True
        assert len(result['steps']) >= 1
        assert len(result['reasoning_process']) >= 2

    def test_emergent_abilities_with_llm(self):
        """测试LLM与涌现能力集成"""
        emergent = EmergentAbilities(self.llm)

        # 测试算术能力
        arithmetic_cases = [
            {"problem": "10 + 20 = ?", "expected": 30}
        ]

        result = emergent.test_arithmetic_reasoning(arithmetic_cases)

        # 验证集成功能
        assert result['success'] is True
        assert result['ability'] == 'arithmetic_reasoning'
        assert 'results' in result

    def test_performance_comparison(self):
        """测试性能比较"""
        # 创建不同配置的模型
        models = {
            'small': LargeLanguageModel(vocab_size=500, d_model=128, num_heads=2, num_layers=2),
            'medium': LargeLanguageModel(vocab_size=1000, d_model=256, num_heads=4, num_layers=4),
            'large': LargeLanguageModel(vocab_size=2000, d_model=512, num_heads=8, num_layers=6)
        }

        # 比较参数数量
        param_counts = {}
        for name, model in models.items():
            param_counts[name] = sum(p.numel() for p in model.parameters())

        # 验证参数规模递增
        assert param_counts['small'] < param_counts['medium'] < param_counts['large']

        # 简单性能测试
        test_input = torch.randint(0, 500, (1, 5))

        performance = {}
        for name, model in models.items():
            import time
            start_time = time.time()

            for _ in range(10):
                output = model.forward(test_input)

            end_time = time.time()
            performance[name] = end_time - start_time

        # 验证性能结果
        assert len(performance) == len(models)
        for time_taken in performance.values():
            assert time_taken > 0


class MockLLM:
    """模拟大语言模型"""

    def __init__(self):
        # 简单的模拟响应
        self.responses = {
            "translation": "这是翻译结果",
            "calculation": "计算结果是{result}",
            "default": "这是通用响应"
        }

    def generate(self, prompt, max_tokens=100, temperature=1.0):
        """模拟生成"""
        if "翻译" in prompt:
            return self.responses["translation"]
        elif "计算" in prompt or "+" in prompt or "*" in prompt:
            # 简单的数学计算
            if "100 + 200" in prompt:
                return self.responses["calculation"].format(result=300)
            return self.responses["calculation"].format(result=0)
        else:
            return self.responses["default"]


# pytest主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
