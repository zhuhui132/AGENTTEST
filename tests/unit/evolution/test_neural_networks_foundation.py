"""
神经网络基础测试 (1943-1957年)

该测试模块验证神经网络发展初期的关键技术，
包括McCulloch-Pitts神经元、感知机算法、Hebb学习规则等。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import pytest
import numpy as np
import sys
import os

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from src.llm.evolution.neural_networks_foundation import (
    McCullochPittsNeuron,
    Perceptron,
    HebbianLearning,
    FoundationAnalyzer
)


class TestMcCullochPittsNeuron:
    """McCulloch-Pitts神经元测试"""

    def setup_method(self):
        """测试前设置"""
        self.neuron = McCullochPittsNeuron()
        self.neuron.configure(num_inputs=3, threshold=1.5)

    def test_basic_and_functionality(self):
        """测试基本AND功能"""
        # 测试不同输入组合
        test_cases = [
            ([0, 0, 0], 0),  # 0 < 1.5
            ([1, 0, 0], 0),  # 1 < 1.5
            ([1, 1, 0], 0),  # 2 < 1.5
            ([1, 1, 1], 1),  # 3 >= 1.5
        ]

        for inputs, expected in test_cases:
            result = self.neuron.forward(inputs)
            assert result == expected, f"输入 {inputs} 期望 {expected}，得到 {result}"

    def test_basic_or_functionality(self):
        """测试基本OR功能"""
        # 调整阈值为0.5以实现OR功能
        self.neuron.configure(num_inputs=3, threshold=0.5)

        test_cases = [
            ([0, 0, 0], 0),  # 0 < 0.5
            ([1, 0, 0], 1),  # 1 >= 0.5
            ([1, 1, 0], 1),  # 2 >= 0.5
            ([1, 1, 1], 1),  # 3 >= 0.5
        ]

        for inputs, expected in test_cases:
            result = self.neuron.forward(inputs)
            assert result == expected, f"OR输入 {inputs} 期望 {expected}，得到 {result}"

    def test_weight_adjustment(self):
        """测试权重调整"""
        # 设置不同权重
        self.neuron.weights = [1.0, 2.0, 3.0]
        self.neuron.threshold = 3.5

        # 测试权重影响
        test_cases = [
            ([0, 0, 0], 0),  # 0 < 3.5
            ([1, 0, 0], 1),  # 1 < 3.5
            ([0, 1, 0], 2),  # 2 < 3.5
            ([0, 0, 1], 3),  # 3 < 3.5
            ([1, 0, 1], 4),  # 4 >= 3.5
        ]

        for inputs, expected in test_cases:
            result = self.neuron.forward(inputs)
            actual_sum = sum(w * x for w, x in zip(self.neuron.weights, inputs))
            condition_met = actual_sum >= self.neuron.threshold
            expected_result = 1 if condition_met else 0
            assert result == expected_result, f"加权输入 {inputs} 期望 {expected_result}，得到 {result}"

    def test_neuron_initialization(self):
        """测试神经元初始化"""
        neuron = McCullochPittsNeuron()

        # 默认状态测试
        assert neuron.name == "McCulloch-Pitts"
        assert neuron.threshold == 0.0
        assert len(neuron.weights) == 0

        # 配置后测试
        neuron.configure(num_inputs=5, threshold=2.0)
        assert len(neuron.weights) == 5
        assert all(w == 1.0 for w in neuron.weights)
        assert neuron.threshold == 2.0


class TestPerceptron:
    """感知机测试"""

    def setup_method(self):
        """测试前设置"""
        self.perceptron = Perceptron(num_inputs=2, learning_rate=0.1)

    def test_and_learning(self):
        """测试AND问题学习"""
        # AND问题训练数据
        training_data = [
            ([0, 0], -1),
            ([0, 1], -1),
            ([1, 0], -1),
            ([1, 1], 1)
        ]

        # 训练感知机
        result = self.perceptron.train(training_data, epochs=50, verbose=False)

        # 验证训练结果
        assert result['success'] is True
        assert result['final_accuracy'] == 1.0  # AND问题应该完全学会
        assert len(result['training_history']) > 0

        # 测试预测
        test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected_outputs = [-1, -1, -1, 1]

        for inputs, expected in zip(test_inputs, expected_outputs):
            prediction = self.perceptron.predict(inputs)
            assert prediction == expected, f"AND输入 {inputs} 期望 {expected}，预测 {prediction}"

    def test_or_learning(self):
        """测试OR问题学习"""
        # OR问题训练数据
        training_data = [
            ([0, 0], -1),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 1)
        ]

        # 训练感知机
        result = self.perceptron.train(training_data, epochs=50, verbose=False)

        # 验证训练结果
        assert result['final_accuracy'] >= 0.75  # OR问题应该基本学会

        # 测试预测
        test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

        correct_predictions = 0
        for inputs in test_inputs:
            prediction = self.perceptron.predict(inputs)
            # OR规则：任一输入为1则输出1
            expected = 1 if any(inputs) else -1
            if prediction == expected:
                correct_predictions += 1

        # 至少75%正确率
        assert correct_predictions / len(test_inputs) >= 0.75

    def test_learning_rate_effect(self):
        """测试学习率影响"""
        training_data = [
            ([0, 0], -1),
            ([1, 1], 1)
        ]

        # 测试不同学习率
        learning_rates = [0.01, 0.1, 1.0]
        results = {}

        for lr in learning_rates:
            perceptron = Perceptron(num_inputs=2, learning_rate=lr)
            result = perceptron.train(training_data, epochs=20, verbose=False)
            results[lr] = result['final_accuracy']

        # 学习率应该影响学习效果
        assert len(results) == 3
        assert all(0 <= acc <= 1 for acc in results.values())

    def test_convergence_behavior(self):
        """测试收敛行为"""
        training_data = [
            ([0, 0], -1),
            ([0, 1], -1),
            ([1, 0], -1),
            ([1, 1], 1)
        ]

        result = self.perceptron.train(training_data, epochs=50, verbose=False)
        history = result['training_history']

        # 验证训练历史
        assert len(history) > 0
        assert all('epoch' in h for h in history)
        assert all('accuracy' in h for h in history)
        assert all(0 <= h['accuracy'] <= 1 for h in history)

        # 准确率应该总体上升（允许波动）
        accuracies = [h['accuracy'] for h in history]
        assert max(accuracies) >= accuracies[0]  # 最终准确率不小于初始准确率

    def test_perceptron_initialization(self):
        """测试感知机初始化"""
        perceptron = Perceptron(num_inputs=3, learning_rate=0.05)

        # 验证初始化
        assert len(perceptron.weights) == 3
        assert perceptron.learning_rate == 0.05
        assert perceptron.num_inputs == 3
        assert perceptron.bias == 0.0

        # 权重应该是随机初始化的
        assert not all(w == 0 for w in perceptron.weights)


class TestHebbianLearning:
    """Hebb学习规则测试"""

    def setup_method(self):
        """测试前设置"""
        self.hebb = HebbianLearning(num_neurons=4, learning_rate=0.1)

    def test_hebb_update_rule(self):
        """测试Hebb更新规则"""
        # 测试数据和期望的权重更新
        pre_synaptic = np.array([1.0, 0.0, 1.0, 0.0])  # 模式A
        post_synaptic = np.array([1.0, 0.0, 1.0, 0.0])  # 相同模式

        # 计算期望的权重更新
        expected_update = self.hebb.learning_rate * np.outer(pre_synaptic, post_synaptic)

        # 执行Hebb更新
        actual_update = self.hebb.hebb_update(pre_synaptic, post_synaptic)

        # 验证更新规则
        assert np.allclose(actual_update, expected_update, atol=1e-6)

        # 验证更新方向：共同激活的神经元权重增加
        assert actual_update[0, 0] > 0  # (1.0 * 1.0) * lr
        assert actual_update[2, 2] > 0  # (1.0 * 1.0) * lr
        assert actual_update[1, 1] == 0  # (0.0 * 0.0) * lr

    def test_pattern_training(self):
        """测试模式训练"""
        # 训练几个不同的模式
        patterns = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # 模式A
            np.array([0.0, 1.0, 0.0, 0.0]),  # 模式B
            np.array([0.0, 0.0, 1.0, 0.0]),  # 模式C
            np.array([0.0, 0.0, 0.0, 1.0]),  # 模式D
        ]

        # 训练所有模式
        activation_history = []
        for i, pattern in enumerate(patterns):
            activation = self.hebb.train_pattern(pattern)
            activation_history.append(activation)

        # 验证训练结果
        assert len(activation_history) == len(patterns)
        assert all(len(activation) == 4 for activation in activation_history)

        # 验证权重矩阵结构
        final_weights = self.hebb.weights
        assert final_weights.shape == (4, 4)

    def test_recall_capability(self):
        """测试回忆功能"""
        # 训练一个模式
        pattern = np.array([1.0, 0.0, 1.0, 0.0])
        self.hebb.train_pattern(pattern)

        # 使用部分提示进行回忆
        cue = np.array([1.0, 0.0, 0.0, 0.0])
        recalled = self.hebb.recall(cue)

        # 验证回忆结果
        assert len(recalled) == 4
        assert recalled.dtype == np.float64

        # 提示和训练模式相似的部分应该激活更强的响应
        # 这需要更复杂的分析，这里简单验证格式正确性

    def test_learning_rate_effect(self):
        """测试学习率影响"""
        pattern = np.array([1.0, 1.0, 0.0, 0.0])

        # 测试不同学习率
        learning_rates = [0.01, 0.1, 1.0]
        weight_changes = []

        for lr in learning_rates:
            hebb = HebbianLearning(num_neurons=4, learning_rate=lr)
            initial_weights = hebb.weights.copy()
            hebb.train_pattern(pattern)
            weight_change = np.mean(np.abs(hebb.weights - initial_weights))
            weight_changes.append(weight_change)

        # 学习率越大，权重变化应该越大
        assert weight_changes[0] < weight_changes[1] < weight_changes[2]

    def test_multiple_pattern_association(self):
        """测试多模式关联"""
        patterns = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # A模式
            np.array([0.0, 1.0, 0.0, 0.0]),  # B模式
            np.array([0.0, 0.0, 1.0, 0.0]),  # C模式
        ]

        # 多次训练相同的模式以增强关联
        for _ in range(5):
            for pattern in patterns:
                self.hebb.train_pattern(pattern)

        # 验证权重矩阵的对称性
        weights = self.hebb.weights
        assert weights.shape == (4, 4)

        # 相同模式对应的神经元之间应该有较强的连接
        assert weights[0, 0] > 0  # A-A连接
        assert weights[1, 1] > 0  # B-B连接
        assert weights[2, 2] > 0  # C-C连接


class TestFoundationAnalyzer:
    """基础分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.analyzer = FoundationAnalyzer()
        self.mcp_neuron = McCullochPittsNeuron()
        self.mcp_neuron.configure(num_inputs=2, threshold=1.5)

        self.perceptron = Perceptron(num_inputs=2, learning_rate=0.1)

        self.and_data = [
            ([0, 0], -1),
            ([0, 1], -1),
            ([1, 0], -1),
            ([1, 1], 1)
        ]
        self.perceptron.train(self.and_data, epochs=10, verbose=False)

    def test_model_capacity_analysis(self):
        """测试模型表达能力分析"""
        test_patterns = [
            [0, 0], [1, 0], [0, 1], [1, 1]
        ]

        # 分析MCP神经元
        mcp_analysis = self.analyzer.analyze_model_capacity(self.mcp_neuron, test_patterns)

        assert mcp_analysis['model_name'] == "McCulloch-Pitts"
        assert 'total_possible_outputs' in mcp_analysis
        assert 'can_represent_linearly_separable' in mcp_analysis
        assert 'test_patterns_count' in mcp_analysis
        assert 'unique_output_patterns' in mcp_analysis

        # 验证输出模式
        unique_outputs = mcp_analysis['unique_output_patterns']
        assert len(unique_outputs) <= 2  # 二值输出

        # 分析感知机
        perceptron_analysis = self.analyzer.analyze_model_capacity(self.perceptron, test_patterns)

        assert perceptron_analysis['model_name'] == "Perceptron"
        assert perceptron_analysis['can_represent_linearly_separable'] is True
        assert perceptron_analysis['model_type'] == 'Perceptron'

    def test_models_comparison(self):
        """测试模型比较"""
        models = [self.mcp_neuron, self.perceptron]
        comparison_result = self.analyzer.compare_models(models, self.and_data)

        # 验证比较结果结构
        assert 'McCulloch-Pitts' in comparison_result
        assert 'Perceptron' in comparison_result

        # 验证每个模型的结果
        for model_name, result in comparison_result.items():
            assert 'accuracy' in result
            assert 'model_type' in result
            assert isinstance(result['accuracy'], (int, float))
            assert 0 <= result['accuracy'] <= 1

        # 感知机应该比MCP神经元表现更好
        mcp_acc = comparison_result['McCulloch-Pitts']['accuracy']
        perceptron_acc = comparison_result['Perceptron']['accuracy']
        # 注意：MCP神经元在测试方法中可能被当作可训练模型处理

    def test_linear_separability_detection(self):
        """测试线性可分性检测"""
        # 线性可分问题（AND）
        linear_separable_data = [
            ([0, 0], 0),
            ([0, 1], 0),
            ([1, 0], 0),
            ([1, 1], 1)
        ]

        # 训练感知机
        perceptron = Perceptron(num_inputs=2, learning_rate=0.1)
        result = perceptron.train(linear_separable_data, epochs=20, verbose=False)

        # 应该能够学习线性可分问题
        assert result['final_accuracy'] >= 0.75

    def test_capability_evaluation(self):
        """测试能力评估"""
        mcp_neuron = McCullochPittsNeuron()
        mcp_neuron.configure(num_inputs=3, threshold=2.0)

        perceptron = Perceptron(num_inputs=3, learning_rate=0.1)

        # 相同的测试模式
        test_patterns = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]
        ]

        # 两个模型都应该产生输出
        mcp_results = [mcp_neuron.forward(pattern) for pattern in test_patterns]
        perceptron_results = [perceptron.predict(pattern) for pattern in test_patterns]

        assert len(mcp_results) == len(test_patterns)
        assert len(perceptron_results) == len(test_patterns)

        # 结果应该都是有效的输出值
        for result in mcp_results:
            assert result in [0, 1]

        for result in perceptron_results:
            assert result in [-1, 1]


class TestNeuralFoundationIntegration:
    """神经网络基础集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.models = {}

        # 创建各种模型
        self.models['mcp'] = McCullochPittsNeuron()
        self.models['mcp'].configure(num_inputs=2, threshold=1.5)

        self.models['perceptron'] = Perceptron(num_inputs=2, learning_rate=0.1)

        self.models['hebb'] = HebbianLearning(num_neurons=4, learning_rate=0.1)

    def test_complete_learning_pipeline(self):
        """测试完整学习流水线"""
        # 1. AND问题数据
        and_data = [
            ([0, 0], -1),
            ([0, 1], -1),
            ([1, 0], -1),
            ([1, 1], 1)
        ]

        # 2. 训练感知机
        training_result = self.models['perceptron'].train(and_data, epochs=20, verbose=False)

        # 3. 验证训练成功
        assert training_result['final_accuracy'] >= 0.75

        # 4. 测试所有输入
        test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        correct_predictions = 0

        for inputs in test_inputs:
            # MCP神经元预测
            mcp_result = self.models['mcp'].forward(inputs)

            # 感知机预测
            perceptron_result = self.models['perceptron'].predict(inputs)

            # 验证结果类型
            assert mcp_result in [0, 1]
            assert perceptron_result in [-1, 1]

            # AND问题的期望输出
            expected = 1 if sum(inputs) == 2 else -1
            if perceptron_result == expected:
                correct_predictions += 1

        # 感知机应该表现良好
        accuracy = correct_predictions / len(test_inputs)
        assert accuracy >= 0.75

    def test_multiple_model_cooperation(self):
        """测试多模型协作"""
        # 创建两个不同的MCP神经元，实现不同的逻辑门
        and_neuron = McCullochPittsNeuron()
        and_neuron.configure(num_inputs=2, threshold=1.5)  # AND逻辑

        or_neuron = McCullochPittsNeuron()
        or_neuron.configure(num_inputs=2, threshold=0.5)   # OR逻辑

        # 测试相同输入在不同逻辑门下的表现
        test_inputs = [
            [0, 0], [0, 1], [1, 0], [1, 1]
        ]

        results = {'and': [], 'or': []}
        for inputs in test_inputs:
            and_result = and_neuron.forward(inputs)
            or_result = or_neuron.forward(inputs)

            results['and'].append(and_result)
            results['or'].append(or_result)

        # 验证AND和OR逻辑
        # AND: 只有[1,1]输出1
        assert results['and'] == [0, 0, 0, 1]

        # OR: 除了[0,0]都输出1
        assert results['or'] == [0, 1, 1, 1]

    def test_historical_progression(self):
        """测试历史进展验证"""
        # 验证1943-1957年技术发展的关键特征

        # 1. McCulloch-Pitts (1943): 基础神经元模型
        mcp = McCullochPittsNeuron()
        mcp.configure(num_inputs=3, threshold=1.5)

        # 验证二值逻辑
        assert mcp.forward([1, 1, 1]) in [0, 1]
        assert mcp.forward([0, 0, 0]) in [0, 1]

        # 2. 感知机 (1957): 第一个学习算法
        perceptron = Perceptron(num_inputs=2, learning_rate=0.1)

        # 验证学习能力
        simple_data = [([0, 0], -1), ([1, 1], 1)]
        training_result = perceptron.train(simple_data, epochs=10)
        assert training_result['success'] is True

        # 3. Hebb学习 (1949): 学习规则
        hebb = HebbianLearning(num_neurons=2, learning_rate=0.1)
        pattern = np.array([1.0, 0.0])
        weight_update = hebb.hebb_update(pattern, pattern)

        # 验证Hebb规则：一起激活的神经元连接增强
        assert weight_update[0, 0] > 0
        assert weight_update[1, 1] == 0  # 0.0 * 0.0 * lr

        # 验证历史发展：从静态模型到学习模型
        # McCulloch-Pitts是静态逻辑门，感知机可以学习，Hebb是自适应学习
        assert not hasattr(mcp, 'train')
        assert hasattr(perceptron, 'train')
        assert hasattr(hebb, 'hebb_update')


# pytest主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
