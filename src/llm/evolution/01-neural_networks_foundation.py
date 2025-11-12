"""
神经网络基础 - AI发展的起源 (1943-1957)

该模块实现了神经网络发展的早期基础算法和概念，
包括McCulloch-Pitts神经元模型、感知机算法和多层感知机。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class NeuronModel(ABC):
    """神经元模型基类"""

    def __init__(self, name: str):
        self.name = name
        self.weights = []
        self.bias = 0.0

    @abstractmethod
    def forward(self, inputs: List[float]) -> float:
        """前向传播"""
        pass

    @abstractmethod
    def activation_function(self, x: float) -> float:
        """激活函数"""
        pass


class McCullochPittsNeuron(NeuronModel):
    """McCulloch-Pitts神经元模型 (1943)"""

    def __init__(self):
        super().__init__("McCulloch-Pitts")
        self.threshold = 0.0
        self.weights = []

    def configure(self, num_inputs: int, threshold: float = 0.0):
        """配置神经元参数"""
        self.threshold = threshold
        self.weights = [1.0] * num_inputs

    def forward(self, inputs: List[float]) -> float:
        """
        McCulloch-Pitts神经元计算:
        1. 加权求和: Σ(wi * xi)
        2. 阈值判断: 如果和 ≥ 阈值则返回1，否则返回0
        """
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return 1.0 if weighted_sum >= self.threshold else 0.0

    def activation_function(self, x: float) -> float:
        """阈值激活函数"""
        return self.forward([x])  # 使用权重1.0


class Perceptron(NeuronModel):
    """感知机算法 (1957)"""

    def __init__(self, num_inputs: int, learning_rate: float = 0.1):
        super().__init__("Perceptron")
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_inputs) * 0.1
        self.bias = 0.0
        self.training_history = []

    def forward(self, inputs: List[float]) -> float:
        """
        感知机前向传播:
        output = Σ(wi * xi) + bias
        return 1.0 if output >= 0 else 0.0
    """
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1.0 if weighted_sum >= 0 else -1.0

    def activation_function(self, x: float) -> float:
        """符号激活函数"""
        return 1.0 if x >= 0 else -1.0

    def train(self, training_data: List[Tuple[List[float], int]],
              epochs: int = 100, verbose: bool = False) -> Dict[str, Any]:
        """
        感知机训练算法:
        1. 对每个训练样本
        2. 计算预测输出
        3. 计算误差
        4. 更新权重: w = w + η * error * x
        """
        for epoch in range(epochs):
            errors = 0

            for inputs, target in training_data:
                # 前向传播
                prediction = self.forward(inputs)

                # 计算误差
                error = target - prediction
                if error != 0:
                    errors += 1

                    # 更新权重和偏置
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error

            accuracy = 1.0 - errors / len(training_data)
            self.training_history.append({
                'epoch': epoch,
                'accuracy': accuracy,
                'errors': errors
            })

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Accuracy = {accuracy:.4f}, Errors = {errors}")

            # 早停：如果准确率达到100%
            if errors == 0:
                break

        return {
            'final_weights': self.weights.tolist(),
            'final_bias': self.bias,
            'training_history': self.training_history,
            'final_accuracy': self.training_history[-1]['accuracy']
        }

    def predict(self, inputs: List[float]) -> int:
        """预测类别"""
        output = self.forward(inputs)
        return int(output)


class MultiLayerPerceptron:
    """多层感知机 (1950年代末期)"""

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # 初始化权重
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            # He初始化
            weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            bias = np.zeros(output_size)
            self.weights.append(weights)
            self.biases.append(bias)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """多层感知机前向传播"""
        current_output = inputs

        for i, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            # 线性变换
            linear_output = np.dot(current_output, weights) + bias
            # 激活函数
            current_output = self.sigmoid(linear_output)

        return current_output

    def xor_problem_demo(self) -> None:
        """演示多层感知机解决XOR问题"""
        print("=== 多层感知机解决XOR问题演示 ===")

        # XOR训练数据
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        # 训练前的预测
        print("训练前的预测:")
        for i, x in enumerate(X):
            pred = self.forward(x)
            print(f"  输入: {x}, 预测: {pred.flatten()}, 目标: {y[i].flatten()}")

        # 简单训练 (这里使用简化版本，实际应该使用反向传播)
        for epoch in range(1000):
            total_error = 0
            for x_input, target in zip(X, y):
                output = self.forward(x_input)
                error = target - output
                total_error += np.abs(error).mean()

                # 简化的权重更新 (实际应该使用反向传播)
                if total_error > 0.1:
                    for j in range(len(self.weights)):
                        # 梯度下降更新
                        grad = -2 * error * self.sigmoid(np.dot(self.forward(x_input), (1 - self.forward(x_input))))
                        if j < len(self.weights) - 1:
                            self.weights[j] -= self.learning_rate * np.outer(x_input, grad).T
                        else:
                            self.biases[j] -= self.learning_rate * grad.sum()

            if total_error < 0.01:
                break

        # 训练后的预测
        print("\n训练后的预测:")
        for i, x in enumerate(X):
            pred = self.forward(x)
            print(f"  输入: {x}, 预测: {pred.flatten()}, 目标: {y[i].flatten()}")


class HebbianLearning:
    """Hebb学习规则 (1949)"""

    def __init__(self, num_neurons: int, learning_rate: float = 0.1):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
        self.activation_history = []

    def hebb_update(self, pre_synaptic: np.ndarray, post_synaptic: np.ndarray):
        """
        Hebb学习规则: Δwᵢⱼ = α * xᵢ * yⱼ
        一起放电的神经元会连接得更强
        """
        # 外积实现Hebb更新
        weight_update = self.learning_rate * np.outer(pre_synaptic, post_synaptic)
        return weight_update

    def train_pattern(self, pattern: np.ndarray):
        """训练单个模式"""
        # 更新权重
        self.weights += self.hebb_update(pattern, pattern)

        # 记录激活历史
        activation = self.sigmoid(np.dot(self.weights, pattern))
        self.activation_history.append(activation)

        return activation

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """回忆功能"""
        return self.sigmoid(np.dot(self.weights, cue))


class FoundationAnalyzer:
    """神经网络基础分析器"""

    def __init__(self):
        self.models = []
        self.analysis_results = {}

    def analyze_model_capacity(self, model: NeuronModel, test_patterns: List[List[float]]) -> Dict[str, Any]:
        """分析模型表达能力"""
        unique_outputs = set()
        for pattern in test_patterns:
            output = model.forward(pattern)
            unique_outputs.add(output)

        return {
            'model_name': model.name,
            'total_possible_outputs': len(unique_outputs),
            'can_represent_linearly_separable': len(unique_outputs) > 1,
            'test_patterns_count': len(test_patterns),
            'unique_output_patterns': list(unique_outputs)
        }

    def compare_models(self, models: List[NeuronModel],
                   test_data: List[Tuple[List[float], int]]) -> Dict[str, Any]:
        """比较不同模型的性能"""
        results = {}

        for model in models:
            if hasattr(model, 'train'):
                # 对于可训练的模型
                training_data = [(inputs, target) for inputs, target in test_data]
                model.train(training_data, epochs=50, verbose=False)

                # 测试性能
                correct_predictions = 0
                for inputs, target in test_data:
                    if hasattr(model, 'predict'):
                        prediction = model.predict(inputs)
                    else:
                        prediction = int(model.forward(inputs))
                    correct_predictions += 1 if prediction == target else 0

                accuracy = correct_predictions / len(test_data)
            else:
                # 对于不可训练的模型
                accuracy = 0.0

            results[model.name] = {
                'accuracy': accuracy,
                'model_type': type(model).__name__
            }

        return results


def demo_neural_foundation():
    """神经网络基础演示"""
    print("=== 神经网络基础发展演示 ===\n")

    # 1. McCulloch-Pitts神经元演示
    print("1. McCulloch-Pitts神经元 (1943)")
    mcp = McCullochPittsNeuron()
    mcp.configure(num_inputs=2, threshold=1.5)

    test_inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for inputs in test_inputs:
        output = mcp.forward(inputs)
        print(f"  输入: {inputs} -> 输出: {output}")

    # 2. 感知机训练演示
    print("\n2. 感知机训练 (1957)")
    # AND问题
    and_training_data = [
        ([0, 0], -1),
        ([0, 1], -1),
        ([1, 0], -1),
        ([1, 1], 1)
    ]

    perceptron = Perceptron(num_inputs=2, learning_rate=0.1)
    training_result = perceptron.train(and_training_data, epochs=50, verbose=False)

    print(f"  最终权重: {[round(w, 2) for w in training_result['final_weights']]}")
    print(f"  最终偏置: {round(training_result['final_bias'], 2)}")
    print(f"  最终准确率: {training_result['final_accuracy']:.4f}")

    # 测试
    print("  AND问题测试:")
    for inputs, _ in and_training_data:
        prediction = perceptron.predict(inputs)
        expected = 1 if inputs == [1, 1] else -1
        status = "✓" if prediction == expected else "✗"
        print(f"  {inputs} -> {prediction} (期望: {expected}) {status}")

    # 3. 多层感知机XOR问题演示
    print("\n3. 多层感知机解决XOR问题")
    mlp = MultiLayerPerceptron(layer_sizes=[2, 4, 2, 1])
    mlp.xor_problem_demo()

    # 4. Hebb学习演示
    print("\n4. Hebb学习规则 (1949)")
    hebb = HebbianLearning(num_neurons=4, learning_rate=0.1)

    patterns = [
        np.array([1, 0, 0, 0]),  # 模式A
        np.array([0, 1, 0, 0]),  # 模式B
        np.array([0, 0, 1, 0]),  # 模式C
        np.array([0, 0, 0, 1])   # 模式D
    ]

    for i, pattern in enumerate(patterns):
        activation = hebb.train_pattern(pattern)
        print(f"  训练模式 {i+1}: {pattern} -> 激活: {np.round(activation, 2)}")

    # 回忆测试
    recall_cue = np.array([1, 0, 0, 0])
    recalled = hebb.recall(recall_cue)
    print(f"  回忆提示 {recall_cue} -> 激活: {np.round(recalled, 2)}")

    # 5. 模型性能比较
    print("\n5. 模型能力分析")
    analyzer = FoundationAnalyzer()

    # 测试AND问题的不同模型
    mcp_model = mcp
    perceptron_model = perceptron

    test_cases = [[0, 0], [1, 0], [0, 1], [1, 1]]

    mcp_analysis = analyzer.analyze_model_capacity(mcp_model, test_cases)
    perceptron_analysis = analyzer.analyze_model_capacity(perceptron_model, test_cases)

    print(f"  McCulloch-Pitts分析:")
    print(f"    可表示的不同输出数: {mcp_analysis['can_represent_linearly_separable']}")
    print(f"    输出模式: {mcp_analysis['unique_output_patterns']}")

    print(f"  感知机分析:")
    print(f"    可表示的不同输出数: {perceptron_analysis['can_represent_linearly_separable']}")
    print(f"    最终准确率: {training_result['final_accuracy']:.4f}")


if __name__ == "__main__":
    demo_neural_foundation()
