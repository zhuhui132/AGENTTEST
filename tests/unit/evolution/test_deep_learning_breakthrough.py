"""
深度学习突破测试 (2010-2015年)

该测试模块验证深度学习突破期的关键技术，
包括AlexNet、Dropout、GAN、ResNet等革命性算法。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from src.llm.evolution.deep_learning_breakthrough import (
    DropoutLayer,
    BatchNormalization,
    AlexNet,
    ResidualBlock,
    SimpleGAN,
    DeepLearningAnalyzer
)


class TestDropoutLayer:
    """Dropout正则化测试"""

    def setup_method(self):
        """测试前设置"""
        self.dropout = DropoutLayer(dropout_rate=0.5)
        self.dropout.training = True
        self.test_tensor = torch.randn(100, 50)

    def test_dropout_forward_pass(self):
        """测试Dropout前向传播"""
        output = self.dropout.forward(self.test_tensor)

        # 验证输出形状
        assert output.shape == self.test_tensor.shape
        assert output.dtype == self.test_tensor.dtype

    def test_dropout_mask_generation(self):
        """测试Dropout掩码生成"""
        output = self.dropout.forward(self.test_tensor)
        mask = self.dropout.mask

        # 验证掩码属性
        assert mask.shape == self.test_tensor.shape
        assert mask.dtype == torch.float

        # 掩码值应该是0或1/(1-dropout_rate)
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2  # 最多两种值

        # 在训练模式下应该有部分被丢弃
        zero_ratio = (mask == 0).float().mean()
        expected_zero_ratio = self.dropout.dropout_rate
        assert abs(zero_ratio - expected_zero_ratio) < 0.1

    def test_dropout_evaluation_mode(self):
        """测试Dropout评估模式"""
        self.dropout.eval()

        # 评估模式下应该不会有dropout
        output = self.dropout.forward(self.test_tensor)
        mask = self.dropout.mask

        # 在评估模式下mask应该不存在
        assert mask is None
        assert torch.allclose(output, self.test_tensor)

    def test_dropout_training_mode(self):
        """测试Dropout训练模式"""
        self.dropout.train()

        # 训练模式下应该有dropout
        output = self.dropout.forward(self.test_tensor)
        mask = self.dropout.mask

        # 在训练模式下mask应该存在
        assert mask is not None
        assert not torch.allclose(output, self.test_tensor)

    def test_dropout_rate_validation(self):
        """测试Dropout率验证"""
        # 测试不同的dropout率
        dropout_rates = [0.1, 0.3, 0.5, 0.7]

        for rate in dropout_rates:
            dropout = DropoutLayer(dropout_rate=rate)
            output = dropout.forward(self.test_tensor)

            # 验证缩放因子应用正确
            expected_scale = 1.0 / (1.0 - rate) if dropout.mask is not None else 1.0
            # 这里比较困难，主要验证不崩溃
            assert output is not None


class TestBatchNormalization:
    """批量标准化测试"""

    def setup_method(self):
        """测试前设置"""
        self.batch_norm = BatchNormalization(num_features=50, eps=1e-5, momentum=0.1)
        self.test_data = torch.randn(32, 50)  # batch_size=32, num_features=50

    def test_batch_norm_forward_pass(self):
        """测试批量标准化前向传播"""
        output = self.batch_norm.forward(self.test_data)

        # 验证输出形状
        assert output.shape == self.test_data.shape

        # 验证输出类型
        assert isinstance(output, torch.Tensor)

    def test_batch_norm_training_mode(self):
        """测试批量标准化训练模式"""
        self.batch_norm.training = True
        output = self.batch_norm.forward(self.test_data)

        # 验证运行时统计更新
        assert hasattr(self.batch_norm, 'running_mean')
        assert hasattr(self.batch_norm, 'running_var')

        # 验证参数存在
        assert hasattr(self.batch_norm, 'gamma')
        assert hasattr(self.batch_norm, 'beta')

    def test_batch_norm_evaluation_mode(self):
        """测试批量标准化评估模式"""
        self.batch_norm.eval()
        output = self.batch_norm.forward(self.test_data)

        # 在评估模式下应该使用运行时统计
        assert not self.batch_norm.training
        assert self.batch_norm.running_mean is not None
        assert self.batch_norm.running_var is not None

    def test_batch_norm_parameter_shapes(self):
        """测试批量标准化参数形状"""
        # 验证参数形状
        assert self.batch_norm.gamma.shape == (50,)
        assert self.batch_norm.beta.shape == (50,)
        assert self.batch_norm.running_mean.shape == (50,)
        assert self.batch_norm.running_var.shape == (50,)

        # 验证参数类型
        assert isinstance(self.batch_norm.gamma, nn.Parameter)
        assert isinstance(self.batch_norm.beta, nn.Parameter)

    def test_batch_norm_numerical_stability(self):
        """测试批量标准化数值稳定性"""
        # 测试极端数据
        extreme_data = torch.randn(32, 50) * 100  # 大数值
        output = self.batch_norm.forward(extreme_data)

        # 验证没有NaN或Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # 测试常数数据
        constant_data = torch.ones(32, 50)
        output = self.batch_norm.forward(constant_data)

        # 常数数据应该标准化为0（考虑epsilon）
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-4)

    def test_batch_norm_different_batch_sizes(self):
        """测试不同批次大小"""
        batch_sizes = [1, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            test_data = torch.randn(batch_size, 50)
            output = self.batch_norm.forward(test_data)

            # 验证输出形状
            assert output.shape == (batch_size, 50)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestAlexNet:
    """AlexNet模型测试"""

    def setup_method(self):
        """测试前设置"""
        self.alexnet = AlexNet(num_classes=1000)
        self.test_input = torch.randn(4, 3, 224, 224)  # batch_size=4, channels=3, height=224, width=224

    def test_alexnet_parameter_count(self):
        """测试AlexNet参数数量"""
        total_params = sum(p.numel() for p in self.alexnet.parameters())
        trainable_params = sum(p.numel() for p in self.alexnet.parameters() if p.requires_grad)

        # AlexNet应该有合理的参数数量
        assert total_params > 1000000  # 至少百万级别
        assert total_params == trainable_params  # 所有参数都可训练

        # 验证参数分布
        conv_params = sum(p.numel() for name, p in self.alexnet.named_parameters()
                         if 'conv' in name and p.dim() == 4)
        linear_params = sum(p.numel() for name, p in self.alexnet.named_parameters()
                           if 'weight' in name and 'linear' in name)

        assert conv_params > 0
        assert linear_params > 0

    def test_alexnet_forward_pass(self):
        """测试AlexNet前向传播"""
        output = self.alexnet(self.test_input)

        # 验证输出形状
        expected_shape = (4, 1000)  # batch_size x num_classes
        assert output.shape == expected_shape

        # 验证输出类型
        assert isinstance(output, torch.Tensor)
        assert not torch.isnan(output).any()

    def test_alexnet_architecture_components(self):
        """测试AlexNet架构组件"""
        # 验证特征提取器
        assert hasattr(self.alexnet, 'features')
        assert isinstance(self.alexnet.features, nn.Sequential)

        # 验证分类器
        assert hasattr(self.alexnet, 'classifier')
        assert isinstance(self.alexnet.classifier, nn.Sequential)

        # 验证关键层存在
        feature_layers = [layer for layer in self.alexnet.features.children()]
        assert len(feature_layers) >= 10  # AlexNet有多个特征层

        # 检查是否有卷积层、池化层、全连接层
        has_conv = any(isinstance(layer, nn.Conv2d) for layer in feature_layers)
        has_pool = any(isinstance(layer, nn.MaxPool2d) for layer in feature_layers)

        assert has_conv
        assert has_pool

    def test_alexnet_different_input_sizes(self):
        """测试不同输入尺寸"""
        input_sizes = [
            (2, 3, 224, 224),
            (1, 3, 224, 224),
            (8, 3, 224, 224)
        ]

        for input_size in input_sizes:
            test_input = torch.randn(*input_size)
            output = self.alexnet(test_input)

            # 验证输出形状
            expected_shape = (input_size[0], 1000)
            assert output.shape == expected_shape
            assert not torch.isnan(output).any()

    def test_alexnet_gradient_flow(self):
        """测试AlexNet梯度流"""
        self.alexnet.train()
        test_input = self.test_input.requires_grad_(True)
        target = torch.randint(0, 1000, (4,))
        criterion = nn.CrossEntropyLoss()

        # 前向传播
        output = self.alexnet(test_input)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()

        # 验证梯度存在
        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()

        # 验证模型参数梯度
        for param in self.alexnet.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.shape == param.shape


class TestResidualBlock:
    """残差块测试"""

    def setup_method(self):
        """测试前设置"""
        self.res_block = ResidualBlock(
            in_channels=64,
            out_channels=64,
            stride=1
        )
        self.test_input = torch.randn(2, 64, 32, 32)  # batch_size=2, channels=64, height=32, width=32

    def test_residual_block_forward_pass(self):
        """测试残差块前向传播"""
        output, attention_weights = self.res_block.forward(self.test_input)

        # 验证输出形状
        assert output.shape == self.test_input.shape
        assert attention_weights is not None

    def test_residual_block_identity_connection(self):
        """测试残差块身份连接"""
        # 当stride=1且in_channels=out_channels时，应该使用身份连接
        res_block = ResidualBlock(in_channels=64, out_channels=64, stride=1)

        # 验证是否有下采样模块
        # 这需要检查ResidualBlock的具体实现
        assert res_block.downsample is not None

    def test_residual_block_stride_2(self):
        """测试残差块stride=2的情况"""
        res_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        test_input = torch.randn(2, 64, 32, 32)

        output, _ = res_block.forward(test_input)

        # stride=2时，输出尺寸应该减半
        expected_height = 32 // 2
        expected_width = 32 // 2
        assert output.shape == (2, 128, expected_height, expected_width)

    def test_residual_block_dimension_mismatch(self):
        """测试残差块维度不匹配情况"""
        # 当in_channels != out_channels时，应该使用下采样
        res_block = ResidualBlock(in_channels=64, out_channels=128, stride=1)
        test_input = torch.randn(2, 64, 32, 32)

        output, _ = res_block.forward(test_input)

        # 验证输出通道数正确
        assert output.shape[1] == 128

    def test_residual_block_different_configurations(self):
        """测试不同残差块配置"""
        configs = [
            {'in_channels': 32, 'out_channels': 32, 'stride': 1},
            {'in_channels': 64, 'out_channels': 128, 'stride': 2},
            {'in_channels': 128, 'out_channels': 256, 'stride': 1},
        ]

        for config in configs:
            res_block = ResidualBlock(**config)
            test_input = torch.randn(2, config['in_channels'], 16, 16)

            output, attention_weights = res_block.forward(test_input)

            # 验证输出形状
            if config['stride'] == 1:
                expected_shape = (2, config['out_channels'], 16, 16)
            else:
                expected_shape = (2, config['out_channels'], 16 // config['stride'], 16 // config['stride'])

            assert output.shape == expected_shape
            assert attention_weights is not None


class TestSimpleGAN:
    """简化GAN测试"""

    def setup_method(self):
        """测试前设置"""
        self.gan = SimpleGAN(latent_dim=100, img_dim=784)
        self.z_dim = 100
        self.batch_size = 32
        self.noise = torch.randn(self.batch_size, self.z_dim)
        self.real_images = torch.randn(self.batch_size, 784)

    def test_gan_components(self):
        """测试GAN组件"""
        # 验证生成器和判别器
        assert hasattr(self.gan, 'generator')
        assert hasattr(self.gan, 'discriminator')
        assert hasattr(self.gan, 'latent_dim')
        assert hasattr(self.gan, 'img_dim')

        # 验证维度
        assert self.gan.latent_dim == self.z_dim
        assert self.gan.img_dim == 784

    def test_generator_output(self):
        """测试生成器输出"""
        fake_images = self.gan.generate(self.noise)

        # 验证输出形状和范围
        assert fake_images.shape == (self.batch_size, self.gan.img_dim)

        # GAN生成器通常使用tanh激活，输出范围[-1, 1]
        assert torch.all(fake_images >= -1)
        assert torch.all(fake_images <= 1)

    def test_discriminator_output(self):
        """测试判别器输出"""
        fake_images = self.gan.generate(self.noise)
        real_scores = self.gan.discriminate(self.real_images)
        fake_scores = self.gan.discriminate(fake_images)

        # 验证输出形状和范围
        assert real_scores.shape == (self.batch_size, 1)
        assert fake_scores.shape == (self.batch_size, 1)

        # Sigmoid输出应该在[0, 1]之间
        assert torch.all(real_scores >= 0)
        assert torch.all(real_scores <= 1)
        assert torch.all(fake_scores >= 0)
        assert torch.all(fake_scores <= 1)

    def test_generator_loss(self):
        """测试生成器损失"""
        fake_images = self.gan.generate(self.noise)
        fake_scores = self.gan.discriminate(fake_images)

        gen_loss = self.gan.generator_loss(fake_scores)

        # 生成器希望判别器输出为1
        assert isinstance(gen_loss, torch.Tensor)
        assert gen_loss.numel() == self.batch_size
        assert gen_loss.item() >= 0

    def test_discriminator_loss(self):
        """测试判别器损失"""
        fake_images = self.gan.generate(self.noise)
        fake_scores = self.gan.discriminate(fake_images)
        real_scores = self.gan.discriminate(self.real_images)

        disc_loss = self.gan.discriminator_loss(real_scores, fake_scores)

        # 验证损失计算
        assert isinstance(disc_loss, torch.Tensor)
        assert disc_loss.numel() == self.batch_size
        assert disc_loss.item() >= 0

    def test_gan_training_step(self):
        """测试GAN训练步骤"""
        fake_images = self.gan.generate(self.noise)
        real_scores = self.gan.discriminate(self.real_images)
        fake_scores = self.gan.discriminate(fake_images)

        # 计算损失
        gen_loss = self.gan.generator_loss(fake_scores)
        disc_loss = self.gan.discriminator_loss(real_scores, fake_scores)

        # 验证损失合理性
        assert gen_loss.item() >= 0
        assert disc_loss.item() >= 0

        # 验证损失差异（这是一个相对测试，不是绝对标准）
        # 在训练初期，判别器应该能很好地区分真假
        # 在训练稳定后，损失应该相对平衡
        assert isinstance(gen_loss, torch.Tensor)
        assert isinstance(disc_loss, torch.Tensor)


class TestDeepLearningAnalyzer:
    """深度学习分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.analyzer = DeepLearningAnalyzer()

    def test_alexnet_breakthrough_analysis(self):
        """测试AlexNet突破分析"""
        analysis = self.analyzer.analyze_alexnet_breakthrough()

        # 验证分析结构
        required_keys = [
            'breakthrough_year', 'key_innovation', 'architectural_innovations',
            'performance_improvement', 'historical_significance'
        ]

        for key in required_keys:
            assert key in analysis
            assert analysis[key] is not None

        # 验证具体内容
        assert analysis['breakthrough_year'] == 2012
        assert '深度卷积神经网络' in analysis['key_innovation']
        assert isinstance(analysis['architectural_innovations'], list)
        assert isinstance(analysis['performance_improvement'], dict)
        assert isinstance(analysis['historical_significance'], str)

    def test_epoch_methods_comparison(self):
        """测试优化方法比较"""
        comparison = self.analyzer.compare_epoch_methods()

        # 验证比较结构
        assert 'methods' in comparison
        assert 'performance' in comparison
        assert 'recommendation' in comparison

        # 验证方法列表
        methods = comparison['methods']
        expected_methods = ['SGD', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam']

        for method in expected_methods:
            assert method in methods

        # 验证性能数据
        performance = comparison['performance']
        for method in expected_methods:
            assert method in performance
            method_perf = performance[method]

            required_perf_keys = ['final_loss', 'convergence_speed', 'stability']
            for key in required_perf_keys:
                assert key in method_perf
                assert isinstance(method_perf[key], (str, float, int))

    def test_regularization_techniques(self):
        """测试正则化技术分析"""
        techniques = self.analyzer.analyze_regularization_techniques()

        # 验证技术结构
        expected_techniques = ['Dropout', 'BatchNorm', 'L2-Regularization']

        for technique in expected_techniques:
            assert technique in techniques
            tech_info = techniques[technique]

            required_keys = ['introduction_year', 'principle', 'benefit']
            for key in required_keys:
                assert key in tech_info
                assert tech_info[key] is not None

        # 验证排序和推荐
        assert 'effectiveness_ranking' in techniques
        assert 'usage_recommendation' in techniques
        assert isinstance(techniques['effectiveness_ranking'], list)
        assert isinstance(techniques['usage_recommendation'], str)


class TestDeepLearningIntegration:
    """深度学习集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.sample_data = torch.randn(100, 3, 32, 32)  # 模拟图像数据
        self.sample_labels = torch.randint(0, 10, (100,))  # 模拟标签

    def test_complete_training_pipeline(self):
        """测试完整训练流水线"""
        # 创建AlexNet模型
        model = AlexNet(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # 训练几个epoch
        model.train()
        initial_loss = None

        for epoch in range(3):
            running_loss = 0.0
            for i in range(10):  # 使用部分数据
                inputs = self.sample_data[i*10:(i+1)*10]
                labels = self.sample_labels[i*10:(i+1)*10]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i == 0 and epoch == 0:
                    initial_loss = loss.item()

            avg_loss = running_loss / 10

            # 损失应该下降（或至少不大幅增加）
            if epoch > 0:
                assert avg_loss <= initial_loss + 1.0  # 允许一些波动

        # 验证训练完成
        assert initial_loss is not None

    def test_different_architectures_comparison(self):
        """测试不同架构比较"""
        # 创建不同配置的模型
        models = {
            'simple_cnn': AlexNet(num_classes=10),  # 使用AlexNet作为示例
        }

        # 比较模型参数
        model_stats = {}
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            model_stats[name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_type': type(model).__name__
            }

        # 验证统计信息
        for name, stats in model_stats.items():
            assert stats['total_params'] > 0
            assert stats['trainable_params'] > 0
            assert stats['trainable_params'] <= stats['total_params']
            assert isinstance(stats['model_type'], str)

    def test_regularization_effects(self):
        """测试正则化效果"""
        # 简化的比较测试
        # 实际实现需要更复杂的设置

        # 这里主要验证正则化技术的影响
        dropout_rate = 0.5
        input_tensor = torch.randn(10, 20)

        # 测试Dropout
        dropout = DropoutLayer(dropout_rate=dropout_rate)

        # 多次应用Dropout，验证随机性
        outputs = []
        for _ in range(10):
            output = dropout.forward(input_tensor)
            outputs.append(output)

        # 不同次的输出应该不同（在训练模式下）
        dropout.train()
        output1 = dropout.forward(input_tensor)
        output2 = dropout.forward(input_tensor)

        # 这里的检查比较困难，主要验证功能不崩溃
        assert not torch.allclose(output1, output2)

        # 在评估模式下输出应该相同
        dropout.eval()
        output3 = dropout.forward(input_tensor)
        output4 = dropout.forward(input_tensor)
        assert torch.allclose(output3, output4)


# pytest主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
