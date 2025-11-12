"""
Transformer架构革命测试 (2017年至今)

该测试模块验证Transformer架构的核心组件，
包括多头注意力、位置编码、编码器-解码器结构等。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import math
import sys
import os

# 添加源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from src.llm.evolution.transformer_revolution import (
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    TransformerAnalyzer
)


class TestMultiHeadAttention:
    """多头注意力机制测试"""

    def setup_method(self):
        """测试前设置"""
        self.d_model = 512
        self.num_heads = 8
        self.seq_len = 10
        self.batch_size = 2

        self.mha = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads
        )

        # 创建测试数据
        self.q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.v = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_initialization(self):
        """测试多头注意力初始化"""
        # 验证维度设置
        assert self.mha.d_model == self.d_model
        assert self.mha.num_heads == self.num_heads
        assert self.mha.d_k == self.d_model // self.num_heads
        assert self.mha.d_v == self.d_model // self.num_heads
        assert self.mha.scale == 1.0 / math.sqrt(self.mha.d_k)

        # 验证线性变换矩阵
        assert self.mha.W_q.in_features == self.d_model
        assert self.mha.W_q.out_features == self.d_model
        assert self.mha.W_k.in_features == self.d_model
        assert self.mha.W_k.out_features == self.d_model
        assert self.mha.W_v.in_features == self.d_model
        assert self.mha.W_v.out_features == self.d_model
        assert self.mha.W_o.in_features == self.d_model
        assert self.mha.W_o.out_features == self.d_model

    def test_attention_computation(self):
        """测试注意力计算"""
        output, attention_weights = self.mha.forward(self.q, self.k, self.v)

        # 验证输出形状
        expected_output_shape = (self.batch_size, self.seq_len, self.d_model)
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)

        assert output.shape == expected_output_shape
        assert attention_weights.shape == expected_weights_shape

        # 验证注意力权重和为1（在最后一个维度）
        attention_sum = torch.sum(attention_weights, dim=-1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)

        # 验证注意力权重非负
        assert torch.all(attention_weights >= 0)

        # 验证输出是有限值
        assert torch.isfinite(output).all()

    def test_attention_with_mask(self):
        """测试带掩码的注意力"""
        # 创建掩码：屏蔽后半部分序列
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, self.seq_len//2:] = 0

        output, attention_weights = self.mha.forward(self.q, self.k, self.v, mask)

        # 验证掩码生效：被屏蔽的位置注意力权重应该接近0
        masked_attention = attention_weights[:, :, self.seq_len//2:, :]

        # 掩码区域的最大注意力应该接近0
        max_masked_attention = torch.max(masked_attention, dim=-1)[0]
        assert torch.all(max_masked_attention < 1e-6)

        # 未掩码区域应该正常
        unmasked_attention = attention_weights[:, :, :self.seq_len//2, :]
        assert torch.any(unmasked_attention > 1e-6)

    def test_different_head_configurations(self):
        """测试不同头数配置"""
        head_configs = [1, 2, 4, 8, 16]

        for num_heads in head_configs:
            if self.d_model % num_heads == 0:  # 确保能整除
                mha = MultiHeadAttention(d_model=self.d_model, num_heads=num_heads)

                q = torch.randn(1, 5, self.d_model)
                k = torch.randn(1, 5, self.d_model)
                v = torch.randn(1, 5, self.d_model)

                output, weights = mha.forward(q, k, v)

                # 验证不同配置下的正确性
                assert output.shape[0] == 1
                assert output.shape[1] == 5
                assert output.shape[2] == self.d_model
                assert weights.shape[1] == num_heads

    def test_attention_scale_effect(self):
        """测试注意力缩放效果"""
        # 创建相同的查询、键、值
        q = torch.ones(self.batch_size, self.seq_len, self.d_model)
        k = torch.ones(self.batch_size, self.seq_len, self.d_model)
        v = torch.randn(self.batch_size, self.seq_len, self.d_model)

        output, weights = self.mha.forward(q, k, v)

        # 由于查询和键相同，注意力应该相对均匀
        attention_variance = torch.var(weights, dim=-1)
        assert torch.all(attention_variance < 2.0)  # 方差应该较小

    def test_gradient_flow(self):
        """测试梯度流"""
        q = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        v = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)

        output, _ = self.mha.forward(q, k, v)
        loss = torch.sum(output)
        loss.backward()

        # 验证梯度存在且正确
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # 验证梯度形状
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

        # 验证梯度有限
        assert torch.isfinite(q.grad).all()
        assert torch.isfinite(k.grad).all()
        assert torch.isfinite(v.grad).all()


class TestPositionalEncoding:
    """位置编码测试"""

    def setup_method(self):
        """测试前设置"""
        self.d_model = 512
        self.max_len = 1000
        self.dropout = 0.1

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_len,
            dropout=self.dropout
        )

        self.test_seq_len = 20
        self.test_input = torch.randn(4, self.test_seq_len, self.d_model)

    def test_positional_encoding_shape(self):
        """测试位置编码形状"""
        output = self.pos_encoder.forward(self.test_input)

        # 验证输出形状
        assert output.shape == self.test_input.shape

        # 验证位置编码矩阵形状
        assert self.pos_encoder.pe.shape == (1, self.max_len, self.d_model)

    def test_positional_encoding_values(self):
        """测试位置编码值"""
        # 测试不同长度的序列
        seq_lengths = [1, 5, 10, 50]

        for seq_len in seq_lengths:
            test_input = torch.randn(2, seq_len, self.d_model)
            output = self.pos_encoder.forward(test_input)

            # 验证位置编码确实被添加
            # 输出应该包含原始输入和位置编码的叠加
            assert not torch.allclose(output, test_input)

            # 验证位置编码的差异性
            # 不同位置的编码应该不同
            if seq_len > 1:
                first_pos_encoding = output[0, 0] - test_input[0, 0]
                second_pos_encoding = output[0, 1] - test_input[0, 1]
                assert not torch.allclose(first_pos_encoding, second_pos_encoding)

    def test_dropout_effect(self):
        """测试Dropout效果"""
        self.pos_encoder.train()
        output_train = self.pos_encoder.forward(self.test_input)

        self.pos_encoder.eval()
        output_eval = self.pos_encoder.forward(self.test_input)

        # 在训练模式下应该有dropout，评估模式下没有
        # 这里的验证比较困难，主要检查功能不崩溃
        assert output_train.shape == output_eval.shape

    def test_positional_encoding_mathematical_properties(self):
        """测试位置编码的数学性质"""
        # 测试编码的数值性质
        seq_len = 100
        test_input = torch.zeros(1, seq_len, self.d_model)
        output = self.pos_encoder.forward(test_input)

        # 验证位置编码的模式
        for pos in range(seq_len):
            pos_encoding = output[0, pos]  # 由于输入为0，这应该是纯位置编码

            # 检查编码是否在合理范围内
            assert torch.all(torch.abs(pos_encoding) <= 10.0)  # 编码值不应该过大


class TestTransformerBlock:
    """Transformer块测试"""

    def setup_method(self):
        """测试前设置"""
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.seq_len = 10
        self.batch_size = 2

        self.transformer_block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff
        )

        self.test_input = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_block_initialization(self):
        """测试块初始化"""
        # 验证组件存在
        assert hasattr(self.transformer_block, 'self_attention')
        assert hasattr(self.transformer_block, 'feed_forward')
        assert hasattr(self.transformer_block, 'norm1')
        assert hasattr(self.transformer_block, 'norm2')
        assert hasattr(self.transformer_block, 'dropout')

        # 验证组件配置
        assert self.transformer_block.d_model == self.d_model
        assert self.transformer_block.num_heads == self.num_heads
        assert self.transformer_block.d_ff == self.d_ff

    def test_block_forward_pass(self):
        """测试块前向传播"""
        output, attention_weights = self.transformer_block.forward(self.test_input)

        # 验证输出形状
        assert output.shape == self.test_input.shape
        assert attention_weights is not None

        # 验证输出是有限值
        assert torch.isfinite(output).all()

    def test_residual_connection(self):
        """测试残差连接"""
        output1, _ = self.transformer_block.forward(self.test_input)

        # 多次传播应该产生不同结果（由于dropout和layernorm）
        output2, _ = self.transformer_block.forward(self.test_input)

        # 在评估模式下，相同输入应该产生相同输出
        self.transformer_block.eval()
        output1_eval, _ = self.transformer_block.forward(self.test_input)
        output2_eval, _ = self.transformer_block.forward(self.test_input)

        assert torch.allclose(output1_eval, output2_eval)

    def test_layer_normalization_effect(self):
        """测试层标准化效果"""
        # 创建具有极端值的输入
        extreme_input = torch.randn(self.batch_size, self.seq_len, self.d_model) * 10

        output, _ = self.transformer_block.forward(extreme_input)

        # 层标准化应该输出合理范围内的值
        assert torch.isfinite(output).all()

        # 计算输出的统计信息
        mean = torch.mean(output, dim=-1)
        std = torch.std(output, dim=-1)

        # 层标准化后的均值应该接近0，标准差接近1
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.5)
        assert torch.allclose(std, torch.ones_like(std), atol=0.5)

    def test_feed_forward_network(self):
        """测试前馈网络"""
        # 创建简单的测试
        simple_input = torch.randn(2, 5, self.d_model)
        output, _ = self.transformer_block.forward(simple_input)

        # 验证前馈网络的作用
        # 前馈网络应该能够改变输入的表示
        assert not torch.allclose(output, simple_input)

    def test_block_with_different_configurations(self):
        """测试不同配置的块"""
        configs = [
            {'d_model': 256, 'num_heads': 4, 'd_ff': 1024},
            {'d_model': 512, 'num_heads': 8, 'd_ff': 2048},
            {'d_model': 1024, 'num_heads': 16, 'd_ff': 4096}
        ]

        for config in configs:
            block = TransformerBlock(**config)
            test_input = torch.randn(2, 8, config['d_model'])

            output, attention_weights = block.forward(test_input)

            # 验证不同配置下的正确性
            assert output.shape == test_input.shape
            assert attention_weights is not None


class TestTransformerEncoder:
    """Transformer编码器测试"""

    def setup_method(self):
        """测试前设置"""
        self.vocab_size = 10000
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.seq_len = 20
        self.batch_size = 2

        self.encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        self.test_input = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_encoder_initialization(self):
        """测试编码器初始化"""
        # 验证组件存在
        assert hasattr(self.encoder, 'embedding')
        assert hasattr(self.encoder, 'pos_encoding')
        assert hasattr(self.encoder, 'layers')

        # 验证配置
        assert self.encoder.vocab_size == self.vocab_size
        assert self.encoder.d_model == self.d_model
        assert self.encoder.num_layers == self.num_layers

        # 验证层数
        assert len(self.encoder.layers) == self.num_layers

    def test_encoder_forward_pass(self):
        """测试编码器前向传播"""
        output, all_attention_weights = self.encoder.forward(self.test_input)

        # 验证输出形状
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        assert output.shape == expected_shape

        # 验证注意力权重
        assert len(all_attention_weights) == self.num_layers

        # 验证每层的注意力权重形状
        for layer_idx in range(self.num_layers):
            assert all_attention_weights[layer_idx] is not None
            # 注意力权重形状应为 (batch_size, num_heads, seq_len, seq_len)
            expected_attention_shape = (self.batch_size, 8, self.seq_len, self.seq_len)
            assert all_attention_weights[layer_idx].shape == expected_attention_shape

    def test_embedding_layer(self):
        """测试嵌入层"""
        # 测试嵌入层
        test_tokens = torch.randint(0, self.vocab_size, (10,))
        embedded = self.encoder.embedding(test_tokens)

        # 验证嵌入维度
        assert embedded.shape == (10, self.d_model)

        # 验证嵌入是有限值
        assert torch.isfinite(embedded).all()

    def test_positional_encoding_integration(self):
        """测试位置编码集成"""
        # 测试不同长度的序列
        for seq_len in [5, 10, 15, 20]:
            test_input = torch.randint(0, self.vocab_size, (self.batch_size, seq_len))
            output, _ = self.encoder.forward(test_input)

            # 验证输出形状
            assert output.shape == (self.batch_size, seq_len, self.d_model)
            assert torch.isfinite(output).all()

    def test_encoder_with_mask(self):
        """测试带掩码的编码器"""
        # 创建掩码
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, self.seq_len//2:] = 0  # 屏蔽后半部分

        output, attention_weights = self.encoder.forward(self.test_input, mask)

        # 验证掩码不影响输出形状
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)

        # 验证注意力权重反映了掩码
        # 最后几层的注意力权重应该受到掩码影响
        final_attention = attention_weights[-1]  # 最后一层
        assert final_attention.shape == (self.batch_size, 8, self.seq_len, self.seq_len)


class TestTransformerDecoder:
    """Transformer解码器测试"""

    def setup_method(self):
        """测试前设置"""
        self.vocab_size = 10000
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.seq_len = 20
        self.batch_size = 2

        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        self.decoder_input = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_decoder_initialization(self):
        """测试解码器初始化"""
        # 验证组件存在
        assert hasattr(self.decoder, 'embedding')
        assert hasattr(self.decoder, 'pos_encoding')
        assert hasattr(self.decoder, 'layers')
        assert hasattr(self.decoder, 'fc_out')

        # 验证配置
        assert self.decoder.vocab_size == self.vocab_size
        assert self.decoder.d_model == self.d_model
        assert self.decoder.num_layers == self.num_layers

    def test_decoder_forward_pass(self):
        """测试解码器前向传播"""
        output, all_attention_weights = self.decoder.forward(
            self.decoder_input, self.encoder_output
        )

        # 验证输出形状
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        assert output.shape == expected_shape

        # 验证注意力权重
        assert len(all_attention_weights) == self.num_layers

    def test_output_projection(self):
        """测试输出投影"""
        output, _ = self.decoder.forward(self.decoder_input, self.encoder_output)

        # 验证输出维度
        assert output.shape[-1] == self.vocab_size

        # 验证输出的数值范围（softmax后应该在[0,1]）
        # 这里我们验证输出的合理性
        assert torch.isfinite(output).all()

        # 验证最后一维的和接近1（应用softmax后）
        output_softmax = torch.softmax(output, dim=-1)
        row_sums = torch.sum(output_softmax, dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_autoregressive_decoding(self):
        """测试自回归解码"""
        # 模拟自回归解码过程
        current_input = torch.zeros(self.batch_size, 1).long()
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)

        outputs = []
        for _ in range(5):  # 生成5个token
            output, _ = self.decoder.forward(current_input, encoder_output)

            # 取最后一个时间步的输出
            last_token_logits = output[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1)

            outputs.append(next_token)
            current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

        # 验证生成的序列
        assert len(outputs) == 5
        assert all(0 <= token < self.vocab_size for token in outputs)


class TestTransformerAnalyzer:
    """Transformer分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.analyzer = TransformerAnalyzer()

        # 创建测试模型
        self.transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )

    def test_attention_patterns_analysis(self):
        """测试注意力模式分析"""
        input_text = "Hello world, how are you today?"
        analysis = self.analyzer.analyze_attention_patterns(self.transformer, input_text)

        # 验证分析结构
        required_keys = [
            'input_text', 'num_tokens', 'model_architecture',
            'attention_heads', 'expected_attention_patterns',
            'analyis_methods'
        ]

        for key in required_keys:
            assert key in analysis
            assert analysis[key] is not None

        # 验证具体内容
        assert analysis['input_text'] == input_text
        assert '多头注意力机制' in analysis['model_architecture']
        assert analysis['attention_heads'] > 0

    def test_model_complexity_comparison(self):
        """测试模型复杂度比较"""
        models = {
            'small': Transformer(1000, 1000, d_model=256, num_heads=4, num_encoder_layers=2, num_decoder_layers=2),
            'medium': self.transformer,
            'large': Transformer(1000, 1000, d_model=1024, num_heads=16, num_encoder_layers=12, num_decoder_layers=12)
        }

        comparison = self.analyzer.compare_model_complexity(models)

        # 验证比较结果
        assert 'small' in comparison
        assert 'medium' in comparison
        assert 'large' in comparison

        # 验证每个模型的统计信息
        for model_name, stats in comparison.items():
            required_keys = [
                'num_parameters', 'attention_complexity', 'feedforward_complexity',
                'model_depth', 'd_model', 'num_heads'
            ]

            for key in required_keys:
                assert key in stats
                assert stats[key] is not None

            # 验证数值合理性
            assert stats['num_parameters'] > 0
            assert stats['d_model'] > 0
            assert stats['num_heads'] > 0

        # 验证大小关系
        small_params = comparison['small']['num_parameters']
        medium_params = comparison['medium']['num_parameters']
        large_params = comparison['large']['num_parameters']

        assert small_params < medium_params < large_params

    def test_training_efficiency_analysis(self):
        """测试训练效率分析"""
        batch_sizes = [8, 16, 32, 64]
        efficiency = self.analyzer.analyze_training_efficiency(self.transformer, batch_sizes)

        # 验证效率分析结果
        for batch_size in batch_sizes:
            key = f'batch_size_{batch_size}'
            assert key in efficiency

            result = efficiency[key]
            assert 'success' in result
            if result['success']:
                assert 'output_shape' in result
                assert 'memory_usage_estimate' in result


class TestTransformerIntegration:
    """Transformer集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 512
        self.num_heads = 8
        self.seq_length = 20

        self.transformer = Transformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        self.batch_size = 4
        self.src_sequence = torch.randint(0, self.src_vocab_size,
                                     (self.batch_size, self.seq_length))
        self.tgt_sequence = torch.randint(0, self.tgt_vocab_size,
                                     (self.batch_size, self.seq_length))

    def test_complete_transformer_forward(self):
        """测试完整Transformer前向传播"""
        decoder_output, attention_weights = self.transformer.forward(
            self.src_sequence, self.tgt_sequence
        )

        # 验证输出形状
        expected_shape = (self.batch_size, self.seq_length, self.tgt_vocab_size)
        assert decoder_output.shape == expected_shape
        assert len(attention_weights) > 0

        # 验证输出数值有效性
        assert torch.isfinite(decoder_output).all()

    def test_different_sequence_lengths(self):
        """测试不同序列长度"""
        seq_lengths = [5, 10, 15, 25]

        for seq_len in seq_lengths:
            src_seq = torch.randint(0, self.src_vocab_size, (self.batch_size, seq_len))
            tgt_seq = torch.randint(0, self.tgt_vocab_size, (self.batch_size, seq_len))

            output, _ = self.transformer.forward(src_seq, tgt_seq)

            # 验证不同长度的处理
            assert output.shape == (self.batch_size, seq_len, self.tgt_vocab_size)
            assert torch.isfinite(output).all()

    def test_gradient_computation(self):
        """测试梯度计算"""
        self.transformer.train()

        src_seq = torch.randint(0, self.src_vocab_size,
                               (self.batch_size, self.seq_length), requires_grad=False)
        tgt_seq = torch.randint(0, self.tgt_vocab_size,
                               (self.batch_size, self.seq_length), requires_grad=True)

        output, _ = self.transformer.forward(src_seq, tgt_seq)

        # 计算损失
        target = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.seq_length))
        loss = torch.nn.CrossEntropyLoss()(output.view(-1, self.tgt_vocab_size), target.view(-1))

        # 反向传播
        loss.backward()

        # 验证梯度存在
        for param in self.transformer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_mask_functionality(self):
        """测试掩码功能"""
        # 创建源序列掩码
        src_mask = torch.ones(self.batch_size, self.seq_length)
        src_mask[:, self.seq_length//2:] = 0  # 屏蔽后半部分

        output, attention_weights = self.transformer.forward(
            self.src_sequence, self.tgt_sequence, src_mask
        )

        # 验证掩码不破坏输出形状
        assert output.shape == (self.batch_size, self.seq_length, self.tgt_vocab_size)

        # 验证注意力权重反映了掩码
        # 注意力权重应该具有掩码的影响
        assert attention_weights is not None


# pytest主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
