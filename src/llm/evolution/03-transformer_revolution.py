"""
Transformer架构革命 - 注意力的时代 (2017年至今)

该模块实现了Transformer架构的核心组件，
包括注意力机制、位置编码、编码器-解码器结构。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class MultiHeadAttention(nn.Module):
    """多头注意力机制 (2017年核心创新)"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len, _ = q.size()

        # 线性变换：得到查询、键、值向量
        Q = self.W_q(q)  # (batch_size, seq_len, d_model)
        K = self.W_k(k)  # (batch_size, seq_len, d_model)
        V = self.W_v(v)  # (batch_size, seq_len, d_model)

        # 重塑为多头形式：(batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_v)

        # 计算注意力分数
        # Q · K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores形状: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            # 应用掩码（将被掩码的位置设为负无穷）
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 对最后一个维度（seq_len）进行softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        # 注意力权重与V相乘
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_v)

        # 合并多头：将num_heads维度移回d_model
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 最终线性变换
        output = self.W_o(output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码 - 为序列添加位置信息"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 计算每个位置的编码
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                if i % 2 == 0:
                    # 使用sin函数处理偶数位置
                    pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                else:
                    # 使用cos函数处理奇数位置
                    pe[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe = pe.unsqueeze(0).transpose(0, 1)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x形状: (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # 截取对应长度的位置编码
        positional_encoding = self.pe[:, :seq_len, :]

        # 添加位置编码
        x = x + positional_encoding
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer块 - 包含多头注意力和前馈网络"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()

        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # 层标准化和dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 自注意力 + 残差连接 + 层标准化
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x1 = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层标准化
        ff_output = self.feed_forward(x1)
        x2 = self.norm2(x1 + self.dropout(ff_output))

        return x2, attention_weights


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 编码器层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过所有编码器层
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 解码器层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                   tgt_mask: Optional[torch.Tensor] = None,
                   memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """前向传播"""
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, encoder_output, encoder_output, tgt_mask)
            all_attention_weights.append(attention_weights)

        # 最终线性变换
        output = self.fc_out(x)

        return output, all_attention_weights


class Transformer(nn.Module):
    """完整的Transformer模型 - 编码器-解码器架构"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):
        super(Transformer, self).__init__()

        # 编码器和解码器
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff, dropout
        )

        self.d_model = d_model

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """创建源序列掩码"""
        return (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 创建源序列掩码
        src_mask = self.create_mask(src)

        # 编码器
        encoder_output, encoder_attention = self.encoder(src, src_mask)

        # 解码器
        decoder_output, decoder_attention = self.decoder(tgt, encoder_output)

        return decoder_output, torch.cat([x for x in encoder_attention], dim=1)


class TransformerAnalyzer:
    """Transformer架构分析器"""

    def __init__(self):
        self.models = {}
        self.analysis_results = {}

    def analyze_attention_patterns(self, model: Transformer, input_text: str) -> Dict[str, Any]:
        """分析注意力模式"""
        print("=== Transformer注意力模式分析 ===")

        # 模拟输入处理
        tokens = input_text.split()
        num_tokens = len(tokens)

        return {
            'input_text': input_text,
            'num_tokens': num_tokens,
            'model_architecture': 'Encoder-Decoder Transformer',
            'attention_heads': model.decoder.layers[0].self_attention.num_heads,
            'expected_attention_patterns': [
                '局部依赖 (相邻词语)',
                '全局依赖 (远距离词语)',
                '语法结构 (主谓宾等)',
                '语义关联 (同义词、反义词)'
            ],
            'analyis_methods': [
                '计算注意力权重分布',
                '可视化注意力热力图',
                '分析不同层的注意力模式'
            ]
        }

    def compare_model_complexity(self, models: Dict[str, Transformer]) -> Dict[str, Any]:
        """比较模型复杂度"""
        print("=== Transformer模型复杂度比较 ===")

        results = {}
        for name, model in models.items():
            num_params = sum(p.numel() for p in model.parameters())

            # 计算注意力复杂度
            d_model = model.d_model
            num_heads = model.decoder.layers[0].self_attention.num_heads
            seq_length = 512  # 假设序列长度

            # 注意力计算复杂度: O(seq_length^2 * d_model)
            attention_complexity = seq_length ** 2 * d_model

            # 前馈网络复杂度: O(seq_length * d_model * d_ff)
            d_ff = model.decoder.layers[0].feed_forward[0].out_features
            ff_complexity = seq_length * d_model * d_ff

            results[name] = {
                'num_parameters': num_params,
                'attention_complexity': attention_complexity,
                'feedforward_complexity': ff_complexity,
                'model_depth': model.num_layers if hasattr(model, 'num_layers') else 'unknown',
                'd_model': d_model,
                'num_heads': num_heads
            }

        return results

    def analyze_training_efficiency(self, model: Transformer, batch_sizes: List[int]) -> Dict[str, Any]:
        """分析训练效率"""
        print("=== Transformer训练效率分析 ===")

        efficiency_results = {}

        for batch_size in batch_sizes:
            # 模拟前向传播
            dummy_input = torch.randint(0, 1000, (batch_size, 32))

            try:
                # 测试不同batch_size的内存使用
                with torch.no_grad():
                    output = model(dummy_input, dummy_input)

                # 计算梯度计算量
                total_gradients = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)

                efficiency_results[f'batch_size_{batch_size}'] = {
                    'success': True,
                    'output_shape': output.shape,
                    'memory_usage_estimate': f'{batch_size * 32 * 512 * 4 / (1024**3):.2f} MB',
                    'total_gradients': total_gradients
                }
            except RuntimeError as e:
                efficiency_results[f'batch_size_{batch_size}'] = {
                    'success': False,
                    'error': str(e),
                    'memory_limit': 'GPU内存不足'
                }

        return efficiency_results


def demo_transformer_architecture():
    """Transformer架构演示"""
    print("=== Transformer架构革命演示 ===")

    # 1. 多头注意力演示
    print("\n1. 多头注意力机制 (2017年核心创新)")
    mha = MultiHeadAttention(d_model=512, num_heads=8)

    # 创建模拟输入
    batch_size, seq_len, d_model = 2, 10, 512
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # 计算注意力
    output, attention_weights = mha(q, k, v)

    print(f"  输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"  注意力权重形状: {attention_weights.shape}")
    print(f"  输出形状: {output.shape}")

    # 2. 位置编码演示
    print("\n2. 位置编码")
    pe = PositionalEncoding(d_model=512, max_len=20)

    # 测试不同长度的序列
    for seq_len in [5, 10, 15]:
        x = torch.randn(2, seq_len, 512)
        output = pe(x)
        print(f"  序列长度 {seq_len}: 输入形状={x.shape}, 输出形状={output.shape}")

    # 3. 完整Transformer演示
    print("\n3. 完整Transformer模型")
    transformer = Transformer(
        src_vocab_size=1000, tgt_vocab_size=1000, d_model=512,
        num_heads=8, num_encoder_layers=6, num_decoder_layers=6
    )

    # 模拟翻译任务
    src = torch.randint(0, 1000, (2, 32))  # 英文句子
    tgt = torch.randint(0, 1000, (2, 32))  # 中文句子

    print(f"  源序列形状: {src.shape}")
    print(f"  目标序列形状: {tgt.shape}")

    # 前向传播
    output, attention_all = transformer(src, tgt)
    print(f"  输出形状: {output.shape}")

    # 4. 架构分析
    print("\n4. Transformer架构分析")
    analyzer = TransformerAnalyzer()

    # 分析注意力模式
    text_input = "Hello world, how are you today?"
    attention_analysis = analyzer.analyze_attention_patterns(transformer, text_input)
    print(f"  注意力分析: {attention_analysis['num_tokens']} 个token")
    print(f"  注意力头数: {attention_analysis['attention_heads']}")

    # 比较模型复杂度
    models = {
        'small_transformer': Transformer(1000, 1000, d_model=256, num_heads=8, num_encoder_layers=2, num_decoder_layers=2),
        'medium_transformer': Transformer(1000, 1000, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6),
        'large_transformer': Transformer(1000, 1000, d_model=1024, num_heads=16, num_encoder_layers=12, num_decoder_layers=12)
    }

    complexity_analysis = analyzer.compare_model_complexity(models)
    print("  模型复杂度比较:")
    for name, analysis in complexity_analysis.items():
        print(f"    {name}:")
        print(f"      参数数量: {analysis['num_parameters']:,}")
        print(f"      注意力复杂度: {analysis['attention_complexity']:,}")
        print(f"      前馈复杂度: {analysis['feedforward_complexity']:,}")

    # 训练效率分析
    batch_sizes = [8, 16, 32, 64, 128]
    efficiency_analysis = analyzer.analyze_training_efficiency(transformer, batch_sizes)
    print("  训练效率分析:")
    for batch_size, result in efficiency_analysis.items():
        status = "✓" if result['success'] else "✗"
        print(f"    Batch {batch_size}: {status}")
        if result['success']:
            print(f"      内存使用: {result['memory_usage_estimate']}")
        else:
            print(f"      错误: {result['error']}")

    print("\n=== Transformer架构革命演示完成 ===")

    return {
        'mha_result': {'output_shape': output.shape, 'attention_shape': attention_weights.shape},
        'pe_result': 'Positional encoding demonstrated for various sequence lengths',
        'transformer_demo': 'Full transformer model executed successfully',
        'complexity_analysis': complexity_analysis,
        'efficiency_analysis': efficiency_analysis
    }


if __name__ == "__main__":
    demo_transformer_architecture()
