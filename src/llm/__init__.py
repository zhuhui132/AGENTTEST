"""
大语言模型模块 - 按发展历程组织

该模块按照LLM发展的四个阶段组织：
1. 神经网络基础 (1940s-1957s)
2. 深度学习突破 (2010s-2015s)
3. Transformer革命 (2017年至今)
4. 大模型时代 (2020年至今)

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

from . import base
from . import manager

# 发展历程模块
from .evolution import (
    neural_networks_foundation,
    deep_learning_breakthrough,
    transformer_revolution,
    large_language_models
)

# 架构技术模块 (待实现)
# from .architecture import (
#     transformer_details,
#     attention_mechanism,
#     positional_encoding,
#     model_parallelism
# )

# 应用技术模块 (待实现)
# from .applications import (
#     dialogue_systems,
#     text_generation,
#     code_generation,
#     multimodal_understanding
# )

# 训练技术模块 (待实现)
# from .training import (
#     pretraining_strategies,
#     fine_tuning_techniques,
#     rlhf_training,
#     distributed_training
# )

# 伦理安全模块 (待实现)
# from .ethics import (
#     content_safety,
#     bias_detection,
#     privacy_protection,
#     interpretability
# )

# 未来发展模块 (待实现)
# from .future import (
#     agi_roadmap,
#     computational_optimization,
#     technology_fusion,
#     industrialization_path
# )

# 保留原有模块
from . import huggingface_llm
from . import openai_llm
from . import mock_llm

# 版本信息
__version__ = "1.0.0"
__author__ = "AI开发团队"
__description__ = "大语言模型模块 - 按发展历程组织"

# 导出的类和函数
__all__ = [
    # 基础类
    'BaseLLM',
    'LLMManager',

    # 发展历程模块
    'neural_networks_foundation',
    'deep_learning_breakthrough',
    'transformer_revolution',
    'large_language_models',

    # 原有模块
    'OpenAILLM',
    'HuggingFaceLLM',
    'MockLLM',

    # 版本信息
    '__version__',
    '__author__',
    '__description__'
]

# 模块配置
LLM_CONFIG = {
    'development_stages': {
        'neural_foundation': {
            'period': '1940s-1957s',
            'key_figures': ['McCulloch', 'Pitts', 'Rosenblatt'],
            'core_concepts': ['neuron_model', 'perceptron', 'learning_algorithm']
        },
        'deep_learning_breakthrough': {
            'period': '2010s-2015s',
            'key_figures': ['Krizhevsky', 'Hinton', 'Goodfellow'],
            'core_concepts': ['cnn', 'dropout', 'gan', 'resnet']
        },
        'transformer_revolution': {
            'period': '2017-present',
            'key_figures': ['Vaswani', 'Devlin', 'Radford'],
            'core_concepts': ['attention', 'transformer', 'bert', 'gpt']
        },
        'large_language_models': {
            'period': '2020-present',
            'key_figures': ['OpenAI', 'Google', 'Anthropic'],
            'core_concepts': ['scaling', 'emergence', 'rlhf', 'multimodal']
        }
    },

    'learning_path': [
        'neural_networks_foundation',
        'deep_learning_breakthrough',
        'transformer_revolution',
        'large_language_models'
    ],

    'prerequisites': {
        'neural_networks_foundation': ['basic_mathematics', 'programming'],
        'deep_learning_breakthrough': ['neural_networks_foundation', 'linear_algebra', 'calculus'],
        'transformer_revolution': ['deep_learning_breakthrough', 'natural_language_processing'],
        'large_language_models': ['transformer_revolution', 'large_scale_training', 'computer_science']
    }
}

def get_development_timeline():
    """获取LLM发展时间线"""
    return LLM_CONFIG['development_stages']

def get_learning_path():
    """获取推荐学习路径"""
    return LLM_CONFIG['learning_path']

def get_prerequisites(stage: str):
    """获取指定阶段的前置知识"""
    return LLM_CONFIG['prerequisites'].get(stage, [])

def is_stage_available(stage: str) -> bool:
    """检查指定阶段的模块是否可用"""
    try:
        module = __import__(f'llm.evolution.{stage}')
        return module is not None
    except ImportError:
        return False

# 模块级别的便捷导入函数
def import_neural_foundation():
    """导入神经网络基础模块"""
    from .evolution.neural_networks_foundation import *
    return locals()

def import_deep_learning():
    """导入深度学习突破模块"""
    from .evolution.deep_learning_breakthrough import *
    return locals()

def import_transformer():
    """导入Transformer革命模块"""
    from .evolution.transformer_revolution import *
    return locals()

def import_large_language_models():
    """导入大语言模型时代模块"""
    from .evolution.large_language_models import *
    return locals()
