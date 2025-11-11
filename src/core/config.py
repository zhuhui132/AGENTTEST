"""
配置管理模块
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from .types import AgentConfig, LLMConfig, MemoryConfig, ToolConfig, RAGConfig
from .exceptions import ConfigError


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache = {}

    def load_config(
        self,
        config_name: str,
        config_type: str = "agent"
    ) -> Dict[str, Any]:
        """加载配置"""
        cache_key = f"{config_type}_{config_name}"

        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config_file = self.config_dir / f"{config_name}.{config_type}.yaml"

        if not config_file.exists():
            # 尝试JSON格式
            config_file = self.config_dir / f"{config_name}.{config_type}.json"

        if not config_file.exists():
            raise ConfigError(f"配置文件不存在: {config_name}")

        try:
            if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            self._config_cache[cache_key] = config
            return config

        except Exception as e:
            raise ConfigError(f"加载配置失败: {e}")

    def save_config(
        self,
        config_name: str,
        config: Dict[str, Any],
        config_type: str = "agent"
    ) -> bool:
        """保存配置"""
        try:
            config_file = self.config_dir / f"{config_name}.{config_type}.yaml"

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False,
                         allow_unicode=True, indent=2)

            # 更新缓存
            cache_key = f"{config_type}_{config_name}"
            self._config_cache[cache_key] = config

            return True

        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}")

    def get_agent_config(self, config_name: str = "default") -> AgentConfig:
        """获取Agent配置"""
        config_dict = self.load_config(config_name, "agent")
        return self._dict_to_agent_config(config_dict)

    def get_llm_config(self, config_name: str = "default") -> LLMConfig:
        """获取LLM配置"""
        config_dict = self.load_config(config_name, "llm")
        return self._dict_to_llm_config(config_dict)

    def get_memory_config(self, config_name: str = "default") -> MemoryConfig:
        """获取记忆配置"""
        config_dict = self.load_config(config_name, "memory")
        return self._dict_to_memory_config(config_dict)

    def get_tool_config(self, config_name: str = "default") -> ToolConfig:
        """获取工具配置"""
        config_dict = self.load_config(config_name, "tool")
        return self._dict_to_tool_config(config_dict)

    def get_rag_config(self, config_name: str = "default") -> RAGConfig:
        """获取RAG配置"""
        config_dict = self.load_config(config_name, "rag")
        return self._dict_to_rag_config(config_dict)

    def _dict_to_agent_config(self, config_dict: Dict[str, Any]) -> AgentConfig:
        """字典转换为Agent配置"""
        # 解析LLM配置
        llm_config_dict = config_dict.get('llm', {})
        llm_config = self._dict_to_llm_config(llm_config_dict)

        return AgentConfig(
            name=config_dict.get('name', 'default_agent'),
            version=config_dict.get('version', '1.0.0'),
            llm_config=llm_config,
            max_context_length=config_dict.get('max_context_length', 4000),
            memory_enabled=config_dict.get('memory_enabled', True),
            rag_enabled=config_dict.get('rag_enabled', True),
            tools_enabled=config_dict.get('tools_enabled', True),
            debug_mode=config_dict.get('debug_mode', False),
            log_level=config_dict.get('log_level', 'INFO'),
            max_concurrent_requests=config_dict.get('max_concurrent_requests', 10),
            response_timeout=config_dict.get('response_timeout', 30.0),
            cache_enabled=config_dict.get('cache_enabled', True),
            cache_ttl=config_dict.get('cache_ttl', 3600)
        )

    def _dict_to_llm_config(self, config_dict: Dict[str, Any]) -> LLMConfig:
        """字典转换为LLM配置"""
        return LLMConfig(
            model_name=config_dict.get('model_name', 'default'),
            max_tokens=config_dict.get('max_tokens', 2048),
            temperature=config_dict.get('temperature', 0.7),
            top_p=config_dict.get('top_p', 0.9),
            top_k=config_dict.get('top_k', 40),
            frequency_penalty=config_dict.get('frequency_penalty', 0.0),
            presence_penalty=config_dict.get('presence_penalty', 0.0),
            stop_sequences=config_dict.get('stop_sequences', []),
            timeout=config_dict.get('timeout', 30.0)
        )

    def _dict_to_memory_config(self, config_dict: Dict[str, Any]) -> MemoryConfig:
        """字典转换为记忆配置"""
        return MemoryConfig(
            max_memories=config_dict.get('max_memories', 10000),
            default_importance=config_dict.get('default_importance', 1.0),
            importance_decay_rate=config_dict.get('importance_decay_rate', 0.99),
            retrieval_limit=config_dict.get('retrieval_limit', 5),
            similarity_threshold=config_dict.get('similarity_threshold', 0.7),
            cleanup_interval=config_dict.get('cleanup_interval', 3600)
        )

    def _dict_to_tool_config(self, config_dict: Dict[str, Any]) -> ToolConfig:
        """字典转换为工具配置"""
        return ToolConfig(
            max_tools=config_dict.get('max_tools', 100),
            default_timeout=config_dict.get('default_timeout', 30.0),
            retry_attempts=config_dict.get('retry_attempts', 3),
            parallel_execution=config_dict.get('parallel_execution', True),
            max_parallel_tools=config_dict.get('max_parallel_tools', 5)
        )

    def _dict_to_rag_config(self, config_dict: Dict[str, Any]) -> RAGConfig:
        """字典转换为RAG配置"""
        return RAGConfig(
            max_documents=config_dict.get('max_documents', 10000),
            retrieval_limit=config_dict.get('retrieval_limit', 5),
            similarity_threshold=config_dict.get('similarity_threshold', 0.7),
            embedding_model=config_dict.get('embedding_model', 'default'),
            reranking_enabled=config_dict.get('reranking_enabled', True),
            chunk_size=config_dict.get('chunk_size', 500),
            chunk_overlap=config_dict.get('chunk_overlap', 50)
        )


# 全局配置管理器实例
config_manager = ConfigManager()


# 环境变量配置
def get_env_config() -> Dict[str, Any]:
    """从环境变量获取配置"""
    return {
        'debug': os.getenv('AGENT_DEBUG', 'false').lower() == 'true',
        'log_level': os.getenv('AGENT_LOG_LEVEL', 'INFO'),
        'max_concurrent_requests': int(os.getenv('AGENT_MAX_CONCURRENT', '10')),
        'response_timeout': float(os.getenv('AGENT_RESPONSE_TIMEOUT', '30.0')),
        'cache_enabled': os.getenv('AGENT_CACHE_ENABLED', 'true').lower() == 'true',
        'llm_model': os.getenv('AGENT_LLM_MODEL', 'default'),
        'llm_temperature': float(os.getenv('AGENT_LLM_TEMPERATURE', '0.7')),
        'memory_enabled': os.getenv('AGENT_MEMORY_ENABLED', 'true').lower() == 'true',
        'rag_enabled': os.getenv('AGENT_RAG_ENABLED', 'true').lower() == 'true',
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置"""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def create_default_configs():
    """创建默认配置文件"""
    config_manager = ConfigManager()

    # 默认Agent配置
    agent_config = {
        'name': 'default_agent',
        'version': '1.0.0',
        'debug_mode': False,
        'log_level': 'INFO',
        'max_context_length': 4000,
        'memory_enabled': True,
        'rag_enabled': True,
        'tools_enabled': True,
        'max_concurrent_requests': 10,
        'response_timeout': 30.0,
        'cache_enabled': True,
        'cache_ttl': 3600,
        'llm': {
            'model_name': 'gpt-3.5-turbo',
            'max_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40
        }
    }

    config_manager.save_config('default', agent_config, 'agent')

    # 默认记忆配置
    memory_config = {
        'max_memories': 10000,
        'default_importance': 1.0,
        'importance_decay_rate': 0.99,
        'retrieval_limit': 5,
        'similarity_threshold': 0.7,
        'cleanup_interval': 3600
    }

    config_manager.save_config('default', memory_config, 'memory')

    # 默认RAG配置
    rag_config = {
        'max_documents': 10000,
        'retrieval_limit': 5,
        'similarity_threshold': 0.7,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'reranking_enabled': True,
        'chunk_size': 500,
        'chunk_overlap': 50
    }

    config_manager.save_config('default', rag_config, 'rag')

    # 默认工具配置
    tool_config = {
        'max_tools': 100,
        'default_timeout': 30.0,
        'retry_attempts': 3,
        'parallel_execution': True,
        'max_parallel_tools': 5
    }

    config_manager.save_config('default', tool_config, 'tool')
