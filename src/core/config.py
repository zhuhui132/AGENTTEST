"""
配置管理模块

提供完整的配置管理功能，包括：
- 多格式配置文件支持（YAML、JSON、TOML）
- 环境变量覆盖
- 配置验证和类型转换
- 配置热重载
- 配置模板和继承
- 敏感信息加密
"""

import os
import json
import yaml
import toml
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .types import AgentConfig, LLMConfig, MemoryConfig, ToolConfig, RAGConfig
from .exceptions import ConfigError, InvalidConfigError

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变化监听器"""

    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json', '.toml')):
            logger.info(f"配置文件发生变化: {event.src_path}")
            # 重新加载相关配置
            self.config_manager._invalidate_cache()


class ConfigManager:
    """高级配置管理器

    支持多格式配置文件、环境变量覆盖、配置验证、热重载等功能
    """

    def __init__(
        self,
        config_dir: str = "config",
        enable_hot_reload: bool = False,
        encryption_key: Optional[str] = None
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache = {}
        self._watchers = {}
        self._observers = []
        self._lock = threading.RLock()
        self._enable_hot_reload = enable_hot_reload
        self._encryption_key = encryption_key
        self._validators = {}
        self._transforms = {}

        # 注册默认验证器和转换器
        self._register_default_validators()
        self._register_default_transforms()

        if enable_hot_reload:
            self._setup_file_watcher()

    def load_config(
        self,
        config_name: str,
        config_type: str = "agent",
        env_override: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """加载配置

        Args:
            config_name: 配置名称
            config_type: 配置类型
            env_override: 是否允许环境变量覆盖
            validate: 是否验证配置

        Returns:
            配置字典
        """
        with self._lock:
            cache_key = f"{config_type}_{config_name}"

            if cache_key in self._config_cache:
                config = self._config_cache[cache_key].copy()
            else:
                config = self._load_config_from_file(config_name, config_type)
                self._config_cache[cache_key] = config.copy()

            # 应用环境变量覆盖
            if env_override:
                config = self._apply_env_overrides(config, config_type)

            # 应用配置转换
            config = self._apply_transforms(config, config_type)

            # 验证配置
            if validate:
                self._validate_config(config, config_type)

            return config

    def _load_config_from_file(self, config_name: str, config_type: str) -> Dict[str, Any]:
        """从文件加载原始配置"""
        # 支持多种格式
        extensions = ['.yaml', '.yml', '.json', '.toml']
        config_file = None

        for ext in extensions:
            potential_file = self.config_dir / f"{config_name}.{config_type}{ext}"
            if potential_file.exists():
                config_file = potential_file
                break

        if not config_file:
            raise ConfigError(f"配置文件不存在: {config_name}.{config_type} (尝试了 {extensions})")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    config = json.load(f)
                elif config_file.suffix == '.toml'::
                    config = toml.load(f)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {config_file.suffix}")

                # 处理配置继承
                if 'extends' in config:
                    config = self._resolve_inheritance(config, config_name, config_type)

                return config

        except Exception as e:
            raise ConfigError(f"加载配置文件失败 {config_file}: {e}") from e

    def _resolve_inheritance(self, config: Dict[str, Any], config_name: str, config_type: str) -> Dict[str, Any]:
        """解析配置继承"""
        base_config_name = config.pop('extends')
        base_config = self.load_config(base_config_name, config_type, env_override=False, validate=False)

        # 深度合并配置
        return self._deep_merge(base_config, config)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(
        self,
        config_name: str,
        config: Dict[str, Any],
        config_type: str = "agent",
        format: str = "yaml",
        encrypt_sensitive: bool = False
    ) -> bool:
        """保存配置

        Args:
            config_name: 配置名称
            config: 配置字典
            config_type: 配置类型
            format: 保存格式 (yaml, json, toml)
            encrypt_sensitive: 是否加密敏感信息
        """
        try:
            # 验证配置
            self._validate_config(config, config_type)

            # 处理敏感信息
            if encrypt_sensitive:
                config = self._encrypt_sensitive_fields(config)

            config_file = self.config_dir / f"{config_name}.{config_type}.{format.lstrip('.')}"

            with open(config_file, 'w', encoding='utf-8') as f:
                if format in ['yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
                elif format in ['json', '.json']:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif format in ['toml', '.toml']:
                    toml.dump(config, f)
                else:
                    raise ConfigError(f"不支持的格式: {format}")

            # 更新缓存
            with self._lock:
                cache_key = f"{config_type}_{config_name}"
                self._config_cache[cache_key] = config.copy()

            logger.info(f"配置已保存: {config_file}")
            return True

        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}") from e

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
