"""
LLM管理器

统一管理和调度各种大语言模型。
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..core.interfaces import BaseLLM
from ..core.types import LLMConfig, LLMResponse, ModelCapabilities
from ..core.exceptions import LLMError, ConfigurationError
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector


class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AZURE = "azure"
    GOOGLE = "google"
    MOCK = "mock"


class ModelStatus(Enum):
    """模型状态"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    provider: ModelProvider
    status: ModelStatus
    capabilities: ModelCapabilities
    config: LLMConfig
    instance: Optional[BaseLLM] = None
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0


@dataclass
class LoadBalancingConfig:
    """负载均衡配置"""
    strategy: str = "round_robin"  # round_robin, least_loaded, fastest, random
    enable_health_check: bool = True
    health_check_interval: int = 60  # 秒
    max_concurrent_requests: int = 10
    request_timeout: int = 30


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, config: LoadBalancingConfig):
        self.config = config
        self.logger = get_logger("llm.load_balancer")
        self._current_index = 0

    async def select_model(
        self,
        models: List[ModelInfo],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelInfo]:
        """选择最优模型"""
        # 过滤可用模型
        available_models = [
            model for model in models
            if model.status == ModelStatus.READY
        ]

        if not available_models:
            return None

        if requirements:
            # 根据需求过滤
            available_models = self._filter_by_requirements(
                available_models, requirements
            )

        if not available_models:
            return None

        # 根据策略选择
        if self.config.strategy == "round_robin":
            return self._round_robin_select(available_models)
        elif self.config.strategy == "least_loaded":
            return self._least_loaded_select(available_models)
        elif self.config.strategy == "fastest":
            return self._fastest_select(available_models)
        elif self.config.strategy == "random":
            return self._random_select(available_models)
        else:
            return available_models[0]

    def _filter_by_requirements(
        self,
        models: List[ModelInfo],
        requirements: Dict[str, Any]
    ) -> List[ModelInfo]:
        """根据需求过滤模型"""
        filtered = []

        for model in models:
            capabilities = model.capabilities

            # 检查最大token长度
            if "max_tokens" in requirements:
                if capabilities.max_tokens < requirements["max_tokens"]:
                    continue

            # 检查支持的功能
            if "features" in requirements:
                required_features = requirements["features"]
                if not all(capabilities.supports_feature(f) for f in required_features):
                    continue

            # 检查模态支持
            if "modalities" in requirements:
                required_modalities = requirements["modalities"]
                if not all(capabilities.supports_modality(m) for m in required_modalities):
                    continue

            filtered.append(model)

        return filtered

    def _round_robin_select(self, models: List[ModelInfo]) -> ModelInfo:
        """轮询选择"""
        model = models[self._current_index % len(models)]
        self._current_index += 1
        return model

    def _least_loaded_select(self, models: List[ModelInfo]) -> ModelInfo:
        """最少负载选择"""
        return min(models, key=lambda m: m.usage_count)

    def _fastest_select(self, models: List[ModelInfo]) -> ModelInfo:
        """最快响应选择"""
        return min(models, key=lambda m: m.average_response_time)

    def _random_select(self, models: List[ModelInfo]) -> ModelInfo:
        """随机选择"""
        import random
        return random.choice(models)


class LLMManager:
    """
    LLM管理器

    负责：
    - 模型注册和管理
    - 负载均衡和路由
    - 健康检查
    - 性能监控
    - 故障转移
    """

    def __init__(self, config: Optional[LoadBalancingConfig] = None):
        self.config = config or LoadBalancingConfig()
        self.logger = get_logger("llm.manager")
        self.metrics = MetricsCollector("llm.manager")

        # 模型注册表
        self._models: Dict[str, ModelInfo] = {}
        self._provider_registry: Dict[ModelProvider, Type[BaseLLM]] = {}

        # 负载均衡器
        self.load_balancer = LoadBalancer(self.config)

        # 健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # 统计信息
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "model_switches": 0,
            "health_checks": 0
        }

    async def initialize(self) -> None:
        """初始化管理器"""
        try:
            # 注册默认提供商
            await self._register_default_providers()

            # 启动健康检查
            if self.config.enable_health_check:
                await self._start_health_check()

            self._running = True
            self.logger.info("LLM管理器初始化完成")

        except Exception as e:
            self.logger.error(f"LLM管理器初始化失败: {e}")
            raise

    async def _register_default_providers(self) -> None:
        """注册默认提供商"""
        from .openai_llm import OpenAILLM
        from .huggingface_llm import HuggingFaceLLM
        from .mock_llm import MockLLM

        self._provider_registry[ModelProvider.OPENAI] = OpenAILLM
        self._provider_registry[ModelProvider.HUGGINGFACE] = HuggingFaceLLM
        self._provider_registry[ModelProvider.MOCK] = MockLLM

    async def register_model(self, name: str, config: LLMConfig) -> bool:
        """
        注册模型

        Args:
            name: 模型名称
            config: 模型配置

        Returns:
            bool: 注册是否成功
        """
        try:
            if name in self._models:
                self.logger.warning(f"模型 {name} 已存在，将覆盖")

            provider = ModelProvider(config.provider)

            # 获取提供商类
            if provider not in self._provider_registry:
                self.logger.error(f"不支持的提供商: {provider}")
                return False

            model_class = self._provider_registry[provider]

            # 创建模型实例
            model_instance = model_class(config)

            # 初始化模型
            model_info = ModelInfo(
                name=name,
                provider=provider,
                status=ModelStatus.INITIALIZING,
                capabilities=await model_instance.get_capabilities(),
                config=config
            )

            # 异步初始化模型
            await self._initialize_model(model_info, model_instance)

            self._models[name] = model_info
            self.logger.info(f"模型 {name} 注册成功")
            return True

        except Exception as e:
            self.logger.error(f"模型 {name} 注册失败: {e}")
            return False

    async def _initialize_model(
        self,
        model_info: ModelInfo,
        model_instance: BaseLLM
    ) -> None:
        """异步初始化模型"""
        try:
            await model_instance.initialize()
            model_info.instance = model_instance
            model_info.status = ModelStatus.READY
            self.logger.info(f"模型 {model_info.name} 初始化完成")

        except Exception as e:
            model_info.status = ModelStatus.ERROR
            model_info.error_count += 1
            self.logger.error(f"模型 {model_info.name} 初始化失败: {e}")

    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        requirements: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本

        Args:
            prompt: 输入提示
            model_name: 指定模型名称（可选）
            requirements: 模型需求（可选）
            **kwargs: 其他参数

        Returns:
            LLMResponse: 生成结果
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        try:
            # 选择模型
            model_info = await self._select_model(model_name, requirements)
            if not model_info:
                raise LLMError("没有可用的模型")

            # 更新统计
            model_info.usage_count += 1
            model_info.last_used = time.time()

            # 生成文本
            response = await model_info.instance.generate(prompt, **kwargs)

            # 更新性能统计
            response_time = time.time() - start_time
            self._update_model_stats(model_info, response_time, True)

            # 记录指标
            self.metrics.record("generation_time", response_time)
            self.metrics.record("tokens_generated", getattr(response, 'tokens_used', 0))
            self._stats["successful_requests"] += 1

            self.logger.info(
                f"文本生成完成，模型: {model_info.name}, "
                f"耗时: {response_time:.2f}s"
            )

            return response

        except Exception as e:
            self._stats["failed_requests"] += 1
            self.logger.error(f"文本生成失败: {e}")
            raise LLMError(f"文本生成失败: {e}")

    async def _select_model(
        self,
        model_name: Optional[str],
        requirements: Optional[Dict[str, Any]]
    ) -> Optional[ModelInfo]:
        """选择模型"""
        if model_name:
            # 指定模型
            if model_name in self._models:
                model_info = self._models[model_name]
                if model_info.status == ModelStatus.READY:
                    return model_info
                else:
                    self.logger.warning(f"模型 {model_name} 不可用，状态: {model_info.status}")
                    return None
            else:
                self.logger.error(f"模型 {model_name} 未注册")
                return None
        else:
            # 自动选择
            models = list(self._models.values())
            return await self.load_balancer.select_model(models, requirements)

    def _update_model_stats(
        self,
        model_info: ModelInfo,
        response_time: float,
        success: bool
    ) -> None:
        """更新模型统计"""
        # 更新平均响应时间
        total_requests = model_info.usage_count
        current_avg = model_info.average_response_time
        new_avg = (current_avg * (total_requests - 1) + response_time) / total_requests
        model_info.average_response_time = new_avg

        # 更新错误计数
        if not success:
            model_info.error_count += 1
            # 如果错误过多，标记为错误状态
            error_rate = model_info.error_count / total_requests
            if error_rate > 0.5:  # 错误率超过50%
                model_info.status = ModelStatus.ERROR
                self.logger.warning(f"模型 {model_info.name} 错误率过高，标记为错误状态")

    async def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if name not in self._models:
            return None

        model_info = self._models[name]
        return {
            "name": model_info.name,
            "provider": model_info.provider.value,
            "status": model_info.status.value,
            "capabilities": model_info.capabilities.to_dict(),
            "usage_count": model_info.usage_count,
            "error_count": model_info.error_count,
            "average_response_time": model_info.average_response_time,
            "last_used": model_info.last_used
        }

    async def list_models(self, status_filter: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """列出所有模型"""
        models = []
        for model_info in self._models.values():
            if status_filter and model_info.status != status_filter:
                continue

            models.append({
                "name": model_info.name,
                "provider": model_info.provider.value,
                "status": model_info.status.value,
                "capabilities": model_info.capabilities.to_dict()
            })

        return models

    async def update_model_config(self, name: str, config_updates: Dict[str, Any]) -> bool:
        """更新模型配置"""
        if name not in self._models:
            return False

        model_info = self._models[name]

        try:
            # 更新配置
            for key, value in config_updates.items():
                if hasattr(model_info.config, key):
                    setattr(model_info.config, key, value)

            # 重新初始化模型
            if model_info.instance:
                await model_info.instance.cleanup()

            provider_class = self._provider_registry[model_info.provider]
            new_instance = provider_class(model_info.config)

            await self._initialize_model(model_info, new_instance)

            self.logger.info(f"模型 {name} 配置更新成功")
            return True

        except Exception as e:
            self.logger.error(f"模型 {name} 配置更新失败: {e}")
            return False

    async def _start_health_check(self) -> None:
        """启动健康检查"""
        if self._health_check_task:
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("健康检查任务已启动")

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康检查出错: {e}")
                await asyncio.sleep(5)  # 出错时短暂等待

    async def _perform_health_checks(self) -> None:
        """执行健康检查"""
        self._stats["health_checks"] += 1

        for model_info in self._models.values():
            try:
                if not model_info.instance:
                    continue

                # 执行健康检查
                is_healthy = await model_info.instance.health_check()

                if is_healthy:
                    if model_info.status == ModelStatus.ERROR:
                        model_info.status = ModelStatus.READY
                        self.logger.info(f"模型 {model_info.name} 恢复正常")
                else:
                    if model_info.status == ModelStatus.READY:
                        model_info.status = ModelStatus.UNAVAILABLE
                        self.logger.warning(f"模型 {model_info.name} 健康检查失败")

            except Exception as e:
                self.logger.error(f"模型 {model_info.name} 健康检查异常: {e}")
                model_info.status = ModelStatus.ERROR

    async def unregister_model(self, name: str) -> bool:
        """注销模型"""
        if name not in self._models:
            return False

        model_info = self._models.pop(name)

        try:
            if model_info.instance:
                await model_info.instance.cleanup()

            self.logger.info(f"模型 {name} 注销成功")
            return True

        except Exception as e:
            self.logger.error(f"模型 {name} 注销失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        model_stats = {}
        for name, model_info in self._models.items():
            model_stats[name] = {
                "status": model_info.status.value,
                "usage_count": model_info.usage_count,
                "error_count": model_info.error_count,
                "average_response_time": model_info.average_response_time
            }

        return {
            **self._stats,
            "registered_models": len(self._models),
            "ready_models": len([
                m for m in self._models.values()
                if m.status == ModelStatus.READY
            ]),
            "model_details": model_stats
        }

    async def health_check(self) -> Dict[str, Any]:
        """管理器健康检查"""
        model_health = {}
        for name, model_info in self._models.items():
            try:
                if model_info.instance:
                    is_healthy = await model_info.instance.health_check()
                    model_health[name] = {
                        "healthy": is_healthy,
                        "status": model_info.status.value,
                        "last_check": time.time()
                    }
                else:
                    model_health[name] = {
                        "healthy": False,
                        "status": model_info.status.value,
                        "error": "模型实例未初始化"
                    }
            except Exception as e:
                model_health[name] = {
                    "healthy": False,
                    "status": ModelStatus.ERROR.value,
                    "error": str(e)
                }

        return {
            "healthy": self._running,
            "models": model_health,
            "stats": self._stats
        }

    async def cleanup(self) -> None:
        """清理资源"""
        self._running = False

        # 停止健康检查
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # 清理所有模型
        for model_info in self._models.values():
            if model_info.instance:
                try:
                    await model_info.instance.cleanup()
                except Exception as e:
                    self.logger.error(f"模型 {model_info.name} 清理失败: {e}")

        self._models.clear()
        self.logger.info("LLM管理器资源清理完成")

