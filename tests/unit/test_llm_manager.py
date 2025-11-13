"""
LLM管理器单元测试

测试LLM管理器的模型管理、负载均衡、健康检查等功能。
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from src.llm.manager import (
    LLMManager, LoadBalancer, ModelInfo, ModelProvider,
    ModelStatus, LoadBalancingConfig
)
from src.core.types import LLMConfig, ModelCapabilities, LLMResponse
from src.core.exceptions import LLMError, ConfigurationError


class TestLoadBalancer:
    """负载均衡器测试"""

    @pytest.fixture
    def config(self):
        """负载均衡配置"""
        return LoadBalancingConfig(
            strategy="round_robin",
            enable_health_check=True,
            health_check_interval=60
        )

    @pytest.fixture
    def load_balancer(self, config):
        """负载均衡器实例"""
        return LoadBalancer(config)

    @pytest.fixture
    def sample_models(self):
        """示例模型列表"""
        return [
            ModelInfo(
                name="model1",
                provider=ModelProvider.OPENAI,
                status=ModelStatus.READY,
                capabilities=ModelCapabilities(
                    max_tokens=4096,
                    supports_vision=True,
                    supports_audio=True
                ),
                config=LLMConfig(model_name="gpt-4", provider="openai"),
                usage_count=10,
                average_response_time=1.5
            ),
            ModelInfo(
                name="model2",
                provider=ModelProvider.ANTHROPIC,
                status=ModelStatus.READY,
                capabilities=ModelCapabilities(
                    max_tokens=2048,
                    supports_vision=False,
                    supports_audio=True
                ),
                config=LLMConfig(model_name="claude-3", provider="anthropic"),
                usage_count=5,
                average_response_time=2.0
            ),
            ModelInfo(
                name="model3",
                provider=ModelProvider.HUGGINGFACE,
                status=ModelStatus.ERROR,  # 不可用
                capabilities=ModelCapabilities(
                    max_tokens=1024,
                    supports_vision=False,
                    supports_audio=False
                ),
                config=LLMConfig(model_name="llama-2", provider="huggingface")
            )
        ]

    @pytest.mark.asyncio
    async def test_round_robin_selection(self, load_balancer, sample_models):
        """测试轮询选择"""
        # 设置轮询策略
        load_balancer.config.strategy = "round_robin"

        # 多次选择，应该轮询
        selections = []
        for _ in range(6):  # 6次选择，应该循环2次
            model = await load_balancer.select_model(sample_models)
            if model:
                selections.append(model.name)

        # 过滤掉不可用模型
        expected_models = ["model1", "model2"]

        # 验证轮询
        for i in range(0, len(selections), 2):
            if i + 1 < len(selections):
                assert selections[i] != selections[i + 1]

        # 验证只有可用模型被选择
        for selection in selections:
            assert selection in expected_models

    @pytest.mark.asyncio
    async def test_least_loaded_selection(self, load_balancer, sample_models):
        """测试最少负载选择"""
        load_balancer.config.strategy = "least_loaded"

        model = await load_balancer.select_model(sample_models)

        # model2的使用次数最少（5次）
        assert model.name == "model2"

    @pytest.mark.asyncio
    async def test_fastest_selection(self, load_balancer, sample_models):
        """测试最快响应选择"""
        load_balancer.config.strategy = "fastest"

        model = await load_balancer.select_model(sample_models)

        # model1的平均响应时间最短（1.5s）
        assert model.name == "model1"

    @pytest.mark.asyncio
    async def test_random_selection(self, load_balancer, sample_models):
        """测试随机选择"""
        load_balancer.config.strategy = "random"

        selections = []
        for _ in range(10):
            model = await load_balancer.select_model(sample_models)
            if model:
                selections.append(model.name)

        # 随机选择应该包含可用模型
        available_models = {"model1", "model2"}
        selected_set = set(selections)
        assert selected_set.issubset(available_models)

    @pytest.mark.asyncio
    async def test_filter_by_requirements(self, load_balancer, sample_models):
        """测试根据需求过滤模型"""
        # 要求支持视觉功能
        requirements = {
            "features": ["vision"],
            "max_tokens": 2048
        }

        model = await load_balancer.select_model(sample_models, requirements)

        # 只有model1支持视觉功能
        assert model.name == "model1"

    @pytest.mark.asyncio
    async def test_filter_by_max_tokens(self, load_balancer, sample_models):
        """测试根据最大token数过滤模型"""
        requirements = {
            "max_tokens": 3000
        }

        model = await load_balancer.select_model(sample_models, requirements)

        # 只有model1的max_tokens >= 3000
        assert model.name == "model1"

    @pytest.mark.asyncio
    async def test_no_available_models(self, load_balancer):
        """测试没有可用模型"""
        all_error_models = [
            ModelInfo(
                name="error_model",
                provider=ModelProvider.OPENAI,
                status=ModelStatus.ERROR,
                capabilities=ModelCapabilities(),
                config=LLMConfig(model_name="error", provider="openai")
            )
        ]

        model = await load_balancer.select_model(all_error_models)
        assert model is None

    @pytest.mark.asyncio
    async def test_filter_by_modalities(self, load_balancer, sample_models):
        """测试根据模态过滤模型"""
        requirements = {
            "modalities": ["vision", "audio"]
        }

        model = await load_balancer.select_model(sample_models, requirements)

        # 只有model1同时支持视觉和音频
        assert model.name == "model1"


class TestLLMManager:
    """LLM管理器测试"""

    @pytest.fixture
    def config(self):
        """管理器配置"""
        return LoadBalancingConfig(
            strategy="least_loaded",
            enable_health_check=True,
            health_check_interval=1,  # 1秒用于测试
            max_concurrent_requests=5
        )

    @pytest.fixture
    def manager(self, config):
        """LLM管理器实例"""
        return LLMManager(config)

    @pytest.fixture
    def mock_llm_class(self):
        """模拟LLM类"""
        class MockLLM:
            def __init__(self, config):
                self.config = config
                self.initialized = False

            async def initialize(self):
                self.initialized = True

            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content=f"Mock response to: {prompt[:50]}...",
                    confidence=0.9,
                    tokens_used=100
                )

            async def health_check(self):
                return self.initialized

            async def get_capabilities(self):
                return ModelCapabilities(
                    max_tokens=2048,
                    supports_vision=False,
                    supports_audio=False
                )

            async def cleanup(self):
                self.initialized = False

        return MockLLM

    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        await manager.initialize()

        assert manager._running is True
        assert ModelProvider.OPENAI in manager._provider_registry
        assert ModelProvider.HUGGINGFACE in manager._provider_registry
        assert ModelProvider.MOCK in manager._provider_registry

    @pytest.mark.asyncio
    async def test_register_model(self, manager, mock_llm_class):
        """测试注册模型"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai", api_key="test")

            success = await manager.register_model("test_model", config)

            assert success is True
            assert "test_model" in manager._models

            model_info = manager._models["test_model"]
            assert model_info.name == "test_model"
            assert model_info.provider == ModelProvider.OPENAI
            assert model_info.status == ModelStatus.READY

    @pytest.mark.asyncio
    async def test_register_model_unsupported_provider(self, manager):
        """测试注册不支持的提供商"""
        config = LLMConfig(model_name="test", provider="unsupported")

        success = await manager.register_model("test_model", config)

        assert success is False
        assert "test_model" not in manager._models

    @pytest.mark.asyncio
    async def test_register_model_initialization_failure(self, manager, mock_llm_class):
        """测试模型初始化失败"""
        class FailingLLM(mock_llm_class):
            async def initialize(self):
                raise Exception("初始化失败")

        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: FailingLLM}):
            config = LLMConfig(model_name="failing", provider="openai", api_key="test")

            success = await manager.register_model("failing_model", config)

            assert success is True  # 注册成功，但初始化失败
            model_info = manager._models["failing_model"]
            assert model_info.status == ModelStatus.ERROR

    @pytest.mark.asyncio
    async def test_generate_with_specific_model(self, manager, mock_llm_class):
        """测试指定模型生成"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            # 注册模型
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 生成文本
            response = await manager.generate(
                "Hello, world!",
                model_name="gpt4"
            )

            assert isinstance(response, LLMResponse)
            assert "Hello, world!" in response.content
            assert response.confidence == 0.9

            # 验证统计更新
            model_info = manager._models["gpt4"]
            assert model_info.usage_count == 1

    @pytest.mark.asyncio
    async def test_generate_auto_select_model(self, manager, mock_llm_class):
        """测试自动选择模型生成"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            # 注册多个模型
            config1 = LLMConfig(model_name="gpt-3.5", provider="openai", api_key="test")
            config2 = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")

            await manager.register_model("gpt35", config1)
            await manager.register_model("gpt4", config2)

            # 生成文本（自动选择）
            response = await manager.generate("Test message")

            assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_generate_no_available_models(self, manager):
        """测试没有可用模型时生成"""
        with pytest.raises(LLMError, match="没有可用的模型"):
            await manager.generate("Test message")

    @pytest.mark.asyncio
    async def test_generate_unspecified_model(self, manager):
        """测试指定未注册模型"""
        with pytest.raises(LLMError, match="没有可用的模型"):
            await manager.generate("Test", model_name="nonexistent")

    @pytest.mark.asyncio
    async def test_generate_model_error_state(self, manager, mock_llm_class):
        """测试模型处于错误状态"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 手动设置为错误状态
            manager._models["gpt4"].status = ModelStatus.ERROR

            with pytest.raises(LLMError, match="没有可用的模型"):
                await manager.generate("Test", model_name="gpt4")

    @pytest.mark.asyncio
    async def test_stats_tracking(self, manager, mock_llm_class):
        """测试统计信息跟踪"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 生成多次
            await manager.generate("Test 1")
            await manager.generate("Test 2")
            await manager.generate("Test 3")

            stats = manager.get_stats()

            assert stats["total_requests"] == 3
            assert stats["successful_requests"] == 3
            assert stats["failed_requests"] == 0
            assert stats["registered_models"] == 1
            assert stats["ready_models"] == 1

    @pytest.mark.asyncio
    async def test_error_stats_tracking(self, manager, mock_llm_class):
        """测试错误统计跟踪"""
        class ErrorLLM(mock_llm_class):
            async def generate(self, prompt, **kwargs):
                raise Exception("生成错误")

        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: ErrorLLM}):
            config = LLMConfig(model_name="error", provider="openai", api_key="test")
            await manager.register_model("error_model", config)

            try:
                await manager.generate("Test")
            except LLMError:
                pass  # 预期错误

            stats = manager.get_stats()

            assert stats["total_requests"] == 1
            assert stats["successful_requests"] == 0
            assert stats["failed_requests"] == 1

    @pytest.mark.asyncio
    async def test_update_model_config(self, manager, mock_llm_class):
        """测试更新模型配置"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(
                model_name="gpt-4",
                provider="openai",
                api_key="test",
                temperature=0.5
            )
            await manager.register_model("gpt4", config)

            # 更新配置
            success = await manager.update_model_config(
                "gpt4",
                {"temperature": 0.8, "max_tokens": 3000}
            )

            assert success is True

            # 验证配置更新
            model_info = manager._models["gpt4"]
            assert model_info.config.temperature == 0.8
            assert model_info.config.max_tokens == 3000

    @pytest.mark.asyncio
    async def test_update_nonexistent_model_config(self, manager):
        """测试更新不存在的模型配置"""
        success = await manager.update_model_config(
            "nonexistent",
            {"temperature": 0.8}
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_list_models(self, manager, mock_llm_class):
        """测试列出模型"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            # 注册不同状态的模型
            configs = [
                LLMConfig(model_name="ready", provider="openai", api_key="test"),
                LLMConfig(model_name="error", provider="openai", api_key="test")
            ]

            await manager.register_model("ready_model", configs[0])
            await manager.register_model("error_model", configs[1])

            # 手动设置一个为错误状态
            manager._models["error_model"].status = ModelStatus.ERROR

            # 列出所有模型
            all_models = await manager.list_models()
            assert len(all_models) == 2

            # 只列出就绪模型
            ready_models = await manager.list_models(ModelStatus.READY)
            assert len(ready_models) == 1
            assert ready_models[0]["name"] == "ready_model"

    @pytest.mark.asyncio
    async def test_get_model_info(self, manager, mock_llm_class):
        """测试获取模型信息"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            info = await manager.get_model_info("gpt4")

            assert info is not None
            assert info["name"] == "gpt4"
            assert info["provider"] == "openai"
            assert info["status"] == "ready"
            assert "capabilities" in info
            assert "usage_count" in info
            assert "average_response_time" in info

    @pytest.mark.asyncio
    async def test_get_nonexistent_model_info(self, manager):
        """测试获取不存在的模型信息"""
        info = await manager.get_model_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_unregister_model(self, manager, mock_llm_class):
        """测试注销模型"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 注销模型
            success = await manager.unregister_model("gpt4")

            assert success is True
            assert "gpt4" not in manager._models

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_model(self, manager):
        """测试注销不存在的模型"""
        success = await manager.unregister_model("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_health_check(self, manager, mock_llm_class):
        """测试健康检查"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            health = await manager.health_check()

            assert "healthy" in health
            assert "models" in health
            assert "stats" in health
            assert health["healthy"] is True
            assert "gpt4" in health["models"]
            assert health["models"]["gpt4"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_with_failing_model(self, manager, mock_llm_class):
        """测试包含失败模型的健康检查"""
        class FailingLLM(mock_llm_class):
            async def health_check(self):
                return False

        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: FailingLLM}):
            config = LLMConfig(model_name="failing", provider="openai", api_key="test")
            await manager.register_model("failing_model", config)

            health = await manager.health_check()

            assert health["healthy"] is True  # 管理器本身健康
            assert health["models"]["failing_model"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_cleanup(self, manager, mock_llm_class):
        """测试清理资源"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 启动健康检查
            await manager._start_health_check()
            assert manager._health_check_task is not None

            # 清理
            await manager.cleanup()

            assert manager._running is False
            assert manager._health_check_task is None
            assert len(manager._models) == 0

    @pytest.mark.asyncio
    async def test_concurrent_generation(self, manager, mock_llm_class):
        """测试并发生成"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 并发生成
            tasks = [
                manager.generate(f"Message {i}")
                for i in range(5)
            ]

            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            for response in responses:
                assert isinstance(response, LLMResponse)

            # 验证统计
            stats = manager.get_stats()
            assert stats["total_requests"] == 5
            assert stats["successful_requests"] == 5

    @pytest.mark.asyncio
    async def test_model_stats_update(self, manager, mock_llm_class):
        """测试模型统计更新"""
        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: mock_llm_class}):
            config = LLMConfig(model_name="gpt-4", provider="openai", api_key="test")
            await manager.register_model("gpt4", config)

            # 多次生成，观察平均响应时间变化
            await manager.generate("Test 1")
            await manager.generate("Test 2")

            model_info = manager._models["gpt4"]
            assert model_info.usage_count == 2
            assert model_info.average_response_time > 0

    @pytest.mark.asyncio
    async def test_error_rate_based_status_update(self, manager, mock_llm_class):
        """测试基于错误率的状态更新"""
        class ErrorLLM(mock_llm_class):
            def __init__(self, config):
                super().__init__(config)
                self.call_count = 0

            async def generate(self, prompt, **kwargs):
                self.call_count += 1
                if self.call_count <= 3:  # 前3次成功，后7次失败
                    return await super().generate(prompt, **kwargs)
                else:
                    raise Exception("模拟错误")

        with patch.dict(manager._provider_registry, {ModelProvider.OPENAI: ErrorLLM}):
            config = LLMConfig(model_name="error_model", provider="openai", api_key="test")
            await manager.register_model("error_model", config)

            # 执行多次请求（包含失败）
            for i in range(10):
                try:
                    await manager.generate(f"Test {i}")
                except LLMError:
                    pass  # 忽略生成错误

            model_info = manager._models["error_model"]
            # 错误率应该很高（7/10 = 70% > 50%）
            assert model_info.status == ModelStatus.ERROR

    def test_load_balancing_config_defaults(self):
        """测试负载均衡配置默认值"""
        config = LoadBalancingConfig()

        assert config.strategy == "round_robin"
        assert config.enable_health_check is True
        assert config.health_check_interval == 60
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30

    def test_model_provider_enum(self):
        """测试模型提供商枚举"""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.HUGGINGFACE.value == "huggingface"
        assert ModelProvider.LOCAL.value == "local"
        assert ModelProvider.AZURE.value == "azure"
        assert ModelProvider.GOOGLE.value == "google"
        assert ModelProvider.MOCK.value == "mock"

    def test_model_status_enum(self):
        """测试模型状态枚举"""
        assert ModelStatus.UNINITIALIZED.value == "uninitialized"
        assert ModelStatus.INITIALIZING.value == "initializing"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.BUSY.value == "busy"
        assert ModelStatus.ERROR.value == "error"
        assert ModelStatus.UNAVAILABLE.value == "unavailable"

    def test_model_info_creation(self):
        """测试模型信息创建"""
        capabilities = ModelCapabilities(max_tokens=4096)
        config = LLMConfig(model_name="test", provider="openai")

        model_info = ModelInfo(
            name="test_model",
            provider=ModelProvider.OPENAI,
            status=ModelStatus.READY,
            capabilities=capabilities,
            config=config
        )

        assert model_info.name == "test_model"
        assert model_info.provider == ModelProvider.OPENAI
        assert model_info.status == ModelStatus.READY
        assert model_info.capabilities == capabilities
        assert model_info.config == config
        assert model_info.usage_count == 0
        assert model_info.error_count == 0
        assert model_info.average_response_time == 0.0
        assert model_info.instance is None
        assert model_info.last_used > 0
