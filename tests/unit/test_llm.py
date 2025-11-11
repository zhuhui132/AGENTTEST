"""
LLM模块测试
测试大语言模型的基础功能
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.llm.base import BaseLLM
from src.llm.openai import OpenAILLM
from src.llm.claude import ClaudeLLM
from src.llm.local import LocalLLM
from src.core.types import LLMConfig, Message


class TestBaseLLM:
    """基础LLM抽象类测试"""

    def test_abstract_class(self):
        """测试抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            BaseLLM()

    def test_interface_methods(self):
        """测试抽象方法定义"""
        assert hasattr(BaseLLM, 'generate')
        assert hasattr(BaseLLM, 'generate_stream')
        assert hasattr(BaseLLM, 'validate_config')

    def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            role="user",
            content="测试消息",
            timestamp="2023-01-01T00:00:00Z"
        )

        assert message.role == "user"
        assert message.content == "测试消息"
        assert message.timestamp == "2023-01-01T00:00:00Z"

    def test_message_validation(self):
        """测试消息验证"""
        # 有效消息
        valid_message = Message(role="user", content="测试")
        assert valid_message.role == "user"

        # 无效消息
        with pytest.raises(ValueError):
            Message(role="", content="测试")  # 空角色

        with pytest.raises(ValueError):
            Message(role="user", content="")  # 空内容


class TestOpenAILLM:
    """OpenAI LLM测试类"""

    @pytest.fixture
    def config(self):
        """配置fixture"""
        return LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            max_tokens=1000,
            temperature=0.7
        )

    @pytest.fixture
    def mock_client(self):
        """模拟OpenAI客户端"""
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[{
                "message": {"content": "测试响应"},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        ))
        return client

    @pytest.fixture
    def llm(self, config, mock_client):
        """LLM实例fixture"""
        with pytest.MonkeyPatch('openai.OpenAI', return_value=mock_client):
            return OpenAILLM(config)

    def test_initialization(self, llm):
        """测试LLM初始化"""
        assert llm.config.provider == "openai"
        assert llm.config.model == "gpt-3.5-turbo"
        assert llm.config.api_key == "test-key"

    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )

        OpenAILLM(valid_config)  # 不应该抛出异常

        # 无效配置
        invalid_configs = [
            LLMConfig(provider="", model="gpt-3.5-turbo", api_key="test-key"),  # 空provider
            LLMConfig(provider="openai", model="", api_key="test-key"),      # 空model
            LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key=""),  # 空api_key
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError):
                OpenAILLM(config)

    @pytest.mark.asyncio
    async def test_generate_response(self, llm):
        """测试生成响应"""
        prompt = "请介绍你自己"

        response = await llm.generate(prompt)

        assert response.content == "测试响应"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_with_history(self, llm):
        """测试带历史记录的生成"""
        messages = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！我是AI助手"),
            Message(role="user", content="你能做什么？")
        ]

        response = await llm.generate(messages)

        assert response.content is not None
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm):
        """测试带系统提示的生成"""
        messages = [
            Message(role="system", content="你是一个专业的AI助手"),
            Message(role="user", content="请回答问题")
        ]

        response = await llm.generate(messages)

        assert response.content is not None

    @pytest.mark.asyncio
    async def test_generate_streaming(self, llm):
        """测试流式生成"""
        # 设置模拟流式响应
        chunks = [
            Mock(choices=[{"delta": {"content": "你"}}]),
            Mock(choices=[{"delta": {"content": "好"}}]),
            Mock(choices=[{"delta": {"content": "！"}}])
        ]

        llm.client.chat.completions.create = AsyncMock(
            side_effect=chunks
        )

        collected_content = ""
        async for chunk in llm.generate_stream("你好"):
            collected_content += chunk.choices[0].delta.content

        assert collected_content == "你好！"

    @pytest.mark.asyncio
    async def test_error_handling(self, llm):
        """测试错误处理"""
        # 模拟API错误
        llm.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API错误")
        )

        with pytest.raises(Exception):
            await llm.generate("测试")

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, llm):
        """测试速率限制处理"""
        # 模拟速率限制错误
        rate_limit_error = Exception("Rate limit exceeded")
        rate_limit_error.status_code = 429

        llm.client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)

        with pytest.raises(Exception):
            await llm.generate("测试")

    @pytest.mark.asyncio
    async def test_token_counting(self, llm):
        """测试令牌计数"""
        long_prompt = "x" * 1000  # 很长的提示

        response = await llm.generate(long_prompt)

        assert response.usage is not None
        assert response.usage.total_tokens > 1000

    def test_model_switching(self):
        """测试模型切换"""
        config1 = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        config2 = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key"
        )

        llm1 = OpenAILLM(config1)
        llm2 = OpenAILLM(config2)

        assert llm1.config.model == "gpt-3.5-turbo"
        assert llm2.config.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llm):
        """测试并发请求"""
        prompts = ["提示1", "提示2", "提示3"]

        import asyncio
        tasks = [llm.generate(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for response in responses:
            assert response.content is not None


class TestClaudeLLM:
    """Claude LLM测试类"""

    @pytest.fixture
    def config(self):
        """Claude配置fixture"""
        return LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            api_key="test-key",
            max_tokens=1000,
            temperature=0.7
        )

    @pytest.fixture
    def mock_client(self):
        """模拟Claude客户端"""
        client = Mock()
        client.messages.create = AsyncMock(return_value=Mock(
            content=[{
                "type": "text",
                "text": "Claude测试响应"
            }],
            usage={
                "input_tokens": 12,
                "output_tokens": 8
            }
        ))
        return client

    @pytest.fixture
    def llm(self, config, mock_client):
        """Claude LLM实例fixture"""
        with pytest.MonkeyPatch('anthropic.Anthropic', return_value=mock_client):
            return ClaudeLLM(config)

    def test_claude_initialization(self, llm):
        """测试Claude初始化"""
        assert llm.config.provider == "anthropic"
        assert llm.config.model == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_claude_generate(self, llm):
        """测试Claude生成"""
        response = await llm.generate("你好Claude")

        assert response.content == "Claude测试响应"
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 8


class TestLocalLLM:
    """本地LLM测试类"""

    @pytest.fixture
    def config(self):
        """本地LLM配置fixture"""
        return LLMConfig(
            provider="local",
            model_path="/path/to/model",
            max_tokens=1000,
            temperature=0.7
        )

    @pytest.fixture
    def mock_model(self):
        """模拟本地模型"""
        model = Mock()
        model.generate = AsyncMock(return_value="本地模型响应")
        return model

    @pytest.fixture
    def llm(self, config, mock_model):
        """本地LLM实例fixture"""
        with pytest.MonkeyPatch('transformers.AutoModel.from_pretrained', return_value=mock_model):
            return LocalLLM(config)

    def test_local_initialization(self, llm):
        """测试本地LLM初始化"""
        assert llm.config.provider == "local"
        assert llm.config.model_path == "/path/to/model"

    @pytest.mark.asyncio
    async def test_local_generate(self, llm):
        """测试本地LLM生成"""
        response = await llm.generate("本地测试")

        assert response.content == "本地模型响应"

    def test_model_loading_error(self):
        """测试模型加载错误"""
        config = LLMConfig(
            provider="local",
            model_path="/non/existent/path"
        )

        with pytest.raises(FileNotFoundError):
            LocalLLM(config)


class TestLLMIntegration:
    """LLM集成测试类"""

    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """测试提供商切换"""
        configs = [
            LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key="test-key"),
            LLMConfig(provider="anthropic", model="claude-3-sonnet", api_key="test-key")
        ]

        llms = [OpenAILLM(config) for config in configs[:1]]
        llms.append(ClaudeLLM(configs[1]))

        prompts = ["测试OpenAI", "测试Claude"]

        import asyncio
        tasks = [llm.generate(prompts[i]) for i, llm in enumerate(llms)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 2
        for response in responses:
            assert response.content is not None

    @pytest.mark.asyncio
    async def test_failover_mechanism(self):
        """测试故障转移机制"""
        # 模拟主LLM失败
        def failing_generate(*args, **kwargs):
            raise Exception("主LLM失败")

        main_llm = Mock()
        main_llm.generate = failing_generate

        backup_llm = Mock()
        backup_llm.generate = AsyncMock(return_value=Mock(
            content="备用LLM响应",
            finish_reason="stop"
        ))

        # 实现故障转移逻辑
        try:
            response = await main_llm.generate("测试")
        except Exception:
            response = await backup_llm.generate("测试")

        assert response.content == "备用LLM响应"

    def test_performance_metrics(self):
        """测试性能指标收集"""
        config = LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key="test-key")

        llm = OpenAILLM(config)

        assert hasattr(llm, 'get_metrics')
        metrics = llm.get_metrics()

        assert 'total_requests' in metrics
        assert 'average_latency' in metrics
        assert 'success_rate' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
