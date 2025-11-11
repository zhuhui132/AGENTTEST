"""
LLM集成测试
测试大语言模型与系统的集成功能
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.llm.base import BaseLLM
from src.llm.openai import OpenAILLM
from src.agents.agent import IntelligentAgent
from src.core.types import AgentConfig, LLMConfig


class TestLLMIntegration:
    """LLM集成测试类"""

    @pytest.fixture
    def llm_config(self):
        """LLM配置fixture"""
        return LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            max_tokens=1000,
            temperature=0.7
        )

    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "测试响应"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })
        return client

    @pytest.fixture
    def llm(self, llm_config, mock_openai_client):
        """LLM实例fixture"""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            return OpenAILLM(llm_config)

    @pytest.fixture
    def agent(self, llm):
        """Agent实例fixture"""
        config = AgentConfig(model_name="test-model")
        return IntelligentAgent(config=config, llm=llm)

    @pytest.mark.asyncio
    async def test_llm_basic_generation(self, llm):
        """测试LLM基本生成功能"""
        prompt = "请介绍一下自己"

        response = await llm.generate(prompt)

        assert response.content == "测试响应"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_llm_tool_calling(self, llm, mock_openai_client):
        """测试LLM工具调用功能"""
        # 模拟工具调用响应
        mock_openai_client.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_function",
                            "arguments": '{"arg1": "value1"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"total_tokens": 20}
        }

        response = await llm.generate("请调用测试工具")

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "test_function"

    @pytest.mark.asyncio
    async def test_llm_streaming_generation(self, llm, mock_openai_client):
        """测试LLM流式生成"""
        # 模拟流式响应
        chunks = [
            {"choices": [{"delta": {"content": "你"}}]},
            {"choices": [{"delta": {"content": "好"}}]},
            {"choices": [{"delta": {"content": "！"}}]}
        ]
        mock_openai_client.chat.completions.create.return_value = chunks

        responses = []
        async for chunk in llm.generate_stream("你好"):
            responses.append(chunk)

        assert len(responses) == 3
        assert "".join([r.content for r in responses]) == "你好！"

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, llm, mock_openai_client):
        """测试LLM错误处理"""
        # 模拟API错误
        mock_openai_client.chat.completions.create.side_effect = Exception("API错误")

        with pytest.raises(Exception):
            await llm.generate("测试")

    @pytest.mark.asyncio
    async def test_llm_token_limit(self, llm, mock_openai_client):
        """测试LLM令牌限制"""
        # 模拟超过令牌限制
        mock_openai_client.chat.completions.create.side_effect = Exception("超过最大令牌限制")

        with pytest.raises(Exception):
            await llm.generate("x" * 10000)  # 很长的文本

    @pytest.mark.asyncio
    async def test_agent_llm_integration(self, agent):
        """测试Agent与LLM的集成"""
        message = "你好，请介绍一下自己"

        response = await agent.process_message(message)

        assert response.content == "测试响应"
        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_concurrent_llm_requests(self, llm, mock_openai_client):
        """测试并发LLM请求"""
        messages = ["消息1", "消息2", "消息3"]

        import asyncio
        tasks = [llm.generate(msg) for msg in messages]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for response in responses:
            assert response.content == "测试响应"

    def test_llm_configuration_validation(self):
        """测试LLM配置验证"""
        # 有效配置
        valid_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        llm = OpenAILLM(valid_config)
        assert llm.config.provider == "openai"

        # 无效配置
        with pytest.raises(ValueError):
            OpenAILLM(LLMConfig(provider="openai", model="", api_key=""))

    def test_llm_rate_limiting(self, llm, mock_openai_client):
        """测试LLM速率限制"""
        # 模拟速率限制错误
        mock_openai_client.chat.completions.create.side_effect = Exception("速率限制")

        import pytest
        with pytest.raises(Exception):
            # 使用同步方式测试
            import asyncio
            asyncio.run(llm.generate("测试"))

    @pytest.mark.asyncio
    async def test_llm_context_window(self, llm, mock_openai_client):
        """测试LLM上下文窗口管理"""
        # 模拟上下文窗口溢出
        mock_openai_client.chat.completions.create.side_effect = Exception("上下文窗口溢出")

        long_context = "x" * 100000
        with pytest.raises(Exception):
            await llm.generate(long_context)

    @pytest.mark.asyncio
    async def test_llm_temperature_effect(self, llm, mock_openai_client):
        """测试温度参数效果"""
        # 模拟不同温度的响应
        mock_openai_client.chat.completions.create.side_effect = [
            {"choices": [{"message": {"content": "确定性响应"}}]},
            {"choices": [{"message": {"content": "创造性响应"}}]}
        ]

        # 低温度
        llm.config.temperature = 0.0
        response1 = await llm.generate("测试")

        # 高温度
        llm.config.temperature = 1.0
        response2 = await llm.generate("测试")

        assert response1.content != response2.content

    @pytest.mark.asyncio
    async def test_llm_max_tokens_enforcement(self, llm, mock_openai_client):
        """测试最大令牌数强制执行"""
        llm.config.max_tokens = 100

        response = await llm.generate("请生成一个很长的回应")

        # 验证调用时使用了正确的max_tokens
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args['max_tokens'] == 100

    @pytest.mark.asyncio
    async def test_llm_tool_response_parsing(self, llm, mock_openai_client):
        """测试LLM工具响应解析"""
        # 模拟复杂的工具调用响应
        mock_openai_client.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "content": "我将调用工具来帮助您",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test search", "limit": 5}'
                        }
                    }, {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression": "2+2"}'
                        }
                    }]
                }
            }]
        }

        response = await llm.generate("请搜索并计算")

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].function.name == "search"
        assert response.tool_calls[1].function.name == "calculate"


if __name__ == "__main__":
    pytest.main([__file__])
