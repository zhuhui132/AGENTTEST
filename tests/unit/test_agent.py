"""
Agent核心类单元测试
"""
import pytest
from datetime import datetime
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent import Agent

class TestAgentConstruction:
    """Agent构造测试"""

    def test_agent_creation_with_valid_name(self):
        """测试有效名称创建Agent"""
        agent = Agent("测试助手")
        assert agent.name == "测试助手"
        assert agent.session_id is not None
        assert agent.state == "initialized"
        assert len(agent.conversation_history) == 0

    def test_agent_creation_with_session_id(self):
        """测试指定session_id创建Agent"""
        session_id = "test-session-123"
        agent = Agent("测试助手", session_id)
        assert agent.session_id == session_id

    def test_agent_creation_empty_name(self):
        """测试空名称抛出异常"""
        with pytest.raises(ValueError, match="Agent名称不能为空"):
            Agent("")

    def test_agent_creation_whitespace_name(self):
        """测试空白名称抛出异常"""
        with pytest.raises(ValueError, match="Agent名称不能为空"):
            Agent("   ")

    def test_agent_creation_long_name(self):
        """测试超长名称抛出异常"""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Agent名称长度不能超过100字符"):
            Agent(long_name)

    def test_agent_creation_unicode_name(self):
        """测试Unicode名称"""
        unicode_name = "🤖测试助手🤖"
        agent = Agent(unicode_name)
        assert agent.name == unicode_name

    def test_agent_name_trim_whitespace(self):
        """测试名称自动去除空格"""
        agent = Agent("  测试助手  ")
        assert agent.name == "测试助手"

    def test_agent_components_initialization(self):
        """测试组件初始化"""
        agent = Agent("测试助手")

        # 检查核心组件是否正确初始化
        assert agent.memory is not None
        assert agent.rag is not None
        assert agent.tools is not None
        assert agent.context is not None

    def test_agent_created_at_timestamp(self):
        """测试创建时间戳"""
        before = datetime.now()
        agent = Agent("测试助手")
        after = datetime.now()

        assert before <= agent.created_at <= after

class TestMessageProcessing:
    """消息处理测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("测试助手")

    def test_process_valid_message(self):
        """测试处理有效消息"""
        message = "你好"
        result = self.agent.process_message(message)

        assert "response" in result
        assert "context" in result
        assert "memories_used" in result
        assert "docs_used" in result
        assert len(self.agent.conversation_history) == 2  # 用户+助手

    def test_process_empty_message(self):
        """测试空消息抛出异常"""
        with pytest.raises(ValueError, match="消息内容不能为空"):
            self.agent.process_message("")

    def test_process_whitespace_message(self):
        """测试空白消息抛出异常"""
        with pytest.raises(ValueError, match="消息内容不能为空"):
            self.agent.process_message("   ")

    def test_process_long_message(self):
        """测试超长消息"""
        long_message = "测试" * 1000  # 4000字符
        result = self.agent.process_message(long_message)
        assert "response" in result

    def test_process_message_with_unicode(self):
        """测试Unicode消息"""
        unicode_message = "🌟你好世界🌟"
        result = self.agent.process_message(unicode_message)
        assert "response" in result

    def test_process_message_with_context(self):
        """测试带上下文的消息处理"""
        message = "查询天气"
        context = {"user_id": "123", "location": "北京"}
        result = self.agent.process_message(message, context)

        assert result["context"] is not None
        assert len(self.agent.conversation_history) == 2

    def test_conversation_history_accumulation(self):
        """测试对话历史累积"""
        messages = ["你好", "今天天气如何", "谢谢"]

        for msg in messages:
            self.agent.process_message(msg)

        assert len(self.agent.conversation_history) == 6  # 3用户+3助手

        # 检查历史记录结构
        for i, entry in enumerate(self.agent.conversation_history):
            assert "role" in entry
            assert "content" in entry
            assert "timestamp" in entry
            if i % 2 == 0:  # 用户消息
                assert entry["role"] == "user"
            else:  # 助手消息
                assert entry["role"] == "assistant"

    def test_message_processing_error_state(self):
        """测试处理错误时的状态变化"""
        # 模拟处理错误
        original_retrieve = self.agent.memory.retrieve
        self.agent.memory.retrieve = lambda x: (_ for _ in ()).throw(Exception("模拟错误"))

        try:
            self.agent.process_message("测试消息")
        except RuntimeError:
            assert self.agent.state == "error"
        finally:
            # 恢复原始方法
            self.agent.memory.retrieve = original_retrieve

class TestAgentState:
    """Agent状态测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("测试助手")

    def test_initial_state(self):
        """测试初始状态"""
        state = self.agent.get_state()

        assert state["name"] == "测试助手"
        assert state["session_id"] is not None
        assert state["state"] == "initialized"
        assert state["conversation_count"] == 0
        assert "created_at" in state

    def test_state_after_message_processing(self):
        """测试消息处理后的状态"""
        self.agent.process_message("测试消息")
        state = self.agent.get_state()

        assert state["conversation_count"] == 2  # 用户+助手
        assert state["state"] == "initialized"  # 正常状态下不改变

    def test_session_id_uniqueness(self):
        """测试session ID唯一性"""
        agent1 = Agent("助手1")
        agent2 = Agent("助手2")

        assert agent1.session_id != agent2.session_id

class TestResponseGeneration:
    """响应生成测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("测试助手")

    def test_response_generation_with_context(self):
        """测试带上下文的响应生成"""
        response = self.agent._generate_response(
            "测试消息",
            {"summary": "测试上下文"},
            [{"content": "相关记忆"}],
            [{"content": "相关文档"}]
        )

        assert "基于上下文理解" in response
        assert "结合了1条相关记忆" in response
        assert "参考了1篇文档" in response
        assert "对消息'测试消息'的回复" in response

    def test_response_generation_without_context(self):
        """测试无上下文的响应生成"""
        response = self.agent._generate_response(
            "测试消息",
            {},
            [],
            []
        )

        assert "对消息'测试消息'的回复" in response

    def test_response_generation_with_multiple_memories(self):
        """测试多个记忆的响应生成"""
        memories = [{"content": "记忆1"}, {"content": "记忆2"}]
        response = self.agent._generate_response(
            "测试消息",
            {},
            memories,
            []
        )

        assert "结合了2条相关记忆" in response

class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        """测试前置设置"""
        self.agent = Agent("测试助手")

    def test_special_characters_in_message(self):
        """测试消息中的特殊字符"""
        special_message = "测试<>{}[]|\\\"'`~!@#$%^&*()_+-="
        result = self.agent.process_message(special_message)
        assert "response" in result

    def test_newline_characters_in_message(self):
        """测试消息中的换行符"""
        multiline_message = "第一行\n第二行\n第三行"
        result = self.agent.process_message(multiline_message)
        assert "response" in result

    def test_tab_characters_in_message(self):
        """测试消息中的制表符"""
        tab_message = "列1\t列2\t列3"
        result = self.agent.process_message(tab_message)
        assert "response" in result

    def test_extremely_long_single_word(self):
        """测试极长单词"""
        long_word = "a" * 1000
        result = self.agent.process_message(long_word)
        assert "response" in result
