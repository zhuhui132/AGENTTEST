"""
核心模块测试
测试配置、类型定义、异常处理等核心功能
"""

import pytest
import yaml
from src.core.config import ConfigManager
from src.core.types import AgentConfig, LLMConfig, RetrievalConfig
from src.core.exceptions import (
    AgentError,
    ConfigError,
    ValidationError,
    LLMError,
    MemoryError
)
from src.core.logging import Logger


class TestConfigManager:
    """配置管理器测试类"""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """临时配置文件fixture"""
        config_data = {
            "model": {
                "name": "test-model",
                "provider": "openai",
                "api_key": "test-key"
            },
            "memory": {
                "max_size": 1000,
                "retention_days": 30
            },
            "retrieval": {
                "top_k": 5,
                "threshold": 0.7
            }
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        return str(config_file)

    def test_load_valid_config(self, temp_config_file):
        """测试加载有效配置"""
        config_manager = ConfigManager()
        config = config_manager.load_config(temp_config_file)

        assert config is not None
        assert config["model"]["name"] == "test-model"
        assert config["memory"]["max_size"] == 1000

    def test_load_invalid_config_file(self):
        """测试加载无效配置文件"""
        config_manager = ConfigManager()

        with pytest.raises(ConfigError):
            config_manager.load_config("non_existent_file.yaml")

    def test_load_malformed_config(self, tmp_path):
        """测试加载格式错误的配置"""
        malformed_file = tmp_path / "malformed.yaml"
        with open(malformed_file, 'w') as f:
            f.write("invalid: yaml: content:")

        config_manager = ConfigManager()
        with pytest.raises(ConfigError):
            config_manager.load_config(str(malformed_file))

    def test_save_config(self, temp_config_file):
        """测试保存配置"""
        config_manager = ConfigManager()
        original_config = config_manager.load_config(temp_config_file)

        # 修改配置
        original_config["model"]["name"] = "modified-model"

        # 保存到新文件
        new_file = temp_config_file.replace(".yaml", "_new.yaml")
        config_manager.save_config(original_config, new_file)

        # 验证保存
        new_config = config_manager.load_config(new_file)
        assert new_config["model"]["name"] == "modified-model"

    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigManager()

        # 有效配置
        valid_config = {
            "model": {"name": "test"},
            "memory": {"max_size": 100}
        }

        try:
            config_manager.validate_config(valid_config)
        except ConfigError:
            pytest.fail("有效配置不应该抛出异常")

        # 无效配置
        invalid_configs = [
            {},  # 空配置
            {"model": {}},  # 缺少必要字段
            {"memory": {"max_size": -1}},  # 无效值
            {"retrieval": {"top_k": 0}}  # 无效值
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ConfigError):
                config_manager.validate_config(invalid_config)

    def test_environment_override(self, temp_config_file):
        """测试环境配置覆盖"""
        config_manager = ConfigManager()

        # 设置环境变量
        import os
        os.environ['AGENT_MODEL_NAME'] = 'env-override-model'

        config = config_manager.load_config(temp_config_file)

        assert config["model"]["name"] == "env-override-model"


class TestTypes:
    """类型定义测试类"""

    def test_agent_config_creation(self):
        """测试Agent配置创建"""
        config = AgentConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7,
            enable_memory=True,
            enable_rag=True
        )

        assert config.model_name == "test-model"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.enable_memory is True
        assert config.enable_rag is True

    def test_agent_config_validation(self):
        """测试Agent配置验证"""
        # 有效配置
        try:
            AgentConfig(
                model_name="test-model",
                max_tokens=1000,
                temperature=0.7
            )
        except ValidationError:
            pytest.fail("有效配置不应该抛出异常")

        # 无效配置
        invalid_configs = [
            {"model_name": ""},  # 空字符串
            {"max_tokens": -1},  # 负数
            {"temperature": 2.0},  # 超出范围
            {"max_tokens": "invalid"}  # 错误类型
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValidationError):
                AgentConfig(**invalid_config)

    def test_llm_config_creation(self):
        """测试LLM配置创建"""
        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )

        assert config.provider == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"

    def test_retrieval_config_creation(self):
        """测试检索配置创建"""
        config = RetrievalConfig(
            top_k=10,
            threshold=0.8,
            include_metadata=True,
            similarity_metric="cosine"
        )

        assert config.top_k == 10
        assert config.threshold == 0.8
        assert config.include_metadata is True
        assert config.similarity_metric == "cosine"

    def test_config_serialization(self):
        """测试配置序列化"""
        config = AgentConfig(model_name="test", max_tokens=1000)

        # 转换为字典
        config_dict = config.dict()
        assert config_dict["model_name"] == "test"
        assert config_dict["max_tokens"] == 1000

        # 从字典重建
        config_restored = AgentConfig(**config_dict)
        assert config_restored.model_name == config.model_name
        assert config_restored.max_tokens == config.max_tokens


class TestExceptions:
    """异常处理测试类"""

    def test_agent_error_creation(self):
        """测试Agent异常创建"""
        error = AgentError("测试错误", error_code="TEST_001")

        assert str(error) == "测试错误"
        assert error.error_code == "TEST_001"

    def test_config_error_creation(self):
        """测试配置异常创建"""
        error = ConfigError("配置文件格式错误", file_path="/path/to/config")

        assert "配置文件格式错误" in str(error)
        assert error.file_path == "/path/to/config"

    def test_validation_error_creation(self):
        """测试验证异常创建"""
        error = ValidationError(
            "字段值无效",
            field="max_tokens",
            value=-1,
            expected="positive integer"
        )

        assert "字段值无效" in str(error)
        assert error.field == "max_tokens"
        assert error.value == -1
        assert error.expected == "positive integer"

    def test_exception_hierarchy(self):
        """测试异常层次结构"""
        try:
            raise ConfigError("配置错误", error_code="CONFIG_001")
        except AgentError as e:
            assert isinstance(e, ConfigError)
            assert e.error_code == "CONFIG_001"
        else:
            pytest.fail("应该是AgentError的子类")

    def test_exception_chaining(self):
        """测试异常链"""
        try:
            try:
                raise ConfigError("原始错误")
            except ConfigError as original:
                raise LLMError("LLM错误") from original
        except LLMError as final:
            assert str(final) == "LLM错误"
            assert final.__cause__ is not None
            assert isinstance(final.__cause__, ConfigError)


class TestLogger:
    """日志系统测试类"""

    @pytest.fixture
    def temp_log_file(self, tmp_path):
        """临时日志文件fixture"""
        return str(tmp_path / "test.log")

    def test_logger_creation(self, temp_log_file):
        """测试日志创建"""
        logger = Logger("test_logger", file_path=temp_log_file)

        assert logger.name == "test_logger"
        assert logger.file_path == temp_log_file

    def test_log_levels(self, temp_log_file):
        """测试日志级别"""
        logger = Logger("test_logger", file_path=temp_log_file)

        logger.debug("调试信息")
        logger.info("信息日志")
        logger.warning("警告信息")
        logger.error("错误信息")
        logger.critical("严重错误")

        # 验证日志文件存在
        import os
        assert os.path.exists(temp_log_file)

    def test_structured_logging(self, temp_log_file):
        """测试结构化日志"""
        logger = Logger("test_logger", file_path=temp_log_file)

        logger.info("用户操作", extra={
            "user_id": "user123",
            "action": "login",
            "timestamp": "2023-01-01T00:00:00Z"
        })

        # 验证结构化信息
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            assert "user_id" in log_content
            assert "action" in log_content
            assert "timestamp" in log_content

    def test_log_rotation(self, temp_log_file):
        """测试日志轮转"""
        # 创建小日志文件限制
        logger = Logger("test_logger", file_path=temp_log_file, max_size=1024)

        # 写入大量日志触发轮转
        for i in range(100):
            logger.info(f"日志消息{i}" * 10)

        # 验证轮转
        import os
        log_files = [f for f in os.listdir(os.path.dirname(temp_log_file))
                   if f.startswith(os.path.basename(temp_log_file).replace('.log', ''))]

        assert len(log_files) >= 1  # 应该有轮转文件

    def test_performance_logging(self, temp_log_file):
        """测试性能日志"""
        logger = Logger("test_logger", file_path=temp_log_file)

        # 模拟性能敏感操作
        logger.info("开始处理", extra={
            "operation": "data_processing",
            "start_time": 1234567890.123
        })

        logger.info("处理完成", extra={
            "operation": "data_processing",
            "end_time": 1234567895.123,
            "duration": 5.0
        })

        # 验证性能日志
        with open(temp_log_file, 'r') as f:
            log_content = f.read()
            assert "duration" in log_content
            assert "operation" in log_content

    def test_sensitive_data_filtering(self, temp_log_file):
        """测试敏感数据过滤"""
        logger = Logger("test_logger", file_path=temp_log_file)

        # 记录包含敏感信息
        logger.info("用户登录", extra={
            "user_id": "user123",
            "password": "secret123",  # 敏感数据
            "credit_card": "1234-5678-9012-3456"  # 敏感数据
        })

        # 验证敏感数据被过滤
        with open(temp_log_file, 'r') as f:
            log_content = f.read()
            assert "user123" in log_content
            assert "secret123" not in log_content
            assert "1234-5678-9012-3456" not in log_content

    def test_log_level_filtering(self, temp_log_file):
        """测试日志级别过滤"""
        # 设置高日志级别
        logger = Logger("test_logger", level="ERROR", file_path=temp_log_file)

        logger.debug("调试信息")  # 不应该记录
        logger.info("信息日志")   # 不应该记录
        logger.warning("警告信息")  # 不应该记录
        logger.error("错误信息")    # 应该记录

        # 验证只有错误及以上级别被记录
        with open(temp_log_file, 'r') as f:
            log_content = f.read()
            assert "错误信息" in log_content
            assert "调试信息" not in log_content
            assert "信息日志" not in log_content
            assert "警告信息" not in log_content


if __name__ == "__main__":
    pytest.main([__file__])
