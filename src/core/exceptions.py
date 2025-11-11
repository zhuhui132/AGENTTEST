"""
核心异常类定义
"""


class AgentError(Exception):
    """Agent基础异常"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class MemoryError(AgentError):
    """记忆系统异常"""
    pass


class ToolError(AgentError):
    """工具系统异常"""
    pass


class RAGError(AgentError):
    """RAG系统异常"""
    pass


class LLMError(AgentError):
    """LLM异常"""
    pass


class ContextError(AgentError):
    """上下文管理异常"""
    pass


class ConfigError(AgentError):
    """配置异常"""
    pass


class ValidationError(AgentError):
    """验证异常"""
    pass


class TimeoutError(AgentError):
    """超时异常"""
    pass


class RateLimitError(AgentError):
    """限流异常"""
    pass


class AuthenticationError(AgentError):
    """认证异常"""
    pass


class PermissionError(AgentError):
    """权限异常"""
    pass


class ResourceExhaustedError(AgentError):
    """资源耗尽异常"""
    pass


# 具体异常类型
class MemoryFullError(MemoryError):
    """记忆已满异常"""
    pass


class MemoryNotFoundError(MemoryError):
    """记忆未找到异常"""
    pass


class ToolNotFoundError(ToolError):
    """工具未找到异常"""
    pass


class ToolExecutionError(ToolError):
    """工具执行异常"""
    pass


class DocumentNotFoundError(RAGError):
    """文档未找到异常"""
    pass


class IndexingError(RAGError):
    """索引异常"""
    pass


class ModelLoadError(LLMError):
    """模型加载异常"""
    pass


class GenerationError(LLMError):
    """生成异常"""
    pass


class ContextTooLongError(ContextError):
    """上下文过长异常"""
    pass


class InvalidConfigError(ConfigError):
    """无效配置异常"""
    pass


class MissingParameterError(ValidationError):
    """缺少参数异常"""
    pass


class InvalidParameterError(ValidationError):
    """无效参数异常"""
    pass
