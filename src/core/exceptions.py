"""
核心异常类定义
"""

from typing import Dict, Any, Optional
import traceback
import logging

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Agent基础异常

    提供统一的异常处理机制，包含错误代码、详细信息和追踪信息
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AGENT_ERROR"
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        self.timestamp = None
        self.traceback_str = traceback.format_exc() if cause else None

        # 记录异常日志
        logger.error(
            f"AgentError: {message} | Code: {self.error_code} | Details: {self.details}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化和日志记录"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str
        }

    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        if self.cause:
            base_msg += f" | Cause: {self.cause}"
        return base_msg


class MemoryError(AgentError):
    """记忆系统异常"""

    def __init__(self, message: str, memory_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MEMORY_ERROR", **kwargs)
        self.memory_id = memory_id
        if memory_id:
            self.details["memory_id"] = memory_id


class ToolError(AgentError):
    """工具系统异常"""

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="TOOL_ERROR", **kwargs)
        self.tool_name = tool_name
        if tool_name:
            self.details["tool_name"] = tool_name


class RAGError(AgentError):
    """RAG系统异常"""

    def __init__(self, message: str, document_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RAG_ERROR", **kwargs)
        self.document_id = document_id
        if document_id:
            self.details["document_id"] = document_id


class LLMError(AgentError):
    """LLM异常"""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)
        self.model_name = model_name
        if model_name:
            self.details["model_name"] = model_name


class ContextError(AgentError):
    """上下文管理异常"""

    def __init__(self, message: str, context_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONTEXT_ERROR", **kwargs)
        self.context_id = context_id
        if context_id:
            self.details["context_id"] = context_id


class ConfigError(AgentError):
    """配置异常"""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class ValidationError(AgentError):
    """验证异常"""

    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field_name = field_name
        if field_name:
            self.details["field_name"] = field_name


class TimeoutError(AgentError):
    """超时异常"""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class RateLimitError(AgentError):
    """限流异常"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="RATE_LIMIT_ERROR", **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class AuthenticationError(AgentError):
    """认证异常"""

    def __init__(self, message: str, auth_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="AUTH_ERROR", **kwargs)
        self.auth_type = auth_type
        if auth_type:
            self.details["auth_type"] = auth_type


class PermissionError(AgentError):
    """权限异常"""

    def __init__(self, message: str, resource: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="PERMISSION_ERROR", **kwargs)
        self.resource = resource
        if resource:
            self.details["resource"] = resource


class ResourceExhaustedError(AgentError):
    """资源耗尽异常"""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_EXHAUSTED", **kwargs)
        self.resource_type = resource_type
        if resource_type:
            self.details["resource_type"] = resource_type


# 具体异常类型
class MemoryFullError(MemoryError):
    """记忆已满异常"""

    def __init__(self, message: str = "记忆存储已满", max_capacity: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "MEMORY_FULL"
        if max_capacity:
            self.details["max_capacity"] = max_capacity


class MemoryNotFoundError(MemoryError):
    """记忆未找到异常"""

    def __init__(self, message: str = "记忆未找到", query: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "MEMORY_NOT_FOUND"
        if query:
            self.details["query"] = query


class ToolNotFoundError(ToolError):
    """工具未找到异常"""

    def __init__(self, message: str = "工具未找到", available_tools: Optional[list] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "TOOL_NOT_FOUND"
        if available_tools:
            self.details["available_tools"] = available_tools


class ToolExecutionError(ToolError):
    """工具执行异常"""

    def __init__(self, message: str = "工具执行失败", execution_time: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "TOOL_EXECUTION_ERROR"
        if execution_time:
            self.details["execution_time"] = execution_time


class DocumentNotFoundError(RAGError):
    """文档未找到异常"""

    def __init__(self, message: str = "文档未找到", search_query: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "DOCUMENT_NOT_FOUND"
        if search_query:
            self.details["search_query"] = search_query


class IndexingError(RAGError):
    """索引异常"""

    def __init__(self, message: str = "索引操作失败", operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "INDEXING_ERROR"
        if operation:
            self.details["operation"] = operation


class ModelLoadError(LLMError):
    """模型加载异常"""

    def __init__(self, message: str = "模型加载失败", model_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "MODEL_LOAD_ERROR"
        if model_path:
            self.details["model_path"] = model_path


class GenerationError(LLMError):
    """生成异常"""

    def __init__(self, message: str = "文本生成失败", prompt_length: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "GENERATION_ERROR"
        if prompt_length:
            self.details["prompt_length"] = prompt_length


class ContextTooLongError(ContextError):
    """上下文过长异常"""

    def __init__(self, message: str = "上下文过长", context_length: Optional[int] = None, max_length: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "CONTEXT_TOO_LONG"
        if context_length:
            self.details["context_length"] = context_length
        if max_length:
            self.details["max_length"] = max_length


class InvalidConfigError(ConfigError):
    """无效配置异常"""

    def __init__(self, message: str = "配置无效", config_value: Optional[Any] = None, expected_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "INVALID_CONFIG"
        if config_value is not None:
            self.details["config_value"] = str(config_value)
        if expected_type:
            self.details["expected_type"] = expected_type


class MissingParameterError(ValidationError):
    """缺少参数异常"""

    def __init__(self, message: str = "缺少必需参数", parameter_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "MISSING_PARAMETER"
        if parameter_name:
            self.details["parameter_name"] = parameter_name


class InvalidParameterError(ValidationError):
    """无效参数异常"""

    def __init__(self, message: str = "参数无效", parameter_name: Optional[str] = None, parameter_value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = "INVALID_PARAMETER"
        if parameter_name:
            self.details["parameter_name"] = parameter_name
        if parameter_value is not None:
            self.details["parameter_value"] = str(parameter_value)


# 新增异常类型
class NetworkError(AgentError):
    """网络异常"""

    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.url = url
        self.status_code = status_code
        if url:
            self.details["url"] = url
        if status_code:
            self.details["status_code"] = status_code


class DatabaseError(AgentError):
    """数据库异常"""

    def __init__(self, message: str, query: Optional[str] = None, table: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)
        self.query = query
        self.table = table
        if query:
            self.details["query"] = query
        if table:
            self.details["table"] = table


class FileOperationError(AgentError):
    """文件操作异常"""

    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FILE_ERROR", **kwargs)
        self.file_path = file_path
        self.operation = operation
        if file_path:
            self.details["file_path"] = file_path
        if operation:
            self.details["operation"] = operation


class ConcurrencyError(AgentError):
    """并发异常"""

    def __init__(self, message: str, resource_id: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONCURRENCY_ERROR", **kwargs)
        self.resource_id = resource_id
        self.operation = operation
        if resource_id:
            self.details["resource_id"] = resource_id
        if operation:
            self.details["operation"] = operation


class ServiceUnavailableError(AgentError):
    """服务不可用异常"""

    def __init__(self, message: str, service_name: Optional[str] = None, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", **kwargs)
        self.service_name = service_name
        self.retry_after = retry_after
        if service_name:
            self.details["service_name"] = service_name
        if retry_after:
            self.details["retry_after"] = retry_after
