"""
核心数据类型定义
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class AgentState(Enum):
    """Agent状态枚举"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    THINKING = "thinking"
    RESPONDING = "responding"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    SYSTEM = "system"


class ToolStatus(Enum):
    """工具状态枚举"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class AgentMessage:
    """消息数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # user, assistant, system, tool
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    token_count: int = 0
    processing_time: float = 0.0
    priority: int = 0  # 消息优先级
    expires_at: Optional[datetime] = None


@dataclass
class AgentResponse:
    """Agent响应结构"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None


@dataclass
class Document:
    """文档数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    source: str = ""
    url: Optional[str] = None
    content_type: str = "text"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str = ""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextInfo:
    """上下文信息"""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_window: List[AgentMessage] = field(default_factory=list)
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    user_intent: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "default"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    timeout: float = 30.0


@dataclass
class AgentConfig:
    """Agent配置"""
    name: str = "default_agent"
    version: str = "1.0.0"
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    max_context_length: int = 4000
    memory_enabled: bool = True
    rag_enabled: bool = True
    tools_enabled: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"

    # 性能配置
    max_concurrent_requests: int = 10
    response_timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    max_memories: int = 10000
    default_importance: float = 1.0
    importance_decay_rate: float = 0.99
    retrieval_limit: int = 5
    similarity_threshold: float = 0.7
    cleanup_interval: int = 3600  # seconds


@dataclass
class ToolConfig:
    """工具系统配置"""
    max_tools: int = 100
    default_timeout: float = 30.0
    retry_attempts: int = 3
    parallel_execution: bool = True
    max_parallel_tools: int = 5


@dataclass
class RAGConfig:
    """RAG系统配置"""
    max_documents: int = 10000
    retrieval_limit: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "default"
    reranking_enabled: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50


# 类型别名
JSONType = Dict[str, Any]
MessageList = List[AgentMessage]
MemoryList = List[MemoryItem]
DocumentList = List[Document]
ToolMap = Dict[str, Callable]
