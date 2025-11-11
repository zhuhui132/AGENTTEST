"""
基础接口定义
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from .types import (
    AgentMessage, AgentResponse, MemoryItem, Document,
    ToolResult, ContextInfo, LLMConfig, AgentConfig
)


class BaseAgent(ABC):
    """Agent基础接口"""

    @abstractmethod
    async def process_message(
        self,
        message: AgentMessage,
        config: Optional[AgentConfig] = None
    ) -> AgentResponse:
        """处理消息的核心方法"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化Agent"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """关闭Agent"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        pass


class BaseMemory(ABC):
    """记忆系统基础接口"""

    @abstractmethod
    async def add_memory(
        self,
        content: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加记忆"""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[MemoryItem]:
        """检索记忆"""
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新记忆"""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    async def cleanup(self) -> int:
        """清理过期记忆"""
        pass


class BaseTool(ABC):
    """工具基础接口"""

    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行工具"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """获取工具参数模式"""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class BaseRAG(ABC):
    """RAG系统基础接口"""

    @abstractmethod
    async def add_document(
        self,
        title: str,
        content: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加文档"""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """检索文档"""
        pass

    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新文档"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        pass

    @abstractmethod
    async def rebuild_index(self) -> bool:
        """重建索引"""
        pass


class BaseLLM(ABC):
    """大语言模型基础接口"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> str:
        """生成文本"""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        pass

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """生成嵌入"""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass


class BaseContextManager(ABC):
    """上下文管理基础接口"""

    @abstractmethod
    async def build_context(
        self,
        messages: List[AgentMessage],
        max_length: int = 4000
    ) -> ContextInfo:
        """构建上下文"""
        pass

    @abstractmethod
    async def update_context(
        self,
        context_id: str,
        message: AgentMessage
    ) -> bool:
        """更新上下文"""
        pass

    @abstractmethod
    async def get_context(
        self,
        context_id: str
    ) -> Optional[ContextInfo]:
        """获取上下文"""
        pass

    @abstractmethod
    async def cleanup_contexts(self) -> int:
        """清理过期上下文"""
        pass


class BaseMetrics(ABC):
    """指标收集基础接口"""

    @abstractmethod
    async def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """记录指标"""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        time_range: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取指标"""
        pass

    @abstractmethod
    async def create_alert(
        self,
        condition: Dict[str, Any],
        action: str
    ) -> str:
        """创建告警"""
        pass


class BaseCache(ABC):
    """缓存基础接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass
