"""Agent 工厂：根据配置组合记忆、RAG、工具与 LLM。"""

from __future__ import annotations

from typing import Optional

from ..core.config import ConfigManager
from ..core.exceptions import LLMError
from ..core.types import AgentConfig, LLMConfig, MemoryConfig, RAGConfig, ToolConfig
from ..llm import LLMManager, MockLLM
from ..memory.memory import MemorySystem
from ..rag.rag import RAGSystem
from ..utils.tools import ToolSystem
from ..utils.context import ContextManager
from .agent import IntelligentAgent


class AgentFactory:
    """负责加载配置并实例化各子系统。"""

    def __init__(self, config_manager: Optional[ConfigManager] = None, *, llm_manager: Optional[LLMManager] = None):
        self._config_manager = config_manager
        self._llm_manager = llm_manager or LLMManager()

    async def create_agent(
        self,
        agent_config: Optional[AgentConfig] = None,
        *,
        load_profile: Optional[str] = None,
    ) -> IntelligentAgent:
        cfg = agent_config or self._load_agent_config(load_profile)
        memory_cfg = self._load_memory_config(load_profile)
        rag_cfg = self._load_rag_config(load_profile)
        tool_cfg = self._load_tool_config(load_profile)

        memory = MemorySystem(memory_cfg)
        rag = RAGSystem(rag_cfg)
        tools = ToolSystem(tool_cfg)
        context = ContextManager()

        llm_config = cfg.llm_config if isinstance(cfg.llm_config, LLMConfig) else LLMConfig()
        try:
            llm = await self._llm_manager.get(llm_config.model_name, config=llm_config)
        except LLMError:
            llm = MockLLM(llm_config)
            await llm.initialize()

        agent = IntelligentAgent(
            config=cfg,
            llm=llm,
            llm_config=llm_config,
            memory_system=memory,
            rag_system=rag,
            tools=tools,
            context_manager=context,
        )
        return agent

    def _load_agent_config(self, profile: Optional[str]) -> AgentConfig:
        if self._config_manager:
            name = profile or "default"
            try:
                return self._config_manager.get_agent_config(name)
            except Exception:
                pass
        return AgentConfig()

    def _load_memory_config(self, profile: Optional[str]) -> MemoryConfig:
        if self._config_manager:
            name = profile or "default"
            try:
                return self._config_manager.get_memory_config(name)
            except Exception:
                pass
        return MemoryConfig()

    def _load_rag_config(self, profile: Optional[str]) -> RAGConfig:
        if self._config_manager:
            name = profile or "default"
            try:
                return self._config_manager.get_rag_config(name)
            except Exception:
                pass
        return RAGConfig()

    def _load_tool_config(self, profile: Optional[str]) -> ToolConfig:
        if self._config_manager:
            name = profile or "default"
            try:
                return self._config_manager.get_tool_config(name)
            except Exception:
                pass
        return ToolConfig()

