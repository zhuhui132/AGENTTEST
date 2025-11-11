"""
RAG系统集成测试
测试检索增强生成系统与Agent的集成
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.rag.retriever import Retriever
from src.rag.embeddings import Embeddings
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import KnowledgeBase
from src.agents.agent import IntelligentAgent
from src.core.types import RetrievalConfig, AgentConfig


class TestRAGIntegration:
    """RAG集成测试类"""

    @pytest.fixture
    def mock_embeddings(self):
        """模拟嵌入服务"""
        embeddings = Mock()
        embeddings.encode = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        return embeddings

    @pytest.fixture
    def mock_vector_store(self):
        """模拟向量存储"""
        store = Mock()
        store.search = Mock(return_value=[
            {"doc_id": "doc1", "score": 0.9},
            {"doc_id": "doc2", "score": 0.8},
            {"doc_id": "doc3", "score": 0.7}
        ])
        return store

    @pytest.fixture
    def mock_knowledge_base(self):
        """模拟知识库"""
        kb = Mock()
        kb.get_documents = Mock(return_value=[
            {"id": "doc1", "content": "机器学习是人工智能的分支"},
            {"id": "doc2", "content": "深度学习使用神经网络"},
            {"id": "doc3", "content": "自然语言处理是AI的重要应用"}
        ])
        return kb

    @pytest.fixture
    def retriever(self, mock_embeddings, mock_vector_store):
        """检索器fixture"""
        config = RetrievalConfig(top_k=3, threshold=0.5)
        return Retriever(embeddings=mock_embeddings, config=config)

    @pytest.fixture
    def rag_system(self, retriever, mock_knowledge_base, mock_vector_store):
        """RAG系统fixture"""
        class MockRAGSystem:
            def __init__(self):
                self.retriever = retriever
                self.knowledge_base = mock_knowledge_base
                self.vector_store = mock_vector_store

            async def retrieve_and_generate(self, query):
                # 模拟检索过程
                results = await self.retriever.retrieve(query, self.knowledge_base)
                return {
                    "query": query,
                    "retrieved_docs": results,
                    "context": self._build_context(results)
                }

            def _build_context(self, results):
                return " | ".join([
                    f"文档{i+1}: {r['document']['content']}"
                    for i, r in enumerate(results)
                ])

        return MockRAGSystem()

    @pytest.fixture
    def mock_llm(self):
        """模拟LLM"""
        llm = Mock()
        llm.generate = AsyncMock(return_value=Mock(
            content="基于检索到的文档，机器学习确实是人工智能的重要分支。",
            finish_reason="stop"
        ))
        return llm

    @pytest.fixture
    def agent_with_rag(self, rag_system, mock_llm):
        """带RAG的Agent fixture"""
        config = AgentConfig(model_name="test-model", enable_rag=True)
        agent = IntelligentAgent(config=config, llm=mock_llm)
        agent.rag_system = rag_system
        return agent

    @pytest.mark.asyncio
    async def test_rag_basic_query(self, agent_with_rag):
        """测试RAG基本查询"""
        query = "什么是机器学习？"

        response = await agent_with_rag.process_message(query)

        assert "机器学习" in response.content
        assert "人工智能" in response.content
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_rag_context_building(self, rag_system):
        """测试上下文构建"""
        query = "机器学习"

        result = await rag_system.retrieve_and_generate(query)

        assert "context" in result
        assert len(result["retrieved_docs"]) > 0
        assert "文档1" in result["context"]

    @pytest.mark.asyncio
    async def test_rag_empty_results_handling(self, agent_with_rag, rag_system):
        """测试RAG空结果处理"""
        # 模拟空检索结果
        rag_system.retriever.retrieve = AsyncMock(return_value=[])

        query = "完全无关的主题"
        response = await agent_with_rag.process_message(query)

        # 应该处理空结果，仍然给出回应
        assert response.content is not None
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_rag_relevance_filtering(self, rag_system):
        """测试相关性过滤"""
        query = "深度学习"

        # 模拟高相关性和低相关性文档
        rag_system.retriever.retrieve = AsyncMock(return_value=[
            {"doc_id": "doc1", "score": 0.9, "document": {"content": "深度学习使用神经网络"}},
            {"doc_id": "doc2", "score": 0.3, "document": {"content": "无关文档"}},
            {"doc_id": "doc3", "score": 0.1, "document": {"content": "更不相关"}}
        ])

        result = await rag_system.retrieve_and_generate(query)

        # 应该只包含高相关性文档
        relevant_docs = [r for r in result["retrieved_docs"] if r["score"] >= 0.5]
        assert len(relevant_docs) >= 1

    @pytest.mark.asyncio
    async def test_rag_concurrent_queries(self, agent_with_rag):
        """测试并发RAG查询"""
        queries = [
            "什么是机器学习？",
            "深度学习有什么特点？",
            "自然语言处理应用有哪些？"
        ]

        import asyncio
        tasks = [agent_with_rag.process_message(q) for q in queries]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.content is not None
            assert queries[i][0] in response.content or any(
                word in response.content for word in ["机器学习", "深度学习", "自然语言处理"]
            )

    @pytest.mark.asyncio
    async def test_rag_context_length_management(self, agent_with_rag, rag_system):
        """测试RAG上下文长度管理"""
        # 模拟长上下文
        long_docs = [{"doc_id": f"doc{i}", "content": f"文档{i} " * 100} for i in range(10)]
        rag_system.retriever.retrieve = AsyncMock(return_value=long_docs[:5])

        query = "需要长上下文的问题"
        response = await agent_with_rag.process_message(query)

        # 应该处理长上下文而不出错
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_rag_embedding_caching(self, agent_with_rag, rag_system, mock_embeddings):
        """测试RAG嵌入缓存"""
        query1 = "机器学习是什么？"
        query2 = "机器学习是什么？"

        # 第一次查询
        await agent_with_rag.process_message(query1)
        first_call_count = mock_embeddings.encode.call_count

        # 第二次相同查询
        await agent_with_rag.process_message(query2)
        second_call_count = mock_embeddings.encode.call_count

        # 应该缓存嵌入结果
        assert second_call_count <= first_call_count

    @pytest.mark.asyncio
    async def test_rag_document_updates(self, rag_system):
        """测试RAG文档更新"""
        query = "最新信息"

        # 初始查询
        result1 = await rag_system.retrieve_and_generate(query)

        # 更新知识库
        rag_system.knowledge_base.add_document({
            "id": "new_doc",
            "content": "最新的机器学习发展"
        })

        # 再次查询
        result2 = await rag_system.retrieve_and_generate(query)

        # 验证文档已更新
        docs1_count = len(result1["retrieved_docs"])
        docs2_count = len(result2["retrieved_docs"])
        assert docs2_count >= docs1_count

    @pytest.mark.asyncio
    async def test_rag_error_handling(self, agent_with_rag, rag_system):
        """测试RAG错误处理"""
        # 模拟检索错误
        rag_system.retriever.retrieve = AsyncMock(side_effect=Exception("检索错误"))

        with pytest.raises(Exception):
            await rag_system.retrieve_and_generate("测试查询")

    def test_rag_configuration_validation(self):
        """测试RAG配置验证"""
        # 有效配置
        valid_config = RetrievalConfig(top_k=5, threshold=0.7)
        assert valid_config.top_k == 5
        assert valid_config.threshold == 0.7

        # 无效配置
        with pytest.raises(ValueError):
            RetrievalConfig(top_k=0, threshold=0.5)

        with pytest.raises(ValueError):
            RetrievalConfig(top_k=5, threshold=1.5)

    @pytest.mark.asyncio
    async def test_rag_score_threshold_filtering(self, rag_system):
        """测试RAG分数阈值过滤"""
        # 模拟不同分数的文档
        rag_system.retriever.retrieve = AsyncMock(return_value=[
            {"doc_id": "doc1", "score": 0.9, "document": {"content": "高相关性"}},
            {"doc_id": "doc2", "score": 0.4, "document": {"content": "低相关性"}},
            {"doc_id": "doc3", "score": 0.2, "document": {"content": "很低相关性"}}
        ])

        result = await rag_system.retrieve_and_generate("测试")

        # 应该过滤掉低于阈值的文档
        high_score_docs = [r for r in result["retrieved_docs"] if r["score"] >= 0.5]
        low_score_docs = [r for r in result["retrieved_docs"] if r["score"] < 0.5]

        assert len(high_score_docs) >= 1
        assert len(low_score_docs) == 0

    @pytest.mark.asyncio
    async def test_rag_multi_language_support(self, rag_system):
        """测试RAG多语言支持"""
        queries = ["What is machine learning?", "什么是机器学习？"]

        results = []
        for query in queries:
            result = await rag_system.retrieve_and_generate(query)
            results.append(result)

        # 应该能处理不同语言
        for result in results:
            assert "context" in result
            assert len(result["retrieved_docs"]) > 0


class TestRAGPerformance:
    """RAG性能测试类"""

    @pytest.mark.asyncio
    async def test_rag_query_latency(self, agent_with_rag):
        """测试RAG查询延迟"""
        import time

        start_time = time.time()
        await agent_with_rag.process_message("测试查询")
        end_time = time.time()

        latency = end_time - start_time

        # 查询延迟应该在合理范围内
        assert latency < 10.0  # 10秒内

    @pytest.mark.asyncio
    async def test_rag_throughput(self, agent_with_rag):
        """测试RAG吞吐量"""
        import time
        import asyncio

        num_queries = 10
        queries = [f"查询{i}" for i in range(num_queries)]

        start_time = time.time()
        tasks = [agent_with_rag.process_message(q) for q in queries]
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time
        queries_per_second = num_queries / total_time

        # 应该能达到一定的吞吐量
        assert queries_per_second >= 1.0  # 至少每秒1个查询


if __name__ == "__main__":
    pytest.main([__file__])
