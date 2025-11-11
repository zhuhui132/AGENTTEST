"""
RAG系统基础功能测试
测试检索增强生成系统的核心功能
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.rag.retriever import Retriever
from src.rag.embeddings import Embeddings
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import KnowledgeBase
from src.core.types import RetrievalConfig


class TestEmbeddings:
    """嵌入功能测试类"""

    @pytest.fixture
    def mock_embeddings(self):
        """模拟嵌入服务"""
        embeddings = Mock()
        embeddings.encode = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        return embeddings

    def test_encode_text(self, mock_embeddings):
        """测试文本编码"""
        text = "这是一个测试文本"

        result = mock_embeddings.encode(text)

        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(x, float) for x in result)

    def test_batch_encode(self, mock_embeddings):
        """测试批量编码"""
        texts = ["文本1", "文本2", "文本3"]

        result = mock_embeddings.encode(texts)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_empty_text_handling(self, mock_embeddings):
        """测试空文本处理"""
        with pytest.raises(ValueError):
            mock_embeddings.encode("")

    def test_embedding_dimensions(self, mock_embeddings):
        """测试嵌入维度一致性"""
        texts = ["短文本", "这是一个长文本，用来测试不同长度的文本处理"]

        results = [mock_embeddings.encode(text) for text in texts]

        for result in results:
            assert len(result) == 5


class TestVectorStore:
    """向量存储测试类"""

    @pytest.fixture
    def vector_store(self):
        """向量存储fixture"""
        return VectorStore(dimensions=5)

    @pytest.fixture
    def sample_vectors(self):
        """示例向量"""
        return [
            ("doc1", [0.1, 0.2, 0.3, 0.4, 0.5]),
            ("doc2", [0.6, 0.7, 0.8, 0.9, 1.0]),
            ("doc3", [0.2, 0.3, 0.4, 0.5, 0.6])
        ]

    def test_vector_store_initialization(self, vector_store):
        """测试向量存储初始化"""
        assert vector_store.dimensions == 5
        assert len(vector_store.vectors) == 0

    def test_add_vector(self, vector_store, sample_vectors):
        """测试添加向量"""
        doc_id, vector = sample_vectors[0]

        success = vector_store.add(doc_id, vector)

        assert success is True
        assert len(vector_store.vectors) == 1
        assert doc_id in vector_store.vectors

    def test_search_vectors(self, vector_store, sample_vectors):
        """测试向量搜索"""
        # 添加示例向量
        for doc_id, vector in sample_vectors:
            vector_store.add(doc_id, vector)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = vector_store.search(query_vector, top_k=2)

        assert len(results) <= 2
        for result in results:
            assert 'doc_id' in result
            assert 'score' in result
            assert 0 <= result['score'] <= 1

    def test_delete_vector(self, vector_store, sample_vectors):
        """测试删除向量"""
        doc_id, vector = sample_vectors[0]
        vector_store.add(doc_id, vector)

        success = vector_store.delete(doc_id)

        assert success is True
        assert len(vector_store.vectors) == 0

    def test_dimension_validation(self, vector_store):
        """测试维度验证"""
        wrong_dimension_vector = [0.1, 0.2, 0.3]

        with pytest.raises(ValueError):
            vector_store.add("test", wrong_dimension_vector)


class TestRetriever:
    """检索器测试类"""

    @pytest.fixture
    def retriever(self, mock_embeddings):
        """检索器fixture"""
        config = RetrievalConfig(top_k=3, threshold=0.5)
        return Retriever(embeddings=mock_embeddings, config=config)

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

    def test_retriever_initialization(self, retriever, mock_embeddings):
        """测试检索器初始化"""
        assert retriever.embeddings == mock_embeddings
        assert retriever.config.top_k == 3
        assert retriever.config.threshold == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_documents(self, retriever, mock_knowledge_base):
        """测试文档检索"""
        query = "什么是机器学习？"

        results = await retriever.retrieve(query, knowledge_base=mock_knowledge_base)

        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert 'document' in result
            assert 'score' in result
            assert result['score'] >= 0.5

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, retriever, mock_knowledge_base):
        """测试相关性评分"""
        query = "深度学习"

        results = await retriever.retrieve(query, knowledge_base=mock_knowledge_base)

        # 找到包含"深度学习"的文档应该有更高分数
        dl_scores = [r['score'] for r in results if '深度学习' in r['document']['content']]
        other_scores = [r['score'] for r in results if '深度学习' not in r['document']['content']]

        if dl_scores and other_scores:
            assert max(dl_scores) >= max(other_scores)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retriever, mock_knowledge_base):
        """测试空查询处理"""
        with pytest.raises(ValueError):
            await retriever.retrieve("", knowledge_base=mock_knowledge_base)

    @pytest.mark.asyncio
    async def test_threshold_filtering(self, retriever, mock_knowledge_base):
        """测试阈值过滤"""
        query = "查询"

        results = await retriever.retrieve(query, knowledge_base=mock_knowledge_base)

        for result in results:
            assert result['score'] >= retriever.config.threshold


class TestKnowledgeBase:
    """知识库测试类"""

    @pytest.fixture
    def knowledge_base(self):
        """知识库fixture"""
        return KnowledgeBase()

    @pytest.fixture
    def sample_documents(self):
        """示例文档"""
        return [
            {"id": "doc1", "content": "Python是一种编程语言", "metadata": {"category": "programming"}},
            {"id": "doc2", "content": "机器学习是AI的分支", "metadata": {"category": "AI"}},
            {"id": "doc3", "content": "深度学习使用神经网络", "metadata": {"category": "AI"}}
        ]

    def test_knowledge_base_initialization(self, knowledge_base):
        """测试知识库初始化"""
        assert len(knowledge_base.documents) == 0
        assert knowledge_base.index is None

    def test_add_document(self, knowledge_base, sample_documents):
        """测试添加文档"""
        doc = sample_documents[0]

        success = knowledge_base.add_document(doc)

        assert success is True
        assert len(knowledge_base.documents) == 1
        assert doc["id"] in [d["id"] for d in knowledge_base.documents]

    def test_get_document(self, knowledge_base, sample_documents):
        """测试获取文档"""
        doc = sample_documents[0]
        knowledge_base.add_document(doc)

        retrieved = knowledge_base.get_document(doc["id"])

        assert retrieved is not None
        assert retrieved["id"] == doc["id"]
        assert retrieved["content"] == doc["content"]

    def test_update_document(self, knowledge_base, sample_documents):
        """测试更新文档"""
        doc = sample_documents[0]
        knowledge_base.add_document(doc)

        updated_content = "Python是一种流行的编程语言"
        success = knowledge_base.update_document(doc["id"], {"content": updated_content})

        assert success is True
        retrieved = knowledge_base.get_document(doc["id"])
        assert retrieved["content"] == updated_content

    def test_delete_document(self, knowledge_base, sample_documents):
        """测试删除文档"""
        doc = sample_documents[0]
        knowledge_base.add_document(doc)

        success = knowledge_base.delete_document(doc["id"])

        assert success is True
        assert len(knowledge_base.documents) == 0
        assert knowledge_base.get_document(doc["id"]) is None

    def test_search_documents(self, knowledge_base, sample_documents):
        """测试文档搜索"""
        for doc in sample_documents:
            knowledge_base.add_document(doc)

        results = knowledge_base.search("Python")

        assert len(results) >= 1
        for result in results:
            assert "Python" in result["content"]

    def test_filter_by_metadata(self, knowledge_base, sample_documents):
        """测试元数据过滤"""
        for doc in sample_documents:
            knowledge_base.add_document(doc)

        ai_docs = knowledge_base.filter_by_metadata({"category": "AI"})

        assert len(ai_docs) == 2
        for doc in ai_docs:
            assert doc["metadata"]["category"] == "AI"


if __name__ == "__main__":
    pytest.main([__file__])

