"""
RAG系统性能测试
测试检索增强生成系统的性能指标
"""

import pytest
import time
import asyncio
import psutil
import os
from unittest.mock import Mock, AsyncMock
from src.rag.retriever import Retriever
from src.rag.embeddings import Embeddings
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import KnowledgeBase
from src.core.types import RetrievalConfig


class TestRetrieverPerformance:
    """检索器性能测试类"""

    @pytest.fixture
    def mock_embeddings(self):
        """模拟嵌入服务"""
        embeddings = Mock()
        embeddings.encode = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        return embeddings

    @pytest.fixture
    def large_vector_store(self):
        """大型向量存储fixture"""
        store = VectorStore(dimensions=512)

        # 添加大量向量用于性能测试
        for i in range(10000):
            vector = [i * 0.0001 + j * 0.001 for j in range(512)]
            store.add(f"doc_{i}", vector)

        return store

    @pytest.fixture
    def retriever(self, mock_embeddings, large_vector_store):
        """检索器fixture"""
        config = RetrievalConfig(top_k=10, threshold=0.5)
        return Retriever(embeddings=mock_embeddings, config=config)

    def test_embedding_batch_performance(self, mock_embeddings):
        """测试嵌入批量编码性能"""
        # 生成大量文本
        texts = [f"测试文本{i}" for i in range(1000)]

        start_time = time.time()
        embeddings_batch = [mock_embeddings.encode(text) for text in texts]
        end_time = time.time()

        batch_time = end_time - start_time
        texts_per_second = len(texts) / batch_time

        # 1000个文本编码应该在2秒内完成
        assert batch_time < 2.0
        assert texts_per_second > 500

        # 验证所有嵌入都有相同维度
        dimensions = set(len(emb) for emb in embeddings_batch)
        assert len(dimensions) == 1

    def test_vector_search_performance(self, large_vector_store):
        """测试向量搜索性能"""
        query_vector = [0.05] * 512
        num_searches = 100

        start_time = time.time()

        for _ in range(num_searches):
            results = large_vector_store.search(query_vector, top_k=10)
            assert len(results) <= 10

        end_time = time.time()

        total_time = end_time - start_time
        searches_per_second = num_searches / total_time

        # 100次搜索应该在5秒内完成
        assert total_time < 5.0
        assert searches_per_second > 20

    def test_vector_storage_performance(self, large_vector_store):
        """测试向量存储性能"""
        # 测试插入性能
        start_time = time.time()

        for i in range(1000):
            vector = [i * 0.0001 + j * 0.001 for j in range(512)]
            large_vector_store.add(f"new_doc_{i}", vector)

        end_time = time.time()
        insert_time = end_time - start_time
        inserts_per_second = 1000 / insert_time

        # 1000个向量插入应该在1秒内完成
        assert insert_time < 1.0
        assert inserts_per_second > 1000

    def test_concurrent_retrieval_performance(self, retriever):
        """测试并发检索性能"""
        async def perform_retrieval(query_id):
            query_vector = [query_id * 0.01 + i * 0.001 for i in range(512)]
            return await retriever.retrieve(f"查询{query_id}", None)

        # 并发执行100次检索
        start_time = time.time()

        tasks = [perform_retrieval(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        concurrent_time = end_time - start_time

        # 100次并发检索应该在10秒内完成
        assert concurrent_time < 10.0
        assert len(results) == 100

        # 验证所有结果都有效
        for result in results:
            assert isinstance(result, list)

    def test_memory_usage_during_retrieval(self, large_vector_store):
        """测试检索过程中的内存使用"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 执行大量搜索操作
        for i in range(500):
            query_vector = [i * 0.002 + j * 0.001 for j in range(512)]
            large_vector_store.search(query_vector, top_k=50)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（小于100MB）
        assert memory_increase < 100 * 1024 * 1024


class TestEmbeddingsPerformance:
    """嵌入性能测试类"""

    @pytest.fixture
    def embeddings(self):
        """嵌入服务fixture"""
        return Embeddings()

    def test_text_length_performance(self, embeddings):
        """测试不同文本长度的性能"""
        texts = [
            "短文本",
            "中等长度的测试文本，包含更多内容和信息",
            "这是一个很长的测试文本，用于测试嵌入服务在处理长文本时的性能表现，" * 10,
            "超长文本" * 100
        ]

        for text in texts:
            start_time = time.time()
            embedding = embeddings.encode(text)
            end_time = time.time()

            processing_time = end_time - start_time
            char_per_second = len(text) / processing_time

            # 验证性能合理
            assert processing_time < 5.0  # 每个文本在5秒内处理
            assert char_per_second > 100  # 每秒至少处理100个字符
            assert embedding is not None
            assert len(embedding) > 0

    def test_batch_size_optimization(self, embeddings):
        """测试批量大小优化"""
        total_texts = 1000
        texts = [f"测试文本{i}" for i in range(total_texts)]

        # 测试不同批量大小
        batch_sizes = [1, 10, 50, 100, 500]

        best_batch_size = None
        best_time = float('inf')

        for batch_size in batch_sizes:
            start_time = time.time()

            for i in range(0, total_texts, batch_size):
                batch = texts[i:i + batch_size]
                embeddings.encode(batch)

            end_time = time.time()
            total_time = end_time - start_time

            if total_time < best_time:
                best_time = total_time
                best_batch_size = batch_size

        # 验证批量大小对性能的影响
        assert best_batch_size is not None
        assert best_time > 0

    def test_embedding_caching_performance(self, embeddings):
        """测试嵌入缓存性能"""
        texts = ["缓存测试文本", "重复的测试文本", "缓存测试文本", "另一个文本"]

        # 第一次编码（无缓存）
        start_time = time.time()
        for text in texts:
            embeddings.encode(text, use_cache=False)
        first_time = time.time() - start_time

        # 第二次编码（使用缓存）
        start_time = time.time()
        for text in texts:
            embeddings.encode(text, use_cache=True)
        second_time = time.time() - start_time

        # 缓存应该显著提升性能
        assert second_time < first_time * 0.5

        # 缓存命中率应该很高
        cache_stats = embeddings.get_cache_stats()
        assert cache_stats['hit_rate'] > 0.5


class TestVectorStorePerformance:
    """向量存储性能测试类"""

    @pytest.fixture
    def vector_store(self):
        """向量存储fixture"""
        return VectorStore(dimensions=256)

    def test_index_building_performance(self, vector_store):
        """测试索引构建性能"""
        # 添加大量向量
        num_vectors = 50000
        dimensions = 256

        start_time = time.time()

        for i in range(num_vectors):
            vector = [i * 0.0001 + j * 0.001 for j in range(dimensions)]
            vector_store.add(f"doc_{i}", vector)

        end_time = time.time()
        build_time = end_time - start_time

        # 5万个向量构建索引应该在30秒内完成
        assert build_time < 30.0
        vectors_per_second = num_vectors / build_time
        assert vectors_per_second > 1500

    def test_disk_vs_memory_performance(self, vector_store):
        """测试磁盘与内存存储性能"""
        # 内存存储测试
        vector_store.set_storage_mode("memory")

        start_time = time.time()
        for i in range(1000):
            vector = [i * 0.001 + j * 0.001 for j in range(256)]
            vector_store.add(f"mem_doc_{i}", vector)
        memory_time = time.time() - start_time

        # 磁盘存储测试
        vector_store.set_storage_mode("disk")

        start_time = time.time()
        for i in range(1000):
            vector = [i * 0.001 + j * 0.001 for j in range(256)]
            vector_store.add(f"disk_doc_{i}", vector)
        disk_time = time.time() - start_time

        # 内存存储应该比磁盘存储更快
        assert memory_time < disk_time

        # 但性能差异应该在合理范围内
        performance_ratio = disk_time / memory_time
        assert 1.0 < performance_ratio < 10.0

    def test_approximate_search_performance(self, vector_store):
        """测试近似搜索性能"""
        # 构建大型索引
        for i in range(20000):
            vector = [i * 0.0001 + j * 0.001 for j in range(256)]
            vector_store.add(f"doc_{i}", vector)

        query_vector = [0.05] * 256

        # 精确搜索
        start_time = time.time()
        exact_results = vector_store.search(query_vector, exact=True, top_k=10)
        exact_time = time.time() - start_time

        # 近似搜索
        start_time = time.time()
        approx_results = vector_store.search(query_vector, exact=False, top_k=10)
        approx_time = time.time() - start_time

        # 近似搜索应该比精确搜索更快
        assert approx_time < exact_time

        # 但结果质量应该仍然合理
        assert len(approx_results) == 10
        assert len(exact_results) == 10

        # 计算性能提升
        speedup_ratio = exact_time / approx_time
        assert speedup_ratio > 1.0


class TestRAGSystemPerformance:
    """RAG系统整体性能测试类"""

    @pytest.fixture
    def rag_system(self):
        """RAG系统fixture"""
        return MockRAGSystem()

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, rag_system):
        """测试端到端延迟"""
        queries = [f"测试查询{i}" for i in range(50)]

        latencies = []

        for query in queries:
            start_time = time.time()
            await rag_system.retrieve_and_generate(query)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

        # 计算延迟统计
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        # 延迟指标应该满足要求
        assert avg_latency < 3.0  # 平均延迟小于3秒
        assert max_latency < 10.0  # 最大延迟小于10秒
        assert p95_latency < 5.0  # P95延迟小于5秒

    @pytest.mark.asyncio
    async def test_throughput_under_load(self, rag_system):
        """测试负载下的吞吐量"""
        num_queries = 100
        queries = [f"负载测试查询{i}" for i in range(num_queries)]

        start_time = time.time()

        # 并发执行查询
        tasks = [rag_system.retrieve_and_generate(query) for query in queries]
        await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_queries / total_time

        # 吞吐量应该满足要求
        assert throughput > 10  # 每秒至少处理10个查询

    @pytest.mark.asyncio
    async def test_resource_usage_under_load(self, rag_system):
        """测试负载下的资源使用"""
        process = psutil.Process(os.getpid())
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss

        # 执行大量查询
        queries = [f"资源测试查询{i}" for i in range(200)]

        tasks = [rag_system.retrieve_and_generate(query) for query in queries]
        await asyncio.gather(*tasks)

        # 等待资源使用稳定
        await asyncio.sleep(1)

        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss

        # 资源使用应该在合理范围内
        assert final_cpu < 80.0  # CPU使用率低于80%

        memory_increase = final_memory - initial_memory
        assert memory_increase < 200 * 1024 * 1024  # 内存增长小于200MB


class MockRAGSystem:
    """模拟RAG系统用于性能测试"""

    async def retrieve_and_generate(self, query):
        """模拟检索和生成"""
        # 模拟检索延迟
        await asyncio.sleep(0.01)

        # 模拟生成延迟
        await asyncio.sleep(0.05)

        return {
            "query": query,
            "retrieved_docs": [
                {"doc_id": f"doc_{i}", "score": 0.9 - i * 0.1}
                for i in range(3)
            ]
        }


if __name__ == "__main__":
    pytest.main([__file__])
