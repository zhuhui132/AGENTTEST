"""
RAG系统综合测试
深度测试检索增强生成系统的各种场景和边界条件
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock
from src.rag.retriever import Retriever
from src.rag.embeddings import Embeddings
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import KnowledgeBase
from src.core.types import RetrievalConfig


class TestRAGComprehensive:
    """RAG系统综合测试类"""

    @pytest.fixture
    def config(self):
        """RAG配置fixture"""
        return RetrievalConfig(
            top_k=5,
            threshold=0.7,
            include_metadata=True,
            similarity_metric="cosine",
            max_context_length=4000
        )

    @pytest.fixture
    def mock_embeddings(self):
        """模拟嵌入服务"""
        embeddings = Mock()

        def embed_batch(texts):
            return [np.random.randn(512) for _ in texts]

        embeddings.encode = Mock(side_effect=embed_batch)
        embeddings.encode_batch = Mock(side_effect=embed_batch)
        return embeddings

    @pytest.fixture
    def mock_vector_store(self):
        """模拟向量存储"""
        store = Mock()

        # 模拟向量存储
        vectors = {}
        for i in range(100):
            doc_id = f"doc_{i}"
            vector = np.random.randn(512)
            vectors[doc_id] = vector

        store.vectors = vectors
        store.get_vector = Mock(side_effect=lambda doc_id: vectors.get(doc_id))

        # 模拟搜索功能
        def search(query_vector, top_k=5, threshold=0.7):
            scores = {}
            for doc_id, vector in vectors.items():
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                scores[doc_id] = similarity

            # 按相似度排序
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [
                {"doc_id": doc_id, "score": score}
                for doc_id, score in sorted_docs[:top_k]
                if score >= threshold
            ]

        store.search = Mock(side_effect=search)
        return store

    @pytest.fixture
    def mock_knowledge_base(self):
        """模拟知识库"""
        kb = Mock()

        # 模拟文档数据
        documents = {}
        for i in range(100):
            doc_id = f"doc_{i}"
            doc_content = f"这是文档{doc_id}的内容，包含各种主题和关键词"
            doc_metadata = {
                "source": f"source_{i%10}",
                "category": f"category_{i%5}",
                "author": f"author_{i%20}",
                "created_date": f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}",
                "tags": [f"tag_{j}" for j in range(3)]
            }
            documents[doc_id] = {
                "id": doc_id,
                "content": doc_content,
                "metadata": doc_metadata
            }

        kb.documents = documents
        kb.get_document = Mock(side_effect=lambda doc_id: documents.get(doc_id))
        kb.get_documents = Mock(side_effect=lambda doc_ids: [
            documents[doc_id] for doc_id in doc_ids if doc_id in documents
        ])

        return kb

    @pytest.fixture
    def rag_system(self, config, mock_embeddings, mock_vector_store, mock_knowledge_base):
        """RAG系统fixture"""
        return Retriever(
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
            knowledge_base=mock_knowledge_base,
            config=config
        )

    @pytest.mark.asyncio
    async def test_hierarchical_retrieval(self, rag_system):
        """测试分层检索"""
        # 构建分层数据
        queries = [
            "查询核心概念",
            "查询详细技术",
            "查询基础知识",
            "查询高级应用"
        ]

        # 测试分层检索策略
        for query in queries:
            # 设置分层检索
            rag_system.enable_hierarchical_retrieval(
                layers=["core", "detailed", "basic", "advanced"]
            )

            results = await rag_system.hierarchical_retrieve(query)

            # 验证分层结果
            assert len(results) > 0
            assert 'layer_scores' in results
            assert 'final_results' in results

            # 验证分层权重
            total_weight = sum(results['layer_scores'].values())
            assert abs(total_weight - 1.0) < 0.01  # 权重总和应该接近1

    @pytest.mark.asyncio
    async def test_adaptive_retrieval(self, rag_system):
        """测试自适应检索"""
        # 测试不同复杂度的查询
        adaptive_queries = [
            ("简单查询", "简单"),
            ("中等复杂度的查询内容", "中等"),
            ("这是一个非常复杂的查询，包含多个关键词和复杂的概念关系", "复杂"),
            ("超复杂的查询，涉及深度技术细节和专业知识", "超复杂")
        ]

        for query, complexity in adaptive_queries:
            # 设置自适应检索
            rag_system.enable_adaptive_retrieval()

            results = await rag_system.retrieve_with_adaptation(query)

            # 验证自适应结果
            assert 'complexity_score' in results
            assert 'adapted_config' in results
            assert 'retrieval_results' in results

            # 验证复杂度评估
            complexity_score = results['complexity_score']
            if complexity == "简单":
                assert complexity_score < 0.3
            elif complexity == "复杂":
                assert complexity_score > 0.7
            elif complexity == "超复杂":
                assert complexity_score > 0.9

            # 验证配置自适应
            adapted_config = results['adapted_config']
            if complexity == "简单":
                assert adapted_config['top_k'] <= 5
                assert adapted_config['threshold'] >= 0.7
            elif complexity == "复杂":
                assert adapted_config['top_k'] >= 10
                assert adapted_config['threshold'] <= 0.5

    @pytest.mark.asyncio
    async def test_context_aware_retrieval(self, rag_system):
        """测试上下文感知检索"""
        # 构建对话上下文
        context_history = [
            {"role": "user", "content": "我对机器学习感兴趣"},
            {"role": "assistant", "content": "机器学习是一个很好的领域"},
            {"role": "user", "content": "特别是深度学习"},
            {"role": "assistant", "content": "深度学习使用神经网络"}
        ]

        # 上下文感知查询
        contextual_query = "请详细介绍"

        # 设置上下文感知检索
        rag_system.set_context_history(context_history)

        results = await rag_system.context_aware_retrieve(contextual_query)

        # 验证上下文感知结果
        assert len(results) > 0
        assert 'context_scores' in results
        assert 'context_boosted_results' in results

        # 验证上下文相关性
        for result in results['context_boosted_results']:
            assert 'original_score' in result
            assert 'context_boost' in result
            assert result['context_boost'] >= 1.0  # 上下文增强应该提升分数

        # 检查是否识别出深度学习相关内容
        dl_related = any(
            "深度学习" in result.get('document', {}).get('content', '')
            or "神经网络" in result.get('document', {}).get('content', '')
            for result in results['context_boosted_results']
        )
        assert dl_related

    @pytest.mark.asyncio
    async def test_multi_modal_retrieval(self, rag_system):
        """测试多模态检索"""
        # 构建多模态查询
        multimodal_queries = [
            {
                "text": "图片中的动物",
                "image_embedding": np.random.randn(512),  # 模拟图像嵌入
                "modality": "text+image"
            },
            {
                "text": "音频内容的描述",
                "audio_embedding": np.random.randn(256),  # 模拟音频嵌入
                "modality": "text+audio"
            },
            {
                "text": "视频分析",
                "video_embedding": np.random.randn(1024),  # 模拟视频嵌入
                "modality": "text+video"
            }
        ]

        for query in multimodal_queries:
            results = await rag_system.multimodal_retrieve(query)

            # 验证多模态结果
            assert len(results) > 0
            assert 'modality_scores' in results
            assert 'cross_modal_results' in results

            # 验证跨模态匹配
            for result in results['cross_modal_results']:
                assert 'modality_type' in result
                assert 'cross_similarity' in result
                assert result['cross_similarity'] >= 0.0
                assert result['cross_similarity'] <= 1.0

    @pytest.mark.asyncio
    async def test_temporal_retrieval(self, rag_system):
        """测试时间感知检索"""
        # 构建时间相关的查询
        temporal_queries = [
            "最近的发现",
            "去年的趋势",
            "未来的预测",
            "历史数据"
        ]

        # 设置时间感知检索
        rag_system.enable_temporal_retrieval(
            time_weight=0.7,
            decay_factor=0.9
        )

        for query in temporal_queries:
            # 模拟当前时间
            current_time = "2023-11-01"
            results = await rag_system.temporal_retrieve(query, current_time)

            # 验证时间感知结果
            assert len(results) > 0
            assert 'temporal_scores' in results
            assert 'time_decayed_results' in results

            # 验证时间权重应用
            for result in results['time_decayed_results']:
                assert 'base_score' in result
                assert 'temporal_weight' in result
                assert 'final_score' in result

                # 时间权重应该基于时间差计算
                final_score = result['base_score'] * result['temporal_weight']
                assert abs(result['final_score'] - final_score) < 0.001

    @pytest.mark.asyncio
    async def test_domain_specific_retrieval(self, rag_system):
        """测试领域特定检索"""
        # 定义不同领域的查询
        domain_queries = [
            ("医学研究", "medical"),
            ("法律条文", "legal"),
            ("金融分析", "financial"),
            ("技术文档", "technical"),
            ("科学论文", "academic")
        ]

        for query, domain in domain_queries:
            # 设置领域特定检索
            rag_system.enable_domain_specific_retrieval(domain)

            results = await rag_system.domain_retrieve(query)

            # 验证领域特定结果
            assert len(results) > 0
            assert 'domain_relevance' in results
            assert results['domain_relevance'] >= 0.0
            assert results['domain_relevance'] <= 1.0

            # 验证领域适配
            domain_config = rag_system.get_domain_config(domain)
            assert domain_config is not None
            assert domain_config['domain'] == domain

    @pytest.mark.asyncio
    async def test_query_expansion_and_refinement(self, rag_system):
        """测试查询扩展和优化"""
        # 测试查询扩展
        expansion_queries = [
            "ML",  # 缩写扩展
            "机器学习",  # 同义词扩展
            "AI",   # 相关概念扩展
            "算法模型"  # 上下文扩展
        ]

        for original_query in expansion_queries:
            results = await rag_system.retrieve_with_expansion(original_query)

            # 验证扩展结果
            assert 'original_query' in results
            assert 'expanded_queries' in results
            assert 'expanded_results' in results

            # 验证扩展查询数量
            assert len(results['expanded_queries']) >= 1
            assert len(results['expanded_results']) >= len(results['expanded_queries'])

            # 验证扩展质量
            for expanded_result in results['expanded_results']:
                assert 'expansion_type' in expanded_result
                assert 'expansion_confidence' in expanded_result
                assert expanded_result['expansion_confidence'] >= 0.0
                assert expanded_result['expansion_confidence'] <= 1.0

    @pytest.mark.asyncio
    async def test_relevance_feedback_loop(self, rag_system):
        """测试相关性反馈循环"""
        # 初始化反馈系统
        rag_system.enable_relevance_feedback()

        # 第一轮检索
        query = "测试查询"
        results1 = await rag_system.retrieve(query)

        # 模拟用户反馈
        feedback_data = []
        for i, result in enumerate(results1[:3]):
            if i == 0:
                feedback = {"doc_id": result['doc_id'], "relevance": 5}  # 高相关
            elif i == 1:
                feedback = {"doc_id": result['doc_id'], "relevance": 3}  # 中等相关
            else:
                feedback = {"doc_id": result['doc_id'], "relevance": 1}  # 低相关
            feedback_data.append(feedback)

            # 提交反馈
            await rag_system.submit_relevance_feedback(feedback)

        # 第二轮检索（应该考虑反馈）
        results2 = await rag_system.retrieve(query)

        # 验证反馈学习效果
        assert len(results2) > 0

        # 验证反馈影响了排名
        feedback_scores = {item['doc_id']: item['relevance'] for item in feedback_data}

        for result in results2:
            doc_id = result['doc_id']
            if doc_id in feedback_scores:
                # 高反馈的文档应该排名更高
                result_position = results2.index(result)
                high_feedback_docs = [item for item in feedback_data if item['relevance'] >= 4]
                if doc_id in [item['doc_id'] for item in high_feedback_docs]:
                    assert result_position <= len(high_feedback_docs)

    @pytest.mark.asyncio
    async def test_reranking_strategies(self, rag_system):
        """测试重排序策略"""
        # 测试查询
        rerank_query = "复杂的技术查询"

        # 获取初始检索结果
        initial_results = await rag_system.basic_retrieve(rerank_query)

        # 测试不同重排序策略
        reranking_strategies = [
            "cross_encoder",
            "learning_to_rank",
            "neural_rerank",
            "semantic_similarity",
            "diversity_aware"
        ]

        for strategy in reranking_strategies:
            rag_system.set_reranking_strategy(strategy)
            reranked_results = await rag_system.rerank_results(initial_results)

            # 验证重排序结果
            assert len(reranked_results) > 0
            assert 'rerank_scores' in reranked_results[0]
            assert 'original_scores' in reranked_results[0]

            # 验证重排序改变了排序
            original_scores = [r['original_score'] for r in reranked_results]
            rerank_scores = [r['rerank_scores'] for r in reranked_results]

            # 排序应该不同（除非策略不生效）
            if strategy != "no_rerank":
                # 重排序后的分数应该用于排序
                assert all(rerank_scores[i] >= rerank_scores[i+1]
                       for i in range(len(rerank_scores)-1))

    @pytest.mark.asyncio
    async def test_cache_aware_retrieval(self, rag_system):
        """测试缓存感知检索"""
        # 启用缓存
        rag_system.enable_caching(cache_size=1000, ttl_seconds=3600)

        # 相同查询的多次检索
        query = "缓存测试查询"

        # 第一次检索
        start_time = asyncio.get_event_loop().time()
        results1 = await rag_system.retrieve(query)
        first_time = asyncio.get_event_loop().time() - start_time

        # 第二次检索（应该从缓存获取）
        start_time = asyncio.get_event_loop().time()
        results2 = await rag_system.retrieve(query)
        second_time = asyncio.get_event_loop().time() - start_time

        # 验证缓存效果
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1['doc_id'] == r2['doc_id']
            assert abs(r1['score'] - r2['score']) < 0.001

        # 缓存查询应该更快
        assert second_time < first_time * 0.5  # 至少快50%

        # 验证缓存统计
        cache_stats = rag_system.get_cache_stats()
        assert cache_stats['hit_rate'] > 0.0
        assert cache_stats['total_queries'] == 2
        assert cache_stats['cache_hits'] == 1

    @pytest.mark.asyncio
    async def test_explainable_retrieval(self, rag_system):
        """测试可解释的检索"""
        query = "需要解释的复杂查询"

        # 启用可解释检索
        rag_system.enable_explainability()

        results = await rag_system.explainable_retrieve(query)

        # 验证可解释性
        for result in results:
            assert 'explanation' in result
            assert 'retrieval_factors' in result
            assert 'confidence_score' in result
            assert 'matching_terms' in result

            # 验证解释质量
            explanation = result['explanation']
            assert len(explanation) > 0
            assert any(factor['factor'] in explanation.lower()
                   for factor in result['retrieval_factors'])

    @pytest.mark.asyncio
    async def test_robust_retrieval(self, rag_system):
        """测试鲁棒性检索"""
        # 测试各种挑战性查询
        robust_queries = [
            "",  # 空查询
            "x" * 1000,  # 超长查询
            "【特殊字符】@#$%^&*()",  # 特殊字符
            "🚀🌟✨🎯",  # 表情符号
            "中 文 混 合",  # 中英文混合
            "查询\n\n\n查询",  # 多行
            "   查询   ",  # 首尾空格
        ]

        for query in robust_queries:
            results = await rag_system.robust_retrieve(query)

            # 验证鲁棒性处理
            if not query:  # 空查询
                assert len(results) == 0
            else:
                # 非空查询应该返回结果或优雅降级
                assert isinstance(results, list)

                # 验证错误处理
                if 'error' in results[0]:
                    assert 'error_type' in results[0]
                    assert 'fallback_results' in results[0]

    @pytest.mark.asyncio
    async def test_integrated_rag_pipeline(self, rag_system):
        """测试集成的RAG流水线"""
        # 测试完整的RAG流水线
        complex_query = "需要深度检索和生成的复杂问题"

        # 执行完整的RAG流水线
        pipeline_result = await rag_system.full_rag_pipeline(complex_query)

        # 验证流水线结果
        assert 'retrieval_results' in pipeline_result
        assert 'context_construction' in pipeline_result
        assert 'generation_input' in pipeline_result
        assert 'final_response' in pipeline_result
        assert 'pipeline_metadata' in pipeline_result

        # 验证上下文构建
        context = pipeline_result['context_construction']
        assert 'total_context_length' in context
        assert 'used_documents' in context
        assert 'truncation_applied' in context

        # 验证流水线指标
        metadata = pipeline_result['pipeline_metadata']
        assert 'retrieval_time' in metadata
        assert 'context_building_time' in metadata
        assert 'generation_time' in metadata
        assert 'total_pipeline_time' in metadata

        # 验证时间合理性
        total_time = metadata['total_pipeline_time']
        retrieval_time = metadata['retrieval_time']
        context_time = metadata['context_building_time']
        generation_time = metadata['generation_time']

        assert total_time >= retrieval_time + context_time + generation_time
        assert total_time > 0  # 应该有时间消耗
        assert total_time < 10.0  # 应该在合理时间内完成

    def test_rag_system_configuration(self, rag_system):
        """测试RAG系统配置"""
        # 测试各种配置组合
        config_combinations = [
            {
                'top_k': 5,
                'threshold': 0.7,
                'similarity_metric': 'cosine'
            },
            {
                'top_k': 10,
                'threshold': 0.5,
                'similarity_metric': 'euclidean'
            },
            {
                'top_k': 3,
                'threshold': 0.9,
                'similarity_metric': 'dot_product'
            }
        ]

        for config in config_combinations:
            rag_system.update_config(config)
            current_config = rag_system.get_config()

            # 验证配置更新
            assert current_config['top_k'] == config['top_k']
            assert current_config['threshold'] == config['threshold']
            assert current_config['similarity_metric'] == config['similarity_metric']

            # 验证配置验证
            assert rag_system.validate_config() is True

    def test_rag_system_metrics(self, rag_system):
        """测试RAG系统指标"""
        # 生成一些检索活动
        metrics_data = [
            {'query': '查询1', 'results_count': 5, 'latency': 0.1},
            {'query': '查询2', 'results_count': 3, 'latency': 0.15},
            {'query': '查询3', 'results_count': 7, 'latency': 0.08}
        ]

        # 记录指标
        for data in metrics_data:
            rag_system.record_metrics(data)

        # 获取系统指标
        metrics = rag_system.get_system_metrics()

        # 验证指标完整性
        assert 'total_queries' in metrics
        assert 'average_latency' in metrics
        assert 'average_results_count' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'error_rate' in metrics
        assert 'throughput' in metrics

        # 验证指标计算
        assert metrics['total_queries'] == 3
        assert abs(metrics['average_latency'] - 0.11) < 0.01
        assert abs(metrics['average_results_count'] - 5.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
