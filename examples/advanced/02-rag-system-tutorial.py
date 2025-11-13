"""
RAG系统教程

详细演示RAG（检索增强生成）系统的构建、配置和优化。
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.rag import RAGSystem
from src.core.types import (
    RAGConfig, RAGDocument, RAGQueryResult,
    VectorStoreConfig, EmbeddingConfig
)
from src.utils.logger import get_logger


# ============================================================================
# 1. RAG系统基础教程
# ============================================================================

class RAGSystemTutorial:
    """RAG系统教程类"""

    def __init__(self):
        self.logger = get_logger("rag_tutorial")
        self.rag_system = None

    async def basic_rag_setup(self):
        """基础RAG系统设置"""
        print("=== 基础RAG系统设置 ===\n")

        # 1. 配置RAG系统
        rag_config = RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_type="chroma",  # 使用Chroma向量存储
            collection_name="tutorial_collection",
            similarity_threshold=0.7,
            retrieval_limit=5,
            chunk_size=512,
            chunk_overlap=50
        )

        # 2. 创建RAG系统
        self.rag_system = RAGSystem(rag_config)
        await self.rag_system.initialize()

        print(f"✓ RAG系统初始化完成")
        print(f"  - 嵌入模型: {rag_config.embedding_model}")
        print(f"  - 向量存储: {rag_config.vector_store_type}")
        print(f"  - 相似度阈值: {rag_config.similarity_threshold}")
        print(f"  - 检索限制: {rag_config.retrieval_limit}")
        print()

    async def add_sample_documents(self):
        """添加示例文档"""
        print("=== 添加示例文档 ===\n")

        # 示例文档集合
        documents = [
            {
                "title": "人工智能概述",
                "content": """
                人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
                致力于创建能够执行通常需要人类智能的任务的系统。
                AI包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。
                现代AI技术已经广泛应用于医疗、金融、交通、教育等领域。
                """,
                "source": "ai_textbook.pdf",
                "document_type": "textbook",
                "tags": ["AI", "概述", "基础"],
                "metadata": {"chapter": 1, "difficulty": "beginner"}
            },
            {
                "title": "机器学习基础",
                "content": """
                机器学习是AI的核心技术之一，它使计算机能够从数据中学习，
                而不需要明确编程。主要类型包括监督学习、无监督学习和强化学习。
                监督学习使用标记数据训练模型，无监督学习发现数据中的模式，
                强化学习通过试错学习最优策略。
                """,
                "source": "ml_guide.pdf",
                "document_type": "tutorial",
                "tags": ["机器学习", "基础", "算法"],
                "metadata": {"chapter": 2, "difficulty": "intermediate"}
            },
            {
                "title": "深度学习与神经网络",
                "content": """
                深度学习是机器学习的子领域，使用多层神经网络来学习数据的复杂模式。
                常见的深度学习架构包括卷积神经网络（CNN）用于图像处理，
                循环神经网络（RNN）用于序列数据，Transformer用于自然语言处理。
                深度学习在图像识别、语音识别和机器翻译等领域取得了突破性进展。
                """,
                "source": "dl_book.pdf",
                "document_type": "textbook",
                "tags": ["深度学习", "神经网络", "CNN", "RNN"],
                "metadata": {"chapter": 3, "difficulty": "advanced"}
            },
            {
                "title": "自然语言处理技术",
                "content": """
                自然语言处理（NLP）是AI的重要应用领域，专注于使计算机理解、
                生成和处理人类语言。现代NLP技术基于Transformer架构，
                包括BERT、GPT等预训练模型。NLP应用包括机器翻译、
                文本摘要、情感分析、问答系统等。
                """,
                "source": "nlp_paper.pdf",
                "document_type": "research",
                "tags": ["NLP", "Transformer", "BERT", "GPT"],
                "metadata": {"chapter": 4, "difficulty": "advanced"}
            },
            {
                "title": "计算机视觉应用",
                "content": """
                计算机视觉使计算机能够从图像和视频中获取有意义的信息。
                主要技术包括图像分类、目标检测、语义分割、图像生成等。
                深度学习特别是CNN的兴起，极大地推动了计算机视觉的发展。
                应用领域包括自动驾驶、医疗影像分析、人脸识别等。
                """,
                "source": "cv_tutorial.pdf",
                "document_type": "tutorial",
                "tags": ["计算机视觉", "CNN", "图像处理"],
                "metadata": {"chapter": 5, "difficulty": "intermediate"}
            }
        ]

        # 添加文档到RAG系统
        for i, doc_data in enumerate(documents, 1):
            document = RAGDocument(
                id=f"doc_{i}",
                title=doc_data["title"],
                content=doc_data["content"],
                source=doc_data["source"],
                document_type=doc_data["document_type"],
                tags=doc_data["tags"],
                metadata=doc_data["metadata"],
                timestamp=time.time()
            )

            await self.rag_system.add_document(document)
            print(f"✓ 添加文档: {document.title}")

        print(f"\n总共添加了 {len(documents)} 个文档")

        # 显示统计信息
        stats = await self.rag_system.get_stats()
        print(f"文档总数: {stats['total_documents']}")
        print(f"总块数: {stats['total_chunks']}")
        print()

    async def basic_retrieval_examples(self):
        """基础检索示例"""
        print("=== 基础检索示例 ===\n")

        queries = [
            "什么是人工智能？",
            "机器学习有哪些类型？",
            "深度学习使用什么架构？",
            "NLP有哪些应用？",
            "计算机视觉的主要技术"
        ]

        for query in queries:
            print(f"查询: {query}")

            # 执行检索
            results = await self.rag_system.retrieve(
                query=query,
                limit=3
            )

            print(f"找到 {len(results)} 个相关文档:")

            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']}")
                print(f"     相关度: {result['score']:.3f}")
                print(f"     内容摘要: {result['content'][:100]}...")
                print(f"     标签: {', '.join(result['tags'])}")
                print()

            print("-" * 60)
            print()


# ============================================================================
# 2. 高级RAG配置
# ============================================================================

class AdvancedRAGConfiguration:
    """高级RAG配置示例"""

    def __init__(self):
        self.logger = get_logger("advanced_rag")

    def multi_vector_strategy_config(self):
        """多向量策略配置"""
        print("=== 多向量策略配置 ===\n")

        config = RAGConfig(
            # 多向量存储
            vector_store_type="chroma",
            multi_vector=True,
            vector_strategies=["dense", "sparse", "colbert"],

            # 不同策略的权重
            strategy_weights={
                "dense": 0.5,
                "sparse": 0.3,
                "colbert": 0.2
            },

            # 重排序配置
            reranker_enabled=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_k=10,

            # 混合检索
            hybrid_search=True,
            hybrid_alpha=0.5,  # 密集和稀疏检索的权重

            other_params={
                "max_context_length": 4096,
                "context_window_size": 3
            }
        )

        return config

    def hierarchical_rag_config(self):
        """分层RAG配置"""
        print("=== 分层RAG配置 ===\n")

        config = RAGConfig(
            # 分层检索
            hierarchical_search=True,
            hierarchy_levels=["summary", "detailed", "raw"],

            # 各层配置
            level_configs={
                "summary": {
                    "chunk_size": 1024,
                    "overlap": 100,
                    "retrieval_limit": 3
                },
                "detailed": {
                    "chunk_size": 512,
                    "overlap": 50,
                    "retrieval_limit": 5
                },
                "raw": {
                    "chunk_size": 256,
                    "overlap": 25,
                    "retrieval_limit": 8
                }
            },

            # 层级选择策略
            hierarchy_selection="adaptive",  # adaptive, fixed, query_based

            other_params={
                "max_summary_depth": 2,
                "context_expansion": True
            }
        )

        return config

    def graph_rag_config(self):
        """知识图谱RAG配置"""
        print("=== 知识图谱RAG配置 ===\n")

        config = RAGConfig(
            # 知识图谱配置
            use_knowledge_graph=True,
            graph_store_type="neo4j",
            graph_config={
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            },

            # 实体抽取
            entity_extraction=True,
            entity_model="ner-model",
            relation_extraction=True,

            # 图检索策略
            graph_retrieval_strategy="neighborhood",
            max_hops=2,
            min_relevance=0.6,

            other_params={
                "graph_reranking": True,
                "path_weighting": "pagerank"
            }
        )

        return config

    def adaptive_rag_config(self):
        """自适应RAG配置"""
        print("=== 自适应RAG配置 ===\n")

        config = RAGConfig(
            # 自适应配置
            adaptive_retrieval=True,
            adaptation_strategy="query_complexity",

            # 复杂度评估
            complexity_model="query-complexity-classifier",
            complexity_thresholds={
                "simple": 0.3,
                "medium": 0.7,
                "complex": 1.0
            },

            # 不同复杂度的配置
            adaptive_configs={
                "simple": {
                    "retrieval_limit": 3,
                    "similarity_threshold": 0.8,
                    "chunk_size": 512
                },
                "medium": {
                    "retrieval_limit": 5,
                    "similarity_threshold": 0.7,
                    "chunk_size": 256
                },
                "complex": {
                    "retrieval_limit": 8,
                    "similarity_threshold": 0.6,
                    "chunk_size": 128
                }
            },

            other_params={
                "feedback_learning": True,
                "performance_tracking": True
            }
        )

        return config


# ============================================================================
# 3. RAG性能优化
# ============================================================================

class RAGPerformanceOptimization:
    """RAG性能优化"""

    def __init__(self):
        self.logger = get_logger("rag_optimization")

    async def indexing_optimization(self):
        """索引优化"""
        print("=== 索引优化 ===\n")

        optimization_strategies = {
            "batch_indexing": {
                "description": "批量索引处理",
                "batch_size": 100,
                "parallel_workers": 4
            },
            "incremental_indexing": {
                "description": "增量索引更新",
                "update_threshold": 50,  # 文档数量阈值
                "background_update": True
            },
            "smart_chunking": {
                "description": "智能分块策略",
                "semantic_chunking": True,
                "adaptive_chunk_size": True,
                "overlap_optimization": True
            },
            "vector_compression": {
                "description": "向量压缩",
                "compression_method": "product_quantization",
                "compression_ratio": 0.25
            }
        }

        for strategy, config in optimization_strategies.items():
            print(f"策略: {strategy}")
            print(f"描述: {config['description']}")
            for key, value in config.items():
                if key != "description":
                    print(f"  {key}: {value}")
            print()

    async def retrieval_optimization(self):
        """检索优化"""
        print("=== 检索优化 ===\n")

        optimization_techniques = {
            "query_preprocessing": {
                "techniques": ["expansion", "reformulation", "decomposition"],
                "llm_expansion": True,
                "synonym_enrichment": True
            },
            "cache_optimization": {
                "query_cache": True,
                "result_cache": True,
                "cache_ttl": 3600,
                "cache_size": 10000
            },
            "parallel_retrieval": {
                "enabled": True,
                "max_concurrent": 10,
                "timeout_per_query": 5.0
            },
            "result_reranking": {
                "enabled": True,
                "reranker_type": "cross_encoder",
                "top_k_rerank": 20,
                "final_top_k": 5
            }
        }

        for technique, config in optimization_techniques.items():
            print(f"技术: {technique}")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()

    async def memory_optimization(self):
        """内存优化"""
        print("=== 内存优化 ===\n")

        memory_strategies = {
            "vector_memory_efficiency": {
                "precision": "float16",  # 减少内存使用
                "index_type": "hnsw",    # 高效索引
                "memory_map": True       # 内存映射
            },
            "document_storage": {
                "compressed_storage": True,
                "lazy_loading": True,
                "document_caching": "lru"
            },
            "embedding_caching": {
                "enabled": True,
                "cache_strategy": "lfu",  # 最少使用频率
                "max_cache_size": 1000000
            },
            "garbage_collection": {
                "enabled": True,
                "collection_interval": 300,  # 5分钟
                "memory_threshold": 0.8       # 80%内存使用率
            }
        }

        for strategy, config in memory_strategies.items():
            print(f"策略: {strategy}")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()


# ============================================================================
# 4. RAG评估和监控
# ============================================================================

class RAGEvaluationMonitoring:
    """RAG评估和监控"""

    def __init__(self):
        self.logger = get_logger("rag_evaluation")

    async def quality_evaluation(self):
        """质量评估"""
        print("=== RAG质量评估 ===\n")

        # 评估指标
        quality_metrics = {
            "retrieval_metrics": {
                "precision": "检索结果的相关性精度",
                "recall": "检索结果的召回率",
                "f1_score": "F1分数",
                "mrr": "平均倒数排名"
            },
            "generation_metrics": {
                "relevance": "生成内容的相关性",
                "faithfulness": "生成内容的忠实度",
                "coherence": "生成内容的连贯性",
                "completeness": "生成内容的完整性"
            },
            "end_to_end_metrics": {
                "answer_quality": "答案整体质量",
                "user_satisfaction": "用户满意度",
                "task_success_rate": "任务成功率",
                "response_time": "响应时间"
            }
        }

        for category, metrics in quality_metrics.items():
            print(f"评估类别: {category}")
            for metric, description in metrics.items():
                print(f"  {metric}: {description}")
            print()

    async def performance_monitoring(self):
        """性能监控"""
        print("=== RAG性能监控 ===\n")

        monitoring_metrics = {
            "system_metrics": {
                "index_size": "索引大小（文档数量）",
                "memory_usage": "内存使用量",
                "cpu_usage": "CPU使用率",
                "disk_io": "磁盘I/O"
            },
            "retrieval_metrics": {
                "query_latency": "查询延迟",
                "throughput": "查询吞吐量",
                "cache_hit_rate": "缓存命中率",
                "index_update_time": "索引更新时间"
            },
            "quality_metrics": {
                "average_similarity": "平均相似度",
                "retrieval_success_rate": "检索成功率",
                "duplicate_rate": "重复结果率",
                "outlier_rate": "异常结果率"
            }
        }

        for category, metrics in monitoring_metrics.items():
            print(f"监控类别: {category}")
            for metric, description in metrics.items():
                print(f"  {metric}: {description}")
            print()

    def evaluation_dataset(self):
        """评估数据集示例"""
        print("=== 评估数据集示例 ===\n")

        evaluation_data = [
            {
                "query": "什么是机器学习？",
                "expected_documents": ["机器学习基础", "AI概述"],
                "expected_answer": "机器学习是使计算机从数据中学习的技术",
                "difficulty": "easy"
            },
            {
                "query": "深度学习如何应用于图像识别？",
                "expected_documents": ["深度学习与神经网络", "计算机视觉应用"],
                "expected_answer": "深度学习使用CNN等架构进行图像识别",
                "difficulty": "medium"
            },
            {
                "query": "比较监督学习和无监督学习的优缺点",
                "expected_documents": ["机器学习基础", "高级机器学习算法"],
                "expected_answer": "监督学习需要标记数据但效果更好，无监督学习无需标记但可能效果较差",
                "difficulty": "hard"
            }
        ]

        for i, data in enumerate(evaluation_data, 1):
            print(f"评估样本 {i}:")
            print(f"  查询: {data['query']}")
            print(f"  期望文档: {', '.join(data['expected_documents'])}")
            print(f"  期望答案: {data['expected_answer']}")
            print(f"  难度: {data['difficulty']}")
            print()


# ============================================================================
# 5. 实际应用案例
# ============================================================================

class RAGRealWorldApplications:
    """RAG实际应用案例"""

    def __init__(self):
        self.logger = get_logger("rag_applications")

    def customer_support_rag(self):
        """客户支持RAG系统"""
        print("=== 客户支持RAG系统 ===\n")

        system_design = {
            "data_sources": [
                "产品文档",
                "FAQ数据库",
                "工单记录",
                "聊天记录",
                "知识库文章"
            ],
            "key_features": [
                "多语言支持",
                "实时更新",
                "个性化推荐",
                "工单自动分类",
                "满意度跟踪"
            ],
            "technical_stack": {
                "embedding": "multilingual-MiniLM-L12-v2",
                "vector_store": "Pinecone",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "llm": "GPT-4"
            },
            "performance_targets": {
                "response_time": "< 2秒",
                "accuracy": "> 85%",
                "satisfaction": "> 4.0/5.0"
            }
        }

        for category, items in system_design.items():
            print(f"{category}:")
            for item in items if isinstance(items, list) else items.items():
                if isinstance(items, list):
                    print(f"  - {item}")
                else:
                    print(f"  {item[0]}: {item[1]}")
            print()

    def research_assistant_rag(self):
        """研究助理RAG系统"""
        print("=== 研究助理RAG系统 ===\n")

        system_design = {
            "data_sources": [
                "学术论文数据库",
                "专利文档",
                "技术报告",
                "会议论文",
                "书籍章节"
            ],
            "specialized_features": [
                "引用网络分析",
                "领域专家推荐",
                "研究趋势分析",
                "文献综述生成",
                "实验建议"
            ],
            "advanced_capabilities": {
                "multi_modal_retrieval": True,  # 文本+图像
                "temporal_analysis": True,      # 时间演化分析
                "citation_ranking": True,       # 引用排名
                "expertise_matching": True      # 专业匹配
            }
        }

        for category, items in system_design.items():
            print(f"{category}:")
            for item in items if isinstance(items, list) else items.items():
                if isinstance(items, list):
                    print(f"  - {item}")
                else:
                    print(f"  {item[0]}: {item[1]}")
            print()

    def legal_document_rag(self):
        """法律文档RAG系统"""
        print("=== 法律文档RAG系统 ===\n")

        system_design = {
            "data_sources": [
                "法律法规库",
                "判例数据库",
                "法律评论",
                "合同模板",
                "法律问答"
            ],
            "legal_features": [
                "法条引用验证",
                "案例相似度分析",
                "风险评估",
                "合规检查",
                "法律建议生成"
            ],
            "accuracy_requirements": {
                "citation_accuracy": "100%",
                "legal_interpretation": "专家验证",
                "compliance_checking": "多级审核",
                "liability_assessment": "保守原则"
            }
        }

        for category, items in system_design.items():
            print(f"{category}:")
            for item in items if isinstance(items, list) else items.items():
                if isinstance(items, list):
                    print(f"  - {item}")
                else:
                    print(f"  {item[0]}: {item[1]}")
            print()


# ============================================================================
# 6. 主程序和完整示例
# ============================================================================

async def complete_rag_tutorial():
    """完整的RAG教程演示"""
    print("=== RAG系统完整教程 ===\n")

    # 1. 基础教程
    tutorial = RAGSystemTutorial()
    await tutorial.basic_rag_setup()
    await tutorial.add_sample_documents()
    await tutorial.basic_retrieval_examples()

    # 2. 高级配置
    print("=== 高级RAG配置 ===\n")
    advanced_config = AdvancedRAGConfiguration()

    multi_vector_config = advanced_config.multi_vector_strategy_config()
    hierarchical_config = advanced_config.hierarchical_rag_config()
    graph_config = advanced_config.graph_rag_config()
    adaptive_config = advanced_config.adaptive_rag_config()

    # 3. 性能优化
    optimization = RAGPerformanceOptimization()
    await optimization.indexing_optimization()
    await optimization.retrieval_optimization()
    await optimization.memory_optimization()

    # 4. 评估监控
    evaluation = RAGEvaluationMonitoring()
    await evaluation.quality_evaluation()
    await evaluation.performance_monitoring()
    evaluation.evaluation_dataset()

    # 5. 实际应用
    applications = RAGRealWorldApplications()
    applications.customer_support_rag()
    applications.research_assistant_rag()
    applications.legal_document_rag()

    print("=== RAG教程完成 ===")


# ============================================================================
# 7. 最佳实践指南
# ============================================================================

class RAGBestPractices:
    """RAG最佳实践指南"""

    @staticmethod
    def data_preparation_practices():
        """数据准备最佳实践"""
        return {
            "data_quality": [
                "确保数据准确性和时效性",
                "去除重复和低质量内容",
                "标准化数据格式和结构",
                "验证数据来源的可靠性"
            ],
            "document_processing": [
                "智能分块策略",
                "保持语义完整性",
                "添加元数据标签",
                "建立文档层次结构"
            ],
            "content_enrichment": [
                "添加摘要和关键词",
                "提取实体和关系",
                "生成主题标签",
                "建立交叉引用"
            ]
        }

    @staticmethod
    def system_design_practices():
        """系统设计最佳实践"""
        return {
            "architecture": [
                "模块化设计便于扩展",
                "微服务架构提高可维护性",
                "异步处理提高性能",
                "容错设计保证稳定性"
            ],
            "scalability": [
                "水平扩展支持",
                "负载均衡策略",
                "缓存机制优化",
                "数据库分区设计"
            ],
            "security": [
                "访问控制和认证",
                "数据加密存储",
                "审计日志记录",
                "隐私保护措施"
            ]
        }

    @staticmethod
    def optimization_practices():
        """优化最佳实践"""
        return {
            "performance": [
                "索引优化策略",
                "查询优化技巧",
                "缓存命中提升",
                "并行处理实现"
            ],
            "quality": [
                "相关性评估指标",
                "用户反馈收集",
                "A/B测试验证",
                "持续改进机制"
            ],
            "cost": [
                "资源使用监控",
                "存储成本优化",
                "计算资源调度",
                "按需扩展策略"
            ]
        }

    @staticmethod
    def deployment_practices():
        """部署最佳实践"""
        return {
            "environment": [
                "多环境管理策略",
                "配置外部化管理",
                "容器化部署方案",
                "自动化CI/CD流程"
            ],
            "monitoring": [
                "实时性能监控",
                "异常告警机制",
                "日志聚合分析",
                "业务指标跟踪"
            ],
            "maintenance": [
                "定期更新维护",
                "备份恢复策略",
                "版本回滚机制",
                "灾难恢复预案"
            ]
        }


async def main():
    """主程序入口"""
    print("=== RAG系统详细教程 ===\n")

    # 运行完整教程
    await complete_rag_tutorial()

    # 显示最佳实践
    print("\n=== RAG最佳实践指南 ===\n")

    practices = RAGBestPractices()

    print("数据准备最佳实践:")
    data_practices = practices.data_preparation_practices()
    for category, items in data_practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")

    print("\n系统设计最佳实践:")
    design_practices = practices.system_design_practices()
    for category, items in design_practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")

    print("\n=== 教程完成 ===")


if __name__ == "__main__":
    # 运行RAG系统教程
    asyncio.run(main())
