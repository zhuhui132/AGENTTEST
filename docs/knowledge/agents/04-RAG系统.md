# 🔍 RAG集成Agent系统

## 📚 概述

检索增强生成(RAG, Retrieval-Augmented Generation)是将信息检索与生成模型结合的技术，使Agent能够基于外部知识库生成更准确、更及时的回答。本文档详细介绍RAG系统的架构设计、实现方法和最佳实践。

## 🏗️ RAG系统架构

### 核心组件设计
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import hashlib
import logging

class RAGComponent(Enum):
    """RAG组件类型"""
    RETRIEVER = "retriever"      # 检索器
    INDEXER = "indexer"          # 索引器
    RERANKER = "reranker"       # 重排序器
    GENERATOR = "generator"        # 生成器
    MEMORY = "memory"             # 记忆系统

class RetrievalMethod(Enum):
    """检索方法"""
    VECTOR_SEARCH = "vector"        # 向量检索
    KEYWORD_SEARCH = "keyword"     # 关键词检索
    HYBRID = "hybrid"              # 混合检索
    SEMANTIC = "semantic"          # 语义检索

@dataclass
class Document:
    """文档对象"""
    id: str = field(default_factory=lambda: str(time.time()))
    content: str = ""
    title: str = ""
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0
    source: str = ""
    chunk_id: Optional[str] = None

@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    documents: List[Document]
    retrieval_time: float = 0.0
    total_candidates: int = 0
    method: RetrievalMethod = RetrievalMethod.VECTOR_SEARCH
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseRAGComponent(ABC):
    """RAG组件基础类"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"rag.{name}")

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化组件"""
        pass

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """处理请求"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

class RAGSystem:
    """RAG系统主类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        self.retrieval_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_generation_time': 0.0,
            'cache_hit_rate': 0.0
        }

        # 初始化组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化RAG组件"""
        # 检索器
        if 'retriever' in self.config:
            self.components[RAGComponent.RETRIEVER] = self._create_retriever(
                self.config['retriever']
            )

        # 索引器
        if 'indexer' in self.config:
            self.components[RAGComponent.INDEXER] = self._create_indexer(
                self.config['indexer']
            )

        # 重排序器
        if 'reranker' in self.config:
            self.components[RAGComponent.RERANKER] = self._create_reranker(
                self.config['reranker']
            )

        # 生成器
        if 'generator' in self.config:
            self.components[RAGComponent.GENERATOR] = self._create_generator(
                self.config['generator']
            )

        # 记忆系统
        if 'memory' in self.config:
            self.components[RAGComponent.MEMORY] = self._create_memory(
                self.config['memory']
            )

    async def initialize(self) -> bool:
        """初始化所有组件"""
        success = True

        for component_name, component in self.components.items():
            try:
                if await component.initialize():
                    self.logger.info(f"Component {component_name} initialized successfully")
                else:
                    self.logger.error(f"Failed to initialize component {component_name}")
                    success = False
            except Exception as e:
                self.logger.error(f"Error initializing {component_name}: {e}")
                success = False

        return success

    async def query(self, query_text: str, top_k: int = 5,
                   retrieval_method: RetrievalMethod = RetrievalMethod.VECTOR_SEARCH,
                   rerank: bool = True) -> Dict[str, Any]:
        """执行RAG查询"""
        start_time = time.time()

        try:
            # 1. 检索文档
            retrieval_result = await self._retrieve_documents(
                query_text, top_k, retrieval_method
            )

            # 2. 重排序（如果启用）
            if rerank and RAGComponent.RERANKER in self.components:
                retrieval_result = await self._rerank_documents(
                    query_text, retrieval_result
                )

            # 3. 生成回答
            if RAGComponent.GENERATOR in self.components:
                generation_result = await self._generate_answer(
                    query_text, retrieval_result
                )
            else:
                generation_result = {
                    'answer': "生成器未配置",
                    'context': retrieval_result
                }

            # 4. 存储到记忆系统
            if RAGComponent.MEMORY in self.components:
                await self._store_to_memory(query_text, retrieval_result, generation_result)

            # 5. 更新统计
            total_time = time.time() - start_time
            self._update_performance_metrics(total_time)

            # 6. 记录查询历史
            self.retrieval_history.append({
                'query': query_text,
                'retrieval_result': retrieval_result,
                'generation_result': generation_result,
                'timestamp': time.time(),
                'total_time': total_time
            })

            return {
                'success': True,
                'query': query_text,
                'retrieval': retrieval_result,
                'generation': generation_result,
                'total_time': total_time
            }

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query_text
            }

    async def _retrieve_documents(self, query_text: str, top_k: int,
                               method: RetrievalMethod) -> RetrievalResult:
        """检索文档"""
        retriever = self.components.get(RAGComponent.RETRIEVER)
        if not retriever:
            raise ValueError("Retriever not configured")

        return await retriever.retrieve(query_text, top_k, method)

    async def _rerank_documents(self, query_text: str,
                               retrieval_result: RetrievalResult) -> RetrievalResult:
        """重排序文档"""
        reranker = self.components[RAGComponent.RERANKER]
        if not reranker:
            return retrieval_result

        reranked_docs = await reranker.rerank(query_text, retrieval_result.documents)

        # 创建新的检索结果
        return RetrievalResult(
            query=retrieval_result.query,
            documents=reranked_docs,
            retrieval_time=retrieval_result.retrieval_time,
            total_candidates=retrieval_result.total_candidates,
            method=retrieval_result.method,
            metadata={
                **retrieval_result.metadata,
                'reranked': True,
                'rerank_time': time.time()
            }
        )

    async def _generate_answer(self, query_text: str,
                            retrieval_result: RetrievalResult) -> Dict[str, Any]:
        """生成回答"""
        generator = self.components.get(RAGComponent.GENERATOR)
        if not generator:
            raise ValueError("Generator not configured")

        # 构建上下文
        context = self._build_context(retrieval_result.documents)

        return await generator.generate(query_text, context)

    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文"""
        if not documents:
            return "没有找到相关文档。"

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_part = f"[文档{i}] {doc.title}\n{doc.content}"
            if doc.url:
                context_part += f"\n来源: {doc.url}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _store_to_memory(self, query: str, retrieval_result: RetrievalResult,
                           generation_result: Dict[str, Any]):
        """存储到记忆系统"""
        memory = self.components.get(RAGComponent.MEMORY)
        if not memory:
            return

        memory_item = {
            'type': 'rag_query',
            'query': query,
            'retrieved_docs': [doc.id for doc in retrieval_result.documents],
            'answer': generation_result.get('answer', ''),
            'timestamp': time.time()
        }

        await memory.store(memory_item)

    def _update_performance_metrics(self, total_time: float):
        """更新性能指标"""
        self.performance_metrics['total_queries'] += 1

        # 更新平均检索时间
        current_avg = self.performance_metrics['avg_retrieval_time']
        n = self.performance_metrics['total_queries']
        self.performance_metrics['avg_retrieval_time'] = (
            (current_avg * (n - 1) + total_time) / n
        )

    async def add_document(self, document: Document) -> bool:
        """添加文档"""
        indexer = self.components.get(RAGComponent.INDEXER)
        if not indexer:
            self.logger.warning("Indexer not configured")
            return False

        return await indexer.add_document(document)

    async def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档"""
        indexer = self.components.get(RAGComponent.INDEXER)
        if not indexer:
            self.logger.warning("Indexer not configured")
            return 0

        return await indexer.add_documents(documents)

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        stats = {
            'performance_metrics': self.performance_metrics,
            'component_stats': {},
            'retrieval_history_size': len(self.retrieval_history)
        }

        # 获取各组件统计
        for component_name, component in self.components.items():
            stats['component_stats'][component_name.value] = component.get_stats()

        return stats
```

## 🔍 文档检索器

### 向量检索实现
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class VectorRetriever(BaseRAGComponent):
    """向量检索器"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dimension: int = 384):
        super().__init__("vector_retriever")
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.index = None
        self.documents = []
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化向量检索器"""
        try:
            # 加载模型
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded model: {self.model_name}")

            # 初始化FAISS索引
            self.index = faiss.IndexFlatIP(self.dimension)
            self.logger.info("Initialized FAISS index")

            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize vector retriever: {e}")
            return False

    async def retrieve(self, query: str, top_k: int = 5,
                       method: RetrievalMethod = RetrievalMethod.VECTOR_SEARCH) -> RetrievalResult:
        """执行向量检索"""
        if not self.is_initialized:
            raise RuntimeError("Vector retriever not initialized")

        start_time = time.time()

        try:
            # 编码查询
            query_embedding = self.model.encode([query])[0]
            query_embedding = query_embedding.astype('float32')

            # 归一化查询向量
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # 执行检索
            search_k = min(top_k * 2, len(self.documents))  # 检索更多候选
            scores, indices = self.index.search(
                np.array([query_embedding]), search_k
            )

            # 构建结果
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.score = float(score)
                    retrieved_docs.append(doc)

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query=query,
                documents=retrieved_docs[:top_k],
                retrieval_time=retrieval_time,
                total_candidates=len(self.documents),
                method=RetrievalMethod.VECTOR_SEARCH,
                metadata={
                    'model_name': self.model_name,
                    'index_type': 'FAISS FlatIP',
                    'search_k': search_k
                }
            )

        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            raise

    async def add_document(self, document: Document) -> bool:
        """添加文档到索引"""
        try:
            # 生成文档嵌入
            if not document.content:
                self.logger.warning(f"Document {document.id} has no content")
                return False

            content = f"{document.title} {document.content}"
            embedding = self.model.encode([content])[0]
            embedding = embedding.astype('float32')

            # 归一化嵌入向量
            embedding = embedding / np.linalg.norm(embedding)

            # 更新文档
            document.embedding = embedding.tolist()
            doc_index = len(self.documents)
            self.documents.append(document)

            # 添加到索引
            self.index.add(np.array([embedding]))

            self.logger.info(f"Added document {document.id} to index")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add document to index: {e}")
            return False

    async def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档"""
        added_count = 0

        for document in documents:
            if await self.add_document(document):
                added_count += 1

        return added_count

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'is_initialized': self.is_initialized,
            'total_documents': len(self.documents),
            'index_type': type(self.index).__name__,
            'index_ntotal': self.index.ntotal if self.index else 0
        }

class HybridRetriever(BaseRAGComponent):
    """混合检索器 - 结合向量检索和关键词检索"""

    def __init__(self, vector_retriever: VectorRetriever, keyword_weight: float = 0.3):
        super().__init__("hybrid_retriever")
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = 1.0 - keyword_weight

    async def initialize(self) -> bool:
        """初始化混合检索器"""
        # 初始化向量检索器
        return await self.vector_retriever.initialize()

    async def retrieve(self, query: str, top_k: int = 5,
                       method: RetrievalMethod = RetrievalMethod.HYBRID) -> RetrievalResult:
        """执行混合检索"""
        start_time = time.time()

        try:
            # 并行执行两种检索
            vector_task = self.vector_retriever.retrieve(query, top_k * 2)
            keyword_task = self._keyword_search(query, top_k * 2)

            vector_result, keyword_result = await asyncio.gather(
                vector_task, keyword_task
            )

            # 合并和重排序结果
            merged_docs = self._merge_results(
                vector_result.documents,
                keyword_result.documents,
                query
            )

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query=query,
                documents=merged_docs[:top_k],
                retrieval_time=retrieval_time,
                total_candidates=len(merged_docs),
                method=RetrievalMethod.HYBRID,
                metadata={
                    'vector_weight': self.vector_weight,
                    'keyword_weight': self.keyword_weight,
                    'vector_candidates': len(vector_result.documents),
                    'keyword_candidates': len(keyword_result.documents)
                }
            )

        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            raise

    async def _keyword_search(self, query: str, top_k: int) -> List[Document]:
        """关键词搜索"""
        # 简化的关键词搜索实现
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in self.vector_retriever.documents:
            content_lower = f"{doc.title} {doc.content}".lower()
            doc_terms = set(content_lower.split())

            # 计算匹配分数
            intersection = query_terms & doc_terms
            union = query_terms | doc_terms

            if intersection:
                jaccard_similarity = len(intersection) / len(union)
                doc.score = jaccard_similarity
                scored_docs.append(doc)

        # 按分数排序
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:top_k]

    def _merge_results(self, vector_docs: List[Document], keyword_docs: List[Document],
                     query: str) -> List[Document]:
        """合并检索结果"""
        # 创建文档ID到文档的映射
        all_docs = {}

        # 添加向量检索结果
        for doc in vector_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = {
                    'doc': doc,
                    'vector_score': doc.score,
                    'keyword_score': 0.0
                }
            else:
                all_docs[doc.id]['vector_score'] = max(
                    all_docs[doc.id]['vector_score'],
                    doc.score
                )

        # 添加关键词检索结果
        for doc in keyword_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'keyword_score': doc.score
                }
            else:
                all_docs[doc.id]['keyword_score'] = max(
                    all_docs[doc.id]['keyword_score'],
                    doc.score
                )

        # 计算混合分数
        for doc_info in all_docs.values():
            combined_score = (
                self.vector_weight * doc_info['vector_score'] +
                self.keyword_weight * doc_info['keyword_score']
            )
            doc_info['doc'].score = combined_score

        # 排序并返回
        merged_docs = [info['doc'] for info in all_docs.values()]
        merged_docs.sort(key=lambda x: x.score, reverse=True)

        return merged_docs

    def get_stats(self) -> Dict[str, Any]:
        """获取混合检索器统计"""
        return {
            'vector_retriever_stats': self.vector_retriever.get_stats(),
            'keyword_weight': self.keyword_weight,
            'vector_weight': self.vector_weight
        }
```

## 📄 文档索引器

### FAISS索引实现
```python
import faiss
import numpy as np
import pickle
import os
from pathlib import Path

class FAISSIndexer(BaseRAGComponent):
    """FAISS索引器"""

    def __init__(self, index_path: str = "faiss_index.index",
                 documents_path: str = "documents.pkl",
                 index_type: str = "flat"):
        super().__init__("faiss_indexer")
        self.index_path = Path(index_path)
        self.documents_path = Path(documents_path)
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.dimension = None

    async def initialize(self) -> bool:
        """初始化索引器"""
        try:
            # 尝试加载现有索引
            if await self._load_index():
                self.logger.info("Loaded existing index")
            else:
                self.logger.info("No existing index found, creating new one")
                self._create_empty_index()

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize indexer: {e}")
            return False

    def _create_empty_index(self):
        """创建空索引"""
        if self.dimension is None:
            self.dimension = 384  # 默认维度

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

    async def _load_index(self) -> bool:
        """加载索引和文档"""
        # 加载索引
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d
        else:
            return False

        # 加载文档
        if self.documents_path.exists():
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            return False

        return True

    async def _save_index(self):
        """保存索引和文档"""
        try:
            # 保存索引
            if self.index:
                faiss.write_index(self.index, str(self.index_path))

            # 保存文档
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)

            self.logger.info("Index and documents saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")

    async def add_document(self, document: Document) -> bool:
        """添加文档到索引"""
        if document.embedding is None:
            self.logger.warning(f"Document {document.id} has no embedding")
            return False

        try:
            # 添加到文档列表
            self.documents.append(document)

            # 添加到索引
            if self.index_type == "ivf":
                # IVF索引需要训练
                if len(self.documents) % 1000 == 0:  # 每1000个文档重新训练
                    await self._retrain_index()
                else:
                    # 直接添加
                    embedding = np.array([document.embedding], dtype='float32')
                    self.index.add(embedding)
            else:
                embedding = np.array([document.embedding], dtype='float32')
                self.index.add(embedding)

            return True

        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            return False

    async def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档"""
        added_count = 0
        embeddings = []

        # 收集有效嵌入
        for doc in documents:
            if doc.embedding:
                embeddings.append(doc.embedding)
                self.documents.append(doc)
                added_count += 1

        if embeddings:
            try:
                embeddings_array = np.array(embeddings, dtype='float32')

                if self.index_type == "ivf":
                    await self._retrain_index()
                else:
                    self.index.add(embeddings_array)

                self.logger.info(f"Added {added_count} documents to index")

            except Exception as e:
                self.logger.error(f"Failed to add embeddings: {e}")
                added_count = 0

        return added_count

    async def _retrain_index(self):
        """重新训练索引（主要用于IVF）"""
        if len(self.documents) < 1000:
            return  # 文档太少，不需要重新训练

        try:
            # 提取所有嵌入
            embeddings = np.array(
                [doc.embedding for doc in self.documents if doc.embedding],
                dtype='float32'
            )

            # 重新创建索引
            if self.index_type == "ivf":
                nlist = min(100, len(self.documents) // 10)
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

                # 训练索引
                index.train(embeddings)
                index.add(embeddings)
                self.index = index

            self.logger.info("Index retrained successfully")

        except Exception as e:
            self.logger.error(f"Failed to retrain index: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取索引器统计"""
        return {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_documents': len(self.documents),
            'index_ntotal': self.index.ntotal if self.index else 0,
            'is_trained': hasattr(self.index, 'is_trained') and self.index.is_trained if self.index else False
        }

    async def cleanup(self):
        """清理资源"""
        try:
            await self._save_index()
        except Exception as e:
            self.logger.error(f"Failed to save index during cleanup: {e}")

class DocumentChunker:
    """文档分块器"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Document]:
        """将文档分块"""
        if not document.content:
            return [document]

        content = document.content
        chunks = []

        # 按句子分块
        sentences = self._split_sentences(content)
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            # 检查添加这个句子是否会超过块大小
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # 创建文档块
                chunk_doc = self._create_chunk_document(
                    document, current_chunk, chunk_id
                )
                chunks.append(chunk_doc)

                current_chunk = sentence  # 重置，保留重叠部分
                chunk_id += 1
            else:
                current_chunk += sentence

        # 处理最后一个块
        if current_chunk:
            chunk_doc = self._create_chunk_document(
                document, current_chunk, chunk_id
            )
            chunks.append(chunk_doc)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        import re

        # 使用正则表达式分割句子
        sentence_endings = r'[.!?]+(?=\s|$)'
        sentences = re.split(sentence_endings, text)

        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunk_document(self, original_doc: Document,
                           content: str, chunk_id: int) -> Document:
        """创建文档块"""
        return Document(
            id=f"{original_doc.id}_chunk_{chunk_id}",
            content=content,
            title=f"{original_doc.title} (Chunk {chunk_id + 1})",
            url=original_doc.url,
            metadata={
                **original_doc.metadata,
                'original_doc_id': original_doc.id,
                'chunk_id': chunk_id,
                'chunk_size': len(content)
            },
            source=original_doc.source,
            chunk_id=str(chunk_id)
        )
```

## 🔄 文档重排序器

### 语义重排序实现
```python
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class SemanticReranker(BaseRAGComponent):
    """语义重排序器"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 20):
        super().__init__("semantic_reranker")
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化重排序器"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.is_initialized = True
            self.logger.info(f"Loaded reranker model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize reranker: {e}")
            return False

    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """重排序文档"""
        if not self.is_initialized:
            self.logger.warning("Reranker not initialized")
            return documents

        if not documents:
            return documents

        try:
            # 限制候选文档数量
            candidates = documents[:self.top_k]

            # 编码查询和文档
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            doc_embeddings = self.model.encode(
                [doc.content for doc in candidates],
                convert_to_tensor=True
            )

            # 计算余弦相似度
            cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

            # 计算重排序分数
            # 结合原始检索分数和语义分数
            reranked_docs = []
            for i, (doc, cosine_score) in enumerate(zip(candidates, cosine_scores)):
                # 加权组合分数
                semantic_weight = 0.7
                original_weight = 0.3

                # 归一化原始分数
                original_score = max(0.0, min(1.0, doc.score))

                combined_score = (
                    semantic_weight * cosine_score.item() +
                    original_weight * original_score
                )

                # 更新文档分数
                doc.score = combined_score
                reranked_docs.append(doc)

            # 按重排序分数排序
            reranked_docs.sort(key=lambda x: x.score, reverse=True)

            self.logger.info(f"Reranked {len(documents)} documents")
            return reranked_docs

        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return documents

    def get_stats(self) -> Dict[str, Any]:
        """获取重排序器统计"""
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'top_k': self.top_k
        }

class CrossEncoderReranker(BaseRAGComponent):
    """跨编码器重排序器"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__("cross_encoder_reranker")
        self.model_name = model_name
        self.model = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化跨编码器重排序器"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.is_initialized = True
            self.logger.info(f"Loaded cross-encoder model: {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize cross-encoder reranker: {e}")
            return False

    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """使用跨编码器重排序"""
        if not self.is_initialized:
            self.logger.warning("Cross-encoder reranker not initialized")
            return documents

        if not documents:
            return documents

        try:
            # 构建查询-文档对
            query_doc_pairs = [
                (query, doc.content)
                for doc in documents[:50]  # 限制候选数量
            ]

            # 计算相似度分数
            scores = self.model.predict(query_doc_pairs)

            # 更新文档分数
            for i, (doc, score) in enumerate(zip(documents, scores)):
                doc.score = float(score)

            # 按分数排序
            documents.sort(key=lambda x: x.score, reverse=True)

            self.logger.info(f"Cross-encoder reranked {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            return documents

    def get_stats(self) -> Dict[str, Any]:
        """获取跨编码器重排序器统计"""
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'model_type': 'cross_encoder'
        }
```

## 🤖 回答生成器

### 上下文感知生成器
```python
class ContextAwareGenerator(BaseRAGComponent):
    """上下文感知生成器"""

    def __init__(self, llm_client, template: str = None):
        super().__init__("context_aware_generator")
        self.llm_client = llm_client
        self.template = template or self._get_default_template()

    async def initialize(self) -> bool:
        """初始化生成器"""
        try:
            # 测试LLM连接
            test_response = await self.llm_client.generate("测试")
            self.is_initialized = True
            self.logger.info("Generator initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize generator: {e}")
            return False

    async def generate(self, query: str, context: str) -> Dict[str, Any]:
        """生成回答"""
        if not self.is_initialized:
            return {
                'answer': "生成器未初始化",
                'success': False
            }

        try:
            # 构建提示
            prompt = self._build_prompt(query, context)

            # 生成回答
            start_time = time.time()
            response = await self.llm_client.generate(prompt)
            generation_time = time.time() - start_time

            # 解析响应
            answer = self._parse_response(response)

            return {
                'answer': answer,
                'success': True,
                'generation_time': generation_time,
                'prompt_length': len(prompt),
                'context_length': len(context)
            }

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                'answer': f"生成失败: {str(e)}",
                'success': False,
                'error': str(e)
            }

    def _build_prompt(self, query: str, context: str) -> str:
        """构建提示"""
        return self.template.format(
            query=query,
            context=context
        )

    def _parse_response(self, response: str) -> str:
        """解析响应"""
        # 简化的响应解析
        if response.startswith("回答："):
            return response[3:].strip()
        elif "无法回答" in response:
            return "抱歉，基于提供的信息我无法回答这个问题。"
        else:
            return response.strip()

    def _get_default_template(self) -> str:
        """获取默认模板"""
        return """请基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文信息：
{context}

问题：{query}

回答："""

    def get_stats(self) -> Dict[str, Any]:
        """获取生成器统计"""
        return {
            'template_length': len(self.template),
            'is_initialized': hasattr(self, 'is_initialized') and self.is_initialized
        }

class StreamingGenerator(ContextAwareGenerator):
    """流式生成器"""

    async def generate_stream(self, query: str, context: str):
        """流式生成回答"""
        if not self.is_initialized:
            yield "生成器未初始化"
            return

        try:
            # 构建提示
            prompt = self._build_prompt(query, context)

            # 流式生成
            async for chunk in self.llm_client.generate_stream(prompt):
                yield chunk

        except Exception as e:
            yield f"生成失败: {str(e)}"
```

## 📊 RAG系统整合

### 完整的RAG实现
```python
async def create_rag_system():
    """创建完整的RAG系统"""

    # 配置
    config = {
        'retriever': {
            'type': 'hybrid',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384
        },
        'indexer': {
            'type': 'faiss',
            'index_path': 'rag_index.faiss',
            'documents_path': 'rag_documents.pkl'
        },
        'reranker': {
            'type': 'semantic',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'generator': {
            'type': 'context_aware',
            'template': None  # 使用默认模板
        }
    }

    # 创建RAG系统
    rag_system = RAGSystem(config)

    # 初始化系统
    if await rag_system.initialize():
        print("RAG系统初始化成功")
    else:
        print("RAG系统初始化失败")
        return None

    return rag_system

async def add_sample_documents(rag_system):
    """添加示例文档"""
    documents = [
        Document(
            id="doc1",
            title="人工智能基础",
            content="人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI包括机器学习、深度学习、自然语言处理等多个子领域。",
            url="https://example.com/ai-basics",
            metadata={'category': 'technology', 'difficulty': 'beginner'}
        ),
        Document(
            id="doc2",
            title="机器学习算法",
            content="机器学习是AI的核心技术之一，包括监督学习、无监督学习和强化学习。常见算法有线性回归、决策树、支持向量机、神经网络等。",
            url="https://example.com/ml-algorithms",
            metadata={'category': 'technology', 'difficulty': 'intermediate'}
        ),
        Document(
            id="doc3",
            title="深度学习原理",
            content="深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。常见的深度学习架构包括CNN、RNN、Transformer等。深度学习在图像识别、自然语言处理等领域取得了突破性进展。",
            url="https://example.com/deep-learning",
            metadata={'category': 'technology', 'difficulty': 'advanced'}
        )
    ]

    added_count = await rag_system.add_documents(documents)
    print(f"成功添加 {added_count} 个文档到RAG系统")
    return added_count

async def test_rag_queries(rag_system):
    """测试RAG查询"""
    test_queries = [
        "什么是人工智能？",
        "机器学习有哪些算法？",
        "深度学习和传统机器学习的区别是什么？",
        "如何开始学习AI？"
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        print("-" * 50)

        result = await rag_system.query(query, top_k=3, rerank=True)

        if result['success']:
            retrieval = result['retrieval']
            generation = result['generation']

            print(f"检索到 {len(retrieval.documents)} 个相关文档:")
            for i, doc in enumerate(retrieval.documents, 1):
                print(f"{i}. {doc.title} (分数: {doc.score:.4f})")
                print(f"   内容: {doc.content[:100]}...")
                print(f"   来源: {doc.url}")

            print(f"\n生成的回答:")
            print(generation['answer'])
            print(f"生成时间: {generation.get('generation_time', 0):.2f}秒")
        else:
            print(f"查询失败: {result.get('error', '未知错误')}")

        print("=" * 60)

async def rag_system_demo():
    """RAG系统演示"""
    print("🔍 RAG系统演示")
    print("=" * 60)

    # 创建RAG系统
    rag_system = await create_rag_system()
    if not rag_system:
        return

    # 添加示例文档
    await add_sample_documents(rag_system)

    # 测试查询
    await test_rag_queries(rag_system)

    # 显示系统统计
    stats = rag_system.get_stats()
    print(f"\n📊 系统统计:")
    print(f"总查询数: {stats['performance_metrics']['total_queries']}")
    print(f"平均检索时间: {stats['performance_metrics']['avg_retrieval_time']:.4f}秒")

    for component_name, component_stats in stats['component_stats'].items():
        print(f"{component_name}: {component_stats}")

# 运行演示
# asyncio.run(rag_system_demo())
```

## 📝 总结

RAG系统是增强Agent知识能力的重要技术，本文档介绍了从检索到生成的完整实现。

### 🎯 关键要点
- **模块化设计**: 清晰的组件分离和接口定义
- **多策略检索**: 向量、关键词、混合检索方法
- **智能重排序**: 语义和跨编码器重排序技术
- **上下文感知**: 基于检索上下文的生成
- **性能优化**: 索引优化和缓存机制

### 🚀 实现特色
- **混合检索**: 结合多种检索策略提高召回率
- **动态索引**: 支持文档的动态添加和索引更新
- **实时重排序**: 使用深度学习模型进行精排序
- **流式生成**: 支持大段文本的流式生成
- **性能监控**: 完整的性能指标收集和监控

### 🔄 下一步
- 学习[上下文管理](05-上下文管理.md)
- 掌握[决策规划](06-决策规划.md)
- 探索[多模态学习](../multimodal/01-基础概念.md)
- 了解[模型部署](../deployment/01-模型量化.md)
