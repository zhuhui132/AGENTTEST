# 🔍 RAG基础架构

## 📚 概述

检索增强生成(RAG, Retrieval-Augmented Generation)是将信息检索与生成模型结合的技术，使AI系统能够基于外部知识库生成更准确、更及时的回答。本文档详细介绍RAG的基础架构和实现方法。

## 🏗️ RAG核心架构

### 基本工作流程
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging

@dataclass
class RAGQuery:
    """RAG查询对象"""
    query: str
    top_k: int = 5
    retrieval_method: str = "hybrid"
    context_length: int = 4000
    temperature: float = 0.7
    min_relevance_score: float = 0.5

@dataclass
class RAGContext:
    """RAG上下文对象"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context_text: str
    retrieval_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    """RAG响应对象"""
    answer: str
    context: RAGContext
    generation_time: float = 0.0
    source_documents: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseRAGSystem(ABC):
    """RAG系统基础抽象类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_metrics = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_generation_time': 0.0,
            'cache_hit_rate': 0.0
        }

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化RAG系统"""
        pass

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档"""
        pass

    @abstractmethod
    async def generate(self, query: str, context: str) -> str:
        """基于上下文生成回答"""
        pass

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """完整的RAG查询流程"""
        start_time = time.time()

        try:
            # 1. 检索阶段
            retrieval_start = time.time()
            retrieved_docs = await self.retrieve(rag_query.query, rag_query.top_k)
            retrieval_time = time.time() - retrieval_start

            # 2. 过滤和排序
            filtered_docs = self._filter_retrieved_docs(retrieved_docs, rag_query)

            # 3. 构建上下文
            context_start = time.time()
            context = self._build_context(rag_query.query, filtered_docs)
            context_time = time.time() - context_start

            # 4. 生成回答
            generation_start = time.time()
            answer = await self.generate(rag_query.query, context)
            generation_time = time.time() - generation_start

            # 5. 构建响应
            rag_context = RAGContext(
                query=rag_query.query,
                retrieved_docs=filtered_docs,
                context_text=context,
                retrieval_time=retrieval_time,
                metadata={
                    'method': rag_query.retrieval_method,
                    'total_retrieved': len(retrieved_docs),
                    'filtered_count': len(filtered_docs),
                    'context_time': context_time
                }
            )

            response = RAGResponse(
                answer=answer,
                context=rag_context,
                generation_time=generation_time,
                source_documents=[doc.get('id', '') for doc in filtered_docs],
                confidence=self._calculate_confidence(filtered_docs),
                metadata={
                    'total_time': time.time() - start_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time
                }
            )

            # 6. 更新性能指标
            self._update_performance_metrics(retrieval_time, generation_time)

            return response

        except Exception as e:
            self.logger.error(f"RAG查询失败: {e}")
            return RAGResponse(
                answer=f"查询处理失败: {str(e)}",
                context=RAGContext(query=rag_query.query, retrieved_docs=[], context_text=""),
                generation_time=0.0,
                confidence=0.0,
                metadata={'error': str(e)}
            )

    def _filter_retrieved_docs(self, docs: List[Dict[str, Any]],
                           rag_query: RAGQuery) -> List[Dict[str, Any]]:
        """过滤检索到的文档"""
        if not docs:
            return []

        filtered = []
        for doc in docs:
            # 基于相关性分数过滤
            relevance_score = doc.get('relevance_score', 0.0)
            if relevance_score >= rag_query.min_relevance_score:
                filtered.append(doc)

        # 按相关性分数排序
        filtered.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

        # 限制上下文长度
        if rag_query.context_length > 0:
            filtered = self._limit_context_length(filtered, rag_query.context_length)

        return filtered

    def _build_context(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """构建上下文文本"""
        if not docs:
            return "没有找到相关文档。"

        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get('title', f"文档{i}")
            content = doc.get('content', '')
            source = doc.get('url', doc.get('source', ''))

            context_part = f"[文档{i}] {title}\n{content}"
            if source:
                context_part += f"\n来源: {source}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def _limit_context_length(self, docs: List[Dict[str, Any]],
                          max_length: int) -> List[Dict[str, Any]]:
        """限制上下文长度"""
        current_length = 0
        limited_docs = []

        for doc in docs:
            doc_length = len(doc.get('content', ''))
            if current_length + doc_length <= max_length:
                limited_docs.append(doc)
                current_length += doc_length
            else:
                break

        return limited_docs

    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """计算回答置信度"""
        if not docs:
            return 0.0

        # 基于检索到的文档数量和质量计算置信度
        doc_count = len(docs)
        avg_relevance = sum(doc.get('relevance_score', 0.0) for doc in docs) / doc_count

        # 置信度 = 文档数量权重 × 平均相关性权重
        confidence = min(1.0, (doc_count / 5.0) * 0.5 + avg_relevance * 0.5)

        return confidence

    def _update_performance_metrics(self, retrieval_time: float, generation_time: float):
        """更新性能指标"""
        self.performance_metrics['total_queries'] += 1

        # 更新平均检索时间
        current_avg_retrieval = self.performance_metrics['avg_retrieval_time']
        n = self.performance_metrics['total_queries']
        self.performance_metrics['avg_retrieval_time'] = (
            current_avg_retrieval * (n - 1) + retrieval_time
        ) / n

        # 更新平均生成时间
        current_avg_generation = self.performance_metrics['avg_generation_time']
        self.performance_metrics['avg_generation_time'] = (
            current_avg_generation * (n - 1) + generation_time
        ) / n

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_metrics.copy()
```

## 🔍 文档检索模块

### 多种检索策略
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict, Any

class DocumentRetriever:
    """文档检索器"""

    def __init__(self, documents: List[Dict[str, Any]],
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.model = None
        self.index = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化检索器"""
        try:
            # 加载嵌入模型
            self.model = SentenceTransformer(self.embedding_model)

            # 构建文档嵌入
            embeddings = []
            valid_docs = []

            for i, doc in enumerate(self.documents):
                content = f"{doc.get('title', '')} {doc.get('content', '')}"
                if content.strip():
                    embedding = self.model.encode(content)
                    embeddings.append(embedding)
                    valid_docs.append(doc)

            # 构建FAISS索引
            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                # 归一化嵌入向量
                embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

                self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
                self.index.add(embeddings_array)

                # 更新文档列表
                self.documents = valid_docs
                self.is_initialized = True
                return True
            else:
                self.logger.error("没有有效的文档进行索引")
                return False

        except Exception as e:
            self.logger.error(f"检索器初始化失败: {e}")
            return False

    async def retrieve(self, query: str, top_k: int = 5,
                    method: str = "vector") -> List[Dict[str, Any]]:
        """检索文档"""
        if not self.is_initialized:
            raise RuntimeError("检索器未初始化")

        if method == "vector":
            return await self._vector_search(query, top_k)
        elif method == "keyword":
            return await self._keyword_search(query, top_k)
        elif method == "hybrid":
            return await self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"不支持的检索方法: {method}")

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            # 编码查询
            query_embedding = self.model.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # 执行搜索
            search_k = min(top_k * 2, len(self.documents))
            scores, indices = self.index.search(
                np.array([query_embedding]), search_k
            )

            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    results.append(doc)

            return results[:top_k]

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词检索"""
        try:
            query_terms = set(query.lower().split())
            results = []

            for doc in self.documents:
                content = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
                content_terms = set(content.split())

                # 计算匹配分数
                intersection = query_terms & content_terms
                union = query_terms | content_terms

                if intersection:
                    jaccard_similarity = len(intersection) / len(union)
                    doc_copy = doc.copy()
                    doc_copy['relevance_score'] = jaccard_similarity
                    results.append(doc_copy)

            # 按分数排序
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            self.logger.error(f"关键词检索失败: {e}")
            return []

    async def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """混合检索（向量+关键词）"""
        try:
            # 并行执行两种检索
            import asyncio
            vector_task = self._vector_search(query, top_k * 2)
            keyword_task = self._keyword_search(query, top_k * 2)

            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task
            )

            # 合并结果
            all_results = []
            doc_scores = {}

            # 添加向量检索结果
            for doc in vector_results:
                doc_id = doc.get('id', str(doc))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'vector_score': doc.get('relevance_score', 0.0),
                        'keyword_score': 0.0,
                        'doc': doc
                    }
                else:
                    doc_scores[doc_id]['vector_score'] = max(
                        doc_scores[doc_id]['vector_score'],
                        doc.get('relevance_score', 0.0)
                    )

            # 添加关键词检索结果
            for doc in keyword_results:
                doc_id = doc.get('id', str(doc))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'vector_score': 0.0,
                        'keyword_score': doc.get('relevance_score', 0.0),
                        'doc': doc
                    }
                else:
                    doc_scores[doc_id]['keyword_score'] = max(
                        doc_scores[doc_id]['keyword_score'],
                        doc.get('relevance_score', 0.0)
                    )

            # 计算混合分数
            for doc_id, score_data in doc_scores.items():
                vector_score = score_data['vector_score']
                keyword_score = score_data['keyword_score']

                # 混合分数 = 0.7 * 向量分数 + 0.3 * 关键词分数
                hybrid_score = 0.7 * vector_score + 0.3 * keyword_score
                doc = score_data['doc']
                doc['relevance_score'] = hybrid_score
                all_results.append(doc)

            # 按混合分数排序
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            return []

class DocumentIndexer:
    """文档索引管理器"""

    def __init__(self, index_path: str = "document_index.faiss"):
        self.index_path = index_path
        self.index = None
        self.document_mapping = []

    async def build_index(self, documents: List[Dict[str, Any]],
                       embedding_model: SentenceTransformer) -> bool:
        """构建文档索引"""
        try:
            if not documents:
                self.logger.warning("文档列表为空")
                return False

            # 生成文档嵌入
            embeddings = []
            self.document_mapping = []

            for doc in documents:
                content = f"{doc.get('title', '')} {doc.get('content', '')}"
                if content.strip():
                    embedding = embedding_model.encode(content)
                    embeddings.append(embedding)
                    self.document_mapping.append(doc)

            if not embeddings:
                self.logger.error("没有有效的文档进行索引")
                return False

            # 创建FAISS索引
            embeddings_array = np.array(embeddings).astype('float32')
            embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

            self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
            self.index.add(embeddings_array)

            # 保存索引
            faiss.write_index(self.index, self.index_path)

            self.logger.info(f"成功构建索引，文档数量: {len(self.document_mapping)}")
            return True

        except Exception as e:
            self.logger.error(f"索引构建失败: {e}")
            return False

    def load_index(self, embedding_model: SentenceTransformer) -> bool:
        """加载现有索引"""
        try:
            if not os.path.exists(self.index_path):
                self.logger.warning(f"索引文件不存在: {self.index_path}")
                return False

            self.index = faiss.read_index(self.index_path)
            self.logger.info(f"成功加载索引，文档数量: {self.index.ntotal}")
            return True

        except Exception as e:
            self.logger.error(f"索引加载失败: {e}")
            return False
```

## 🤖 生成模块

### 上下文感知生成
```python
import openai
from typing import List, Dict, Any
import json

class ContextAwareGenerator:
    """上下文感知的回答生成器"""

    def __init__(self, model_name: str = "gpt-3.5-turbo",
                 api_key: str = None,
                 max_tokens: int = 1000):
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.client = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化生成器"""
        try:
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                # 使用环境变量
                self.client = openai.OpenAI()

            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"生成器初始化失败: {e}")
            return False

    async def generate(self, query: str, context: str,
                    temperature: float = 0.7,
                    max_context_length: int = 4000) -> str:
        """基于上下文生成回答"""
        if not self.is_initialized:
            raise RuntimeError("生成器未初始化")

        try:
            # 构建提示
            prompt = self._build_prompt(query, context, max_context_length)

            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个基于给定文档回答问题的智能助手。请基于提供的上下文信息准确回答用户的问题。如果上下文中没有相关信息，请说明无法回答。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content
            return answer.strip()

        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}"

    def _build_prompt(self, query: str, context: str, max_length: int) -> str:
        """构建提示"""
        # 限制上下文长度
        if len(context) > max_length:
            # 智能截断，保留完整的文档
            truncated_context = []
            current_length = 0

            for doc_part in context.split("\n\n"):
                if current_length + len(doc_part) + 4 <= max_length:  # +4 for "..."
                    truncated_context.append(doc_part)
                    current_length += len(doc_part) + 2  # +2 for "\n\n"
                else:
                    break

            context = "\n\n".join(truncated_context)
            if current_length < len(context):
                context += "\n\n[...文档被截断...]"

        prompt = f"""请基于以下文档信息回答问题。如果文档中没有相关信息，请说明无法回答。

上下文信息：
{context}

问题：{query}

请基于上下文信息回答问题："""

        return prompt

class LocalLLMGenerator:
    """本地LLM生成器"""

    def __init__(self, model_path: str, max_tokens: int = 1000):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """初始化本地LLM"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

            # 如果有GPU，使用GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()
            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"本地LLM初始化失败: {e}")
            return False

    async def generate(self, query: str, context: str,
                    temperature: float = 0.7,
                    max_context_length: int = 4000) -> str:
        """本地模型生成"""
        if not self.is_initialized:
            raise RuntimeError("本地LLM未初始化")

        try:
            # 限制上下文长度
            if len(context) > max_context_length:
                context = context[:max_context_length-10] + "[...截断]"

            # 构建完整输入
            prompt = self._build_prompt(query, context)

            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if hasattr(self.model, 'cuda') and self.model.cuda.is_cuda:
                inputs = inputs.cuda()

            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )

            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取生成的部分
            if prompt in response:
                answer = response[len(prompt):].strip()
            else:
                answer = response.strip()

            return answer

        except Exception as e:
            self.logger.error(f"本地生成失败: {e}")
            return f"生成失败: {str(e)}"

    def _build_prompt(self, query: str, context: str) -> str:
        """构建本地模型提示"""
        return f"""上下文：
{context}

问题：{query}

回答："""
```

## 🔄 完整RAG实现

### 端到端RAG系统
```python
class CompleteRAGSystem(BaseRAGSystem):
    """完整的RAG系统实现"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.document_store = None
        self.retriever = None
        self.generator = None

    async def initialize(self) -> bool:
        """初始化完整RAG系统"""
        try:
            # 1. 初始化文档存储
            await self._initialize_document_store()

            # 2. 初始化检索器
            await self._initialize_retriever()

            # 3. 初始化生成器
            await self._initialize_generator()

            return True

        except Exception as e:
            self.logger.error(f"RAG系统初始化失败: {e}")
            return False

    async def _initialize_document_store(self):
        """初始化文档存储"""
        from .document_store import DocumentStore

        self.document_store = DocumentStore(
            storage_type=self.config.get('storage_type', 'local'),
            storage_path=self.config.get('storage_path', 'documents')
        )

        await self.document_store.initialize()

    async def _initialize_retriever(self):
        """初始化检索器"""
        embedding_model = self.config.get('embedding_model',
                                       'sentence-transformers/all-MiniLM-L6-v2')
        documents = await self.document_store.get_all_documents()

        self.retriever = DocumentRetriever(documents, embedding_model)
        await self.retriever.initialize()

    async def _initialize_generator(self):
        """初始化生成器"""
        generator_type = self.config.get('generator_type', 'openai')

        if generator_type == 'openai':
            api_key = self.config.get('openai_api_key')
            self.generator = ContextAwareGenerator(
                model_name=self.config.get('model_name', 'gpt-3.5-turbo'),
                api_key=api_key,
                max_tokens=self.config.get('max_tokens', 1000)
            )
        elif generator_type == 'local':
            model_path = self.config.get('local_model_path')
            self.generator = LocalLLMGenerator(
                model_path=model_path,
                max_tokens=self.config.get('max_tokens', 1000)
            )
        else:
            raise ValueError(f"不支持的生成器类型: {generator_type}")

        await self.generator.initialize()

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索文档"""
        retrieval_method = self.config.get('retrieval_method', 'hybrid')
        return await self.retriever.retrieve(query, top_k, retrieval_method)

    async def generate(self, query: str, context: str) -> str:
        """生成回答"""
        temperature = self.config.get('temperature', 0.7)
        max_context_length = self.config.get('max_context_length', 4000)

        return await self.generator.generate(
            query, context, temperature, max_context_length
        )

# 使用示例
async def rag_system_demo():
    """RAG系统演示"""

    # 配置
    config = {
        'storage_type': 'local',
        'storage_path': 'sample_documents',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'retrieval_method': 'hybrid',
        'generator_type': 'openai',
        'model_name': 'gpt-3.5-turbo',
        'openai_api_key': 'your-api-key',  # 需要设置实际的API密钥
        'max_tokens': 1000,
        'temperature': 0.7,
        'max_context_length': 4000
    }

    # 创建RAG系统
    rag_system = CompleteRAGSystem(config)

    # 初始化
    if await rag_system.initialize():
        print("✅ RAG系统初始化成功")
    else:
        print("❌ RAG系统初始化失败")
        return

    # 示例文档
    sample_documents = [
        {
            'id': 'doc1',
            'title': '人工智能基础',
            'content': '人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI包括机器学习、深度学习、自然语言处理等多个子领域。',
            'url': 'https://example.com/ai-basics',
            'metadata': {'category': 'technology', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc2',
            'title': '机器学习算法',
            'content': '机器学习是AI的核心技术，包括监督学习、无监督学习和强化学习。常见算法有线性回归、决策树、支持向量机、神经网络等。',
            'url': 'https://example.com/ml-algorithms',
            'metadata': {'category': 'technology', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc3',
            'title': '深度学习原理',
            'content': '深度学习是机器学习的子集，使用多层神经网络来学习数据的复杂模式。CNN、RNN、Transformer是常见的深度学习架构。',
            'url': 'https://example.com/dl-principles',
            'metadata': {'category': 'technology', 'difficulty': 'advanced'}
        }
    ]

    # 添加文档到系统
    for doc in sample_documents:
        await rag_system.document_store.add_document(doc)

    # 重新构建索引
    await rag_system.retriever.initialize()

    # 测试查询
    test_queries = [
        "什么是人工智能？",
        "机器学习有哪些算法？",
        "深度学习和传统机器学习的区别？",
        "如何开始学习AI？"
    ]

    print("\n🔍 开始测试RAG查询:")
    print("=" * 60)

    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        print("-" * 40)

        # 创建RAG查询
        rag_query = RAGQuery(
            query=query,
            top_k=3,
            retrieval_method='hybrid',
            temperature=0.7
        )

        # 执行查询
        response = await rag_system.query(rag_query)

        # 显示结果
        print(f"📄 检索到 {len(response.context.retrieved_docs)} 个相关文档")
        for i, doc in enumerate(response.context.retrieved_docs, 1):
            print(f"  {i}. {doc['title']} (分数: {doc.get('relevance_score', 0):.3f})")

        print(f"\n🤖 生成的回答:")
        print(response.answer)
        print(f"📊 置信度: {response.confidence:.3f}")
        print(f"⏱️ 总耗时: {response.metadata.get('total_time', 0):.2f}秒")
        print("=" * 60)

    # 显示性能统计
    stats = rag_system.get_performance_stats()
    print(f"\n📊 性能统计:")
    print(f"总查询数: {stats['total_queries']}")
    print(f"平均检索时间: {stats['avg_retrieval_time']:.3f}秒")
    print(f"平均生成时间: {stats['avg_generation_time']:.3f}秒")

# 运行演示
# import asyncio
# asyncio.run(rag_system_demo())
```

## 📊 性能优化

### 缓存和批处理
```python
import time
from typing import Dict, List, Any
from functools import lru_cache
import asyncio

class RAGOptimizer:
    """RAG性能优化器"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.query_cache = {}
        self.cache_ttl = 300  # 5分钟缓存

    @lru_cache(maxsize=1000)
    def _cached_embedding(self, text: str) -> List[float]:
        """缓存的嵌入计算"""
        return self.rag_system.retriever.model.encode([text])[0]

    async def optimized_query(self, query: str, **kwargs) -> RAGResponse:
        """优化的查询处理"""
        start_time = time.time()

        # 检查缓存
        cache_key = self._get_cache_key(query, kwargs)
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            cached_result.metadata['from_cache'] = True
            return cached_result

        # 执行查询
        result = await self.rag_system.query(RAGQuery(query=query, **kwargs))

        # 缓存结果
        self._cache_result(cache_key, result)

        return result

    def _get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        import json

        cache_data = {
            'query': query,
            'params': sorted(params.items())
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[RAGResponse]:
        """获取缓存结果"""
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]

            # 检查缓存是否过期
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['response']
            else:
                del self.query_cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, response: RAGResponse):
        """缓存结果"""
        self.query_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

        # 清理过期缓存
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for cache_key, cache_entry in self.query_cache.items():
            if current_time - cache_entry['timestamp'] >= self.cache_ttl:
                expired_keys.append(cache_key)

        for key in expired_keys:
            del self.query_cache[key]

class BatchRAGProcessor:
    """批量RAG处理器"""

    def __init__(self, rag_system, max_concurrent: int = 5):
        self.rag_system = rag_system
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(self, queries: List[str],
                          **kwargs) -> List[RAGResponse]:
        """批量处理查询"""
        tasks = []

        for query in queries:
            task = self._process_single_query(query, **kwargs)
            tasks.append(task)

        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(RAGResponse(
                    answer=f"查询失败: {str(result)}",
                    context=RAGContext(query=queries[i], retrieved_docs=[], context_text=""),
                    confidence=0.0,
                    metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _process_single_query(self, query: str, **kwargs) -> RAGResponse:
        """处理单个查询（带并发控制）"""
        async with self.semaphore:
            rag_query = RAGQuery(query=query, **kwargs)
            return await self.rag_system.query(rag_query)
```

## 📝 总结

RAG系统是增强AI系统知识能力的重要技术，本文档介绍了RAG的基础架构和完整实现。

### 🎯 关键要点
- **检索增强**: 结合信息检索与生成模型
- **多策略检索**: 向量、关键词、混合检索方法
- **上下文感知**: 基于检索上下文生成回答
- **性能优化**: 缓存和批处理机制

### 🚀 实现特色
- **模块化设计**: 清晰的组件分离和接口定义
- **多模型支持**: OpenAI API和本地模型支持
- **高性能**: FAISS索引和异步处理
- **易扩展**: 支持自定义检索器和生成器

### 🔄 下一步
- 学习[向量检索](02-向量检索.md)
- 了解[检索策略](03-检索策略.md)
- 掌握[生成检索融合](04-生成检索融合.md)
- 探索[Agent集成](../agents/04-RAG系统.md)
