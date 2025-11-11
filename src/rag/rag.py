"""RAG 检索系统组件 - 异步实现"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.exceptions import DocumentNotFoundError, RAGError
from ..core.interfaces import BaseRAG
from ..core.types import Document, RAGConfig


class RAGSystem(BaseRAG):
    """简化的检索增强生成系统"""

    def __init__(self, config: Optional[RAGConfig | Dict[str, object]] = None):
        if isinstance(config, dict):
            config = RAGConfig(**config)
        self.config = config or RAGConfig()

        self._documents: Dict[str, Document] = {}
        self._lock = asyncio.Lock()
        self._index_by_source: Dict[str, set[str]] = defaultdict(set)
        self._index_by_tag: Dict[str, set[str]] = defaultdict(set)

    async def add_document(
        self,
        title: str,
        content: str,
        source: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> str:
        if not content or not content.strip():
            raise RAGError("文档内容不能为空")

        document = Document(
            title=title.strip(),
            content=content.strip(),
            source=source,
            metadata=metadata or {},
        )

        async with self._lock:
            if len(self._documents) >= self.config.max_documents:
                await self._evict_documents()

            if self.config.chunk_size and len(document.content) > self.config.chunk_size:
                chunks = self._chunk_document(document)
                for chunk in chunks:
                    self._store_document(chunk)
                return chunks[0].id

            self._store_document(document)

        return document.id

    async def add_documents(
        self,
        documents: Sequence[Tuple[str, str, str, Optional[Dict[str, object]]]],
    ) -> List[str]:
        ids: List[str] = []
        for title, content, source, metadata in documents:
            ids.append(await self.add_document(title, content, source, metadata))
        return ids

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        retrieval_limit = limit or self.config.retrieval_limit

        async with self._lock:
            documents = self._get_candidates(filters)

        scored_docs: List[Tuple[float, Document]] = []
        for doc in documents:
            score = self._similarity(query_lower, doc.content.lower())
            if score < self.config.similarity_threshold:
                continue
            rerank_boost = self._recency_boost(doc) if self.config.reranking_enabled else 0.0
            scored_docs.append((score + rerank_boost, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: max(1, retrieval_limit)]]

    async def search(self, query: str, **kwargs) -> List[Document]:
        """`retrieve` 的语义化别名"""
        return await self.retrieve(query, **kwargs)

    async def update_document(
        self,
        document_id: str,
        updates: Dict[str, object],
    ) -> bool:
        async with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"文档 {document_id} 不存在")

            doc = self._documents[document_id]
            original_source = doc.source
            original_tags = set(doc.tags)

            if "title" in updates:
                doc.title = str(updates["title"]).strip()
            if "content" in updates:
                content = str(updates["content"]).strip()
                if not content:
                    raise RAGError("文档内容不能为空")
                doc.content = content
            if "metadata" in updates and isinstance(updates["metadata"], dict):
                doc.metadata.update(updates["metadata"])
            if "tags" in updates and isinstance(updates["tags"], list):
                doc.tags = list({str(tag) for tag in updates["tags"]})
            if "source" in updates:
                doc.source = str(updates["source"])

            doc.updated_at = datetime.now()
            self._reindex_document(doc, original_source, original_tags)

        return True

    async def delete_document(self, document_id: str) -> bool:
        async with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"文档 {document_id} 不存在")
            doc = self._documents.pop(document_id)
            self._index_by_source[doc.source].discard(document_id)
            for tag in doc.tags:
                self._index_by_tag[tag].discard(document_id)
        return True

    async def rebuild_index(self) -> bool:
        async with self._lock:
            self._index_by_source.clear()
            self._index_by_tag.clear()
            for doc in self._documents.values():
                self._index_by_source[doc.source].add(doc.id)
                for tag in doc.tags:
                    self._index_by_tag[tag].add(doc.id)
                doc.updated_at = datetime.now()
        return True

    async def cleanup(self) -> int:
        async with self._lock:
            count = len(self._documents)
            self._documents.clear()
            self._index_by_source.clear()
            self._index_by_tag.clear()
        return count

    async def stats(self) -> Dict[str, object]:
        async with self._lock:
            total = len(self._documents)
            per_source = {source: len(ids) for source, ids in self._index_by_source.items()}
            per_tag = {tag: len(ids) for tag, ids in self._index_by_tag.items()}
        return {
            "total_documents": total,
            "max_documents": self.config.max_documents,
            "per_source": per_source,
            "per_tag": per_tag,
        }

    def _store_document(self, document: Document) -> None:
        self._documents[document.id] = document
        self._index_by_source[document.source].add(document.id)
        for tag in document.tags:
            self._index_by_tag[tag].add(document.id)

    def _get_candidates(self, filters: Optional[Dict[str, object]]) -> Iterable[Document]:
        if not filters:
            return list(self._documents.values())

        filtered_ids: Optional[set[str]] = None
        source = filters.get("source")
        if source:
            filtered_ids = set(self._index_by_source.get(str(source), set()))

        tag = filters.get("tag")
        if tag:
            tag_ids = set(self._index_by_tag.get(str(tag), set()))
            filtered_ids = tag_ids if filtered_ids is None else filtered_ids & tag_ids

        if filtered_ids is None:
            return [
                doc
                for doc in self._documents.values()
                if self._matches_metadata(doc, filters)
            ]

        return [
            self._documents[doc_id]
            for doc_id in filtered_ids
            if doc_id in self._documents and self._matches_metadata(self._documents[doc_id], filters)
        ]

    def _matches_metadata(self, document: Document, filters: Dict[str, object]) -> bool:
        metadata_filters = {
            key: value for key, value in filters.items() if key not in {"source", "tag"}
        }
        if not metadata_filters:
            return True

        return all(
            str(document.metadata.get(key)) == str(value)
            for key, value in metadata_filters.items()
        )

    def _chunk_document(self, document: Document) -> List[Document]:
        chunks: List[Document] = []
        chunk_size = max(self.config.chunk_size, 1)
        overlap = max(min(self.config.chunk_overlap, chunk_size - 1), 0)
        content = document.content

        start = 0
        index = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunk = Document(
                title=f"{document.title} (chunk {index + 1})",
                content=chunk_text,
                source=document.source,
                metadata={**document.metadata, "chunk_index": index},
                tags=document.tags.copy(),
            )
            chunks.append(chunk)
            if end == len(content):
                break
            start = end - overlap
            index += 1
        return chunks

    @staticmethod
    def _similarity(query: str, content: str) -> float:
        if query in content:
            return 1.0
        query_terms = set(query.split())
        content_terms = set(content.split())
        if not query_terms or not content_terms:
            return 0.0
        intersection = len(query_terms & content_terms)
        union = len(query_terms | content_terms)
        return intersection / union

    @staticmethod
    def _recency_boost(document: Document) -> float:
        days = (datetime.now() - document.updated_at).total_seconds() / 86400
        return math.exp(-max(days, 0) / 10) * 0.05

    async def _evict_documents(self) -> None:
        sorted_docs = sorted(
            self._documents.values(),
            key=lambda doc: doc.updated_at,
        )
        to_remove = sorted_docs[: max(1, len(sorted_docs) // 10)]
        for doc in to_remove:
            self._documents.pop(doc.id, None)
            self._index_by_source[doc.source].discard(doc.id)
            for tag in doc.tags:
                self._index_by_tag[tag].discard(doc.id)


__all__ = ["RAGSystem"]
