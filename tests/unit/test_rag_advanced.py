"""RAG系统详细单元测试"""
import pytest
from src.rag import RAGSystem

class TestRAGSystem:
    """RAG系统测试类"""

    @pytest.fixture
    def rag_system(self):
        return RAGSystem()

    @pytest.mark.unit
    def test_add_document_basic(self, rag_system):
        """测试基本添加文档"""
        title = "Python教程"
        content = "Python是一种编程语言"
        doc_id = rag_system.add_document(title, content)
        assert doc_id is not None
        assert len(rag_system.documents) == 1
        print("✅ 基本文档添加测试通过")

    @pytest.mark.unit
    def test_retrieve_document(self, rag_system):
        """测试文档检索"""
        title = "Python教程"
        content = "Python是一种编程语言"
        doc_id = rag_system.add_document(title, content)
        results = rag_system.retrieve("Python")
        assert len(results) >= 1
        print("✅ 文档检索测试通过")

if __name__ == "__main__":
    pytest.main([__file__])
