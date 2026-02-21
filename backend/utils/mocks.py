from typing import List, Dict, Any, Optional

class MockVectorStore:
    async def similarity_search(self, embedding: List[float], top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
        return [
            {
                "content": "This is a mock document content from vector store.",
                "metadata": {"source": "mock_doc_1", "doc_id": "1"},
                "score": 0.9,
                "embedding": [0.1] * 10
            }
        ]

class MockKeywordRetriever:
    async def search(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
        return [
             {
                "content": "This is a mock document content from keyword search.",
                "metadata": {"source": "mock_doc_2", "doc_id": "2"},
                "score": 0.8
            }
        ]

class MockReranker:
    async def rerank(self, pairs: List[tuple]) -> List[float]:
        return [0.95] * len(pairs)

class MockEmbeddingService:
    async def embed_query(self, query: str) -> List[float]:
        return [0.1] * 768

class MockTemplateStore:
    async def get(self, template_id: str):
        # Return None to trigger default template
        return None
