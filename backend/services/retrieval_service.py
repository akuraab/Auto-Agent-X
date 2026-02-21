from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Simple Ensemble Retriever implementation
# This is a fallback implementation because LangChain's EnsembleRetriever is missing in some versions
class EnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # This sync method is required by BaseRetriever but not used in async context
        # We'll implement a simple version that runs sync retrievers if possible
        all_docs = []
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query, run_manager=run_manager)
                all_docs.extend(docs)
            except Exception:
                pass
        return self._rerank(all_docs)
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Execute all retrievers concurrently
        tasks = [
            retriever.ainvoke(query, run_manager=run_manager) 
            for retriever in self.retrievers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_docs = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                # We could use weights here to adjust scores if needed
                # weight = self.weights[i] if i < len(self.weights) else 1.0
                all_docs.extend(res)
                
        return self._rerank(all_docs)

    def _rerank(self, docs: List[Document]) -> List[Document]:
        # Simple deduplication based on page_content
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs

# Custom Mock Retriever for Vector Search
class MockVectorRetriever(BaseRetriever):
    k: int = 4
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Mock results based on query
        return [
            Document(
                page_content=f"Vector result for {query}: This is relevant content chunk {i}",
                metadata={"source": f"doc_{i}.py", "type": "vector", "score": 0.9 - (i * 0.1)}
            )
            for i in range(self.k)
        ]

# Custom Mock Retriever for Keyword Search
class MockKeywordRetriever(BaseRetriever):
    k: int = 4
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [
            Document(
                page_content=f"Keyword result for {query}: Function definition and usage {i}",
                metadata={"source": f"utils_{i}.py", "type": "keyword", "score": 0.8 - (i * 0.1)}
            )
            for i in range(self.k)
        ]

class RetrievalService:
    """Retrieval Service - LangChain Implementation"""

    def __init__(self):
        # Initialize retrievers
        self.vector_retriever = MockVectorRetriever(k=5)
        self.keyword_retriever = MockKeywordRetriever(k=5)
        
        # Initialize Ensemble Retriever (Hybrid Search)
        # Weights: 0.7 for vector, 0.3 for keyword
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.keyword_retriever],
            weights=[0.7, 0.3]
        )

    async def retrieve(self, query: str) -> List[Document]:
        """
        Execute hybrid retrieval using LangChain EnsembleRetriever
        """
        # invoke is sync, ainvoke is async
        return await self.ensemble_retriever.ainvoke(query)
