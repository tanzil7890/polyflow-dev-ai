from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from polyflow.types import RerankerOutput


class RerankerEngine(ABC):
    """
    Abstract base class for document reranking engines.
    Rerankers take a query and document list and reorder them 
    by relevance to the query.
    """

    def __init__(self) -> None:
        """Initialize the reranker with usage tracking."""
        self.rerank_operation_count: int = 0
        self.caching_status: bool = False

    @abstractmethod
    def rerank_documents(self, query: str, documents: List[str], result_count: int) -> RerankerOutput:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query text
            documents: List of documents to rerank
            result_count: Number of top results to return

        Returns:
            RerankerOutput containing the indices of reranked documents
        """
        pass

    def activate_caching(self) -> None:
        """Enable caching for reranking operations."""
        self.caching_status = True

    def deactivate_caching(self) -> None:
        """Disable caching for reranking operations."""
        self.caching_status = False

    def get_performance_metrics(self) -> Dict:
        """
        Get performance and usage statistics for the reranker.

        Returns:
            Dictionary with usage metrics
        """
        return {
            "total_rerank_operations": self.rerank_operation_count,
            "caching_enabled": self.caching_status
        }
        
    # Alias methods for backward compatibility
    def __call__(self, query: str, docs: List[str], K: int) -> RerankerOutput:
        return self.rerank_documents(query, docs, K)
        
    def enable_cache(self) -> None:
        return self.activate_caching()
        
    def disable_cache(self) -> None:
        return self.deactivate_caching()
        
    def get_usage_stats(self) -> Dict:
        return self.get_performance_metrics()
