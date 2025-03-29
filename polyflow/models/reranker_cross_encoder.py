from typing import List, Optional

from sentence_transformers import CrossEncoder

from polyflow.models.reranker_base import RerankerEngine
from polyflow.types import RerankerOutput


class EncoderReranker(RerankerEngine):
    """
    Reranker implementation using sentence-transformers CrossEncoder models.
    CrossEncoders directly score query-document pairs for relevance ranking.
    
    Args:
        model_name: Name of the cross-encoder model to use
        compute_device: Device to run the model on (cpu, cuda, etc)
        batch_processing_size: Maximum batch size for processing
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        compute_device: Optional[str] = None,
        batch_processing_size: int = 64,
    ):
        """Initialize the cross-encoder reranker with the specified model."""
        super().__init__()
        self.batch_processing_size: int = batch_processing_size
        self.encoder = CrossEncoder(model_name, device=compute_device)  # type: ignore # CrossEncoder has wrong type stubs

    def rerank_documents(self, query: str, documents: List[str], result_count: int) -> RerankerOutput:
        """
        Rerank documents using the cross-encoder model.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            result_count: Number of top results to return
            
        Returns:
            Reranker output containing indices of the top documents
        """
        # Track usage for metrics
        self.rerank_operation_count += 1
        
        # Use the cross-encoder to rank documents
        ranked_results = self.encoder.rank(
            query, 
            documents, 
            top_k=result_count, 
            batch_size=self.batch_processing_size, 
            show_progress_bar=False
        )
        
        # Extract the indices of the top documents
        top_indices = [int(result["corpus_id"]) for result in ranked_results]
        
        return RerankerOutput(indices=top_indices)
