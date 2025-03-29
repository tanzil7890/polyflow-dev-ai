from typing import List, Dict, Union, Optional, Any
import faiss
import numpy as np
import pandas as pd
from litellm import embedding
from litellm.types.utils import EmbeddingResponse
from numpy.typing import NDArray
from tqdm import tqdm

from polyflow.dtype_extensions import convert_to_base_data
from polyflow.models.retriever_faiss import FaissRetriever
from polyflow.types import EmbeddingConfig, EmbeddingResult, RMOutput


class EmbeddingRetriever(FaissRetriever):
    """
    Vector retrieval engine using LLM-based embeddings.
    Uses LiteLLM to access embeddings from various providers.
    
    This class handles text embedding generation, indexing, and similarity search
    with support for batched processing and progress tracking.
    """
    
    def __init__(
        self,
        model_identifier: str = "text-embedding-3-small",
        processing_batch_size: int = 64,
        index_structure: str = "Flat",
        similarity_function: int = faiss.METRIC_INNER_PRODUCT,
        embedding_options: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedding-based retriever.
        
        Args:
            model_identifier: Name of the embedding model to use
            processing_batch_size: Number of items to process in each batch
            index_structure: FAISS index type configuration
            similarity_function: Metric for similarity computation
            embedding_options: Additional configuration options
        """
        super().__init__(index_structure, similarity_function)
        self.embedding_model_name = model_identifier
        self.processing_batch_size = processing_batch_size
        self.vector_dimension: Optional[int] = None
        self.options = embedding_options or EmbeddingConfig()
        
    def _embed(self, documents: Union[pd.Series, List[str]]) -> NDArray[np.float64]:
        """
        Transform documents into embedding vectors.
        
        Args:
            documents: Text documents to embed
            
        Returns:
            Array of document embedding vectors
        """
        return self.create_embeddings(documents)
        
    def create_embeddings(
        self, 
        content: Union[pd.Series, List[str]]
    ) -> NDArray[np.float64]:
        """
        Generate embeddings for a collection of texts.
        
        Args:
            content: Text content to embed
            
        Returns:
            Array of embedding vectors
        """
        embedding_arrays = []
        
        # Process content in batches with progress tracking
        for batch_start in tqdm(
            range(0, len(content), self.processing_batch_size),
            desc="Generating embeddings",
            disable=not self.options.show_progress
        ):
            # Extract and process current batch
            batch_end = batch_start + self.processing_batch_size
            current_batch = content[batch_start:batch_end]
            batch_vectors = self.process_content_batch(current_batch)
            embedding_arrays.append(batch_vectors)
            
            # Store dimension information from first batch
            if self.vector_dimension is None and batch_vectors.size > 0:
                self.vector_dimension = batch_vectors.shape[1]
        
        # Combine all batch results
        return np.vstack(embedding_arrays)
    
    def process_content_batch(
        self, 
        batch: Union[pd.Series, List[str]]
    ) -> NDArray[np.float64]:
        """
        Process a single batch of content to generate embeddings.
        
        Args:
            batch: Batch of text content
            
        Returns:
            Embedding vectors for the batch
        """
        try:
            # Convert to appropriate format
            normalized_batch = convert_to_base_data(batch)
            
            # Generate embeddings via LiteLLM
            embedding_response: EmbeddingResponse = embedding(
                model=self.embedding_model_name, 
                input=normalized_batch
            )
            
            # Extract embedding vectors
            vectors = np.array([
                item["embedding"] for item in embedding_response.data
            ])
            
            return vectors
            
        except Exception as e:
            # Handle errors according to configuration
            if self.options.raise_errors:
                raise
            print(f"Warning: Embedding generation failed - {str(e)}")
            return np.array([])
    
    def find_similar_items(
        self,
        query_text: str,
        corpus: Union[pd.Series, List[str]],
        result_count: int = 5
    ) -> EmbeddingResult:
        """
        Find items most similar to a query text.
        
        Args:
            query_text: Text to compare against corpus
            corpus: Collection of texts to search in
            result_count: Number of similar items to return
            
        Returns:
            Result containing similarities, indices and matched texts
        """
        # Generate embeddings
        query_vector = self.create_embeddings([query_text])
        corpus_vectors = self.create_embeddings(corpus)
        
        # Perform vector similarity search
        similarity_scores, match_indices = self.vector_similarity_search(
            query_vector, 
            corpus_vectors,
            k=result_count
        )
        
        # Prepare results
        return EmbeddingResult(
            similarities=similarity_scores[0],
            indices=match_indices[0],
            texts=[corpus[i] for i in match_indices[0]]
        )
        
    def vector_similarity_search(
        self, 
        query_vectors: NDArray[np.float64],
        corpus_vectors: NDArray[np.float64],
        k: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Perform vector similarity search using FAISS.
        
        Args:
            query_vectors: Query embedding vectors
            corpus_vectors: Corpus embedding vectors
            k: Number of results to return
            
        Returns:
            Tuple of similarity scores and indices
        """
        # Create a temporary index for the search
        dimension = corpus_vectors.shape[1]
        temp_index = faiss.index_factory(dimension, self.index_config, self.distance_metric)
        
        # Add corpus vectors to index
        temp_index.add(corpus_vectors)
        
        # Search for nearest neighbors
        scores, indices = temp_index.search(query_vectors, k)
        
        return scores, indices
    
    def search(
        self,
        query: Union[pd.Series, str, List, NDArray[np.float64]],
        results_count: int,
        **search_params: Dict[str, Any],
    ) -> RMOutput:
        """
        Search for similar items to the query.
        
        Args:
            query: Search query (text or vector)
            results_count: Number of results to return
            search_params: Additional search parameters
            
        Returns:
            Search results with distances and indices
        """
        # If we have an existing index, use the parent class search
        if self.vector_index is not None:
            return super().search(query, results_count, **search_params)
            
        # Handle direct search without a persistent index
        if isinstance(query, str):
            query = [query]
            
        if not isinstance(query, np.ndarray):
            query_vectors = self.create_embeddings(query)
        else:
            query_vectors = query
            
        # If we have text content in the search params, use it
        corpus = search_params.get("corpus")
        if corpus is not None:
            corpus_vectors = self.create_embeddings(corpus)
            
            # Perform search
            distances, indices = self.vector_similarity_search(
                query_vectors, corpus_vectors, results_count
            )
            
            return RMOutput(distances=distances, indices=indices)
            
        raise ValueError("No index loaded and no corpus provided in search_params")