import os
import pickle
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from polyflow.models.retriever_base import RetrieverEngine
from polyflow.types import RMOutput


class FaissRetriever(RetrieverEngine):
    """
    Base retriever implementation using FAISS for vector indexing and search.
    This class handles index creation, storage, loading, and searching.
    Derived classes must implement embedding generation.
    """
    
    def __init__(self, index_config: str = "Flat", distance_metric=faiss.METRIC_INNER_PRODUCT):
        """
        Initialize the FAISS-based retriever.
        
        Args:
            index_config: FAISS index factory configuration string
            distance_metric: Similarity metric to use (inner product, L2, etc.)
        """
        super().__init__()
        self.index_config = index_config
        self.distance_metric = distance_metric
        self.index_location: Optional[str] = None
        self.vector_index: Optional[faiss.Index] = None
        self.stored_vectors: Optional[NDArray[np.float64]] = None

    def create_index(self, documents: pd.Series, storage_directory: str, **index_params: Dict[str, Any]) -> None:
        """
        Create and save a FAISS index from document embeddings.
        
        Args:
            documents: Documents to index
            storage_directory: Where to store the index
            index_params: Additional indexing parameters
        """
        # Generate vector embeddings for documents
        vectors = self._embed(documents)
        
        # Create FAISS index of appropriate dimension and type
        vector_dimension = vectors.shape[1]
        self.vector_index = faiss.index_factory(vector_dimension, self.index_config, self.distance_metric)
        
        # Add vectors to index
        self.vector_index.add(vectors)
        self.index_location = storage_directory

        # Save index and vectors to disk
        os.makedirs(storage_directory, exist_ok=True)
        
        # Save raw vectors for later use
        with open(f"{storage_directory}/vectors.pkl", "wb") as vector_file:
            pickle.dump(vectors, vector_file)
            
        # Save FAISS index
        faiss.write_index(self.vector_index, f"{storage_directory}/faiss.index")

    def initialize_from_index(self, storage_directory: str) -> None:
        """
        Load a previously created index from disk.
        
        Args:
            storage_directory: Directory containing the index
        """
        self.index_location = storage_directory
        
        # Load the FAISS index
        self.vector_index = faiss.read_index(f"{storage_directory}/faiss.index")
        
        # Load the raw vectors
        with open(f"{storage_directory}/vectors.pkl", "rb") as vector_file:
            self.stored_vectors = pickle.load(vector_file)

    def retrieve_vectors(self, storage_directory: str, vector_ids: List[int]) -> NDArray[np.float64]:
        """
        Get specific vectors from a stored index by their IDs.
        
        Args:
            storage_directory: Location of the index
            vector_ids: IDs of vectors to retrieve
            
        Returns:
            Array of vectors corresponding to the IDs
        """
        # Load vectors from the specified directory
        with open(f"{storage_directory}/vectors.pkl", "rb") as vector_file:
            all_vectors: NDArray[np.float64] = pickle.load(vector_file)
            
        # Return only the requested vectors
        return all_vectors[vector_ids]

    def search(
        self, 
        query: Union[pd.Series, str, Image.Image, List, NDArray[np.float64]], 
        results_count: int, 
        **search_params: Dict[str, Any]
    ) -> RMOutput:
        """
        Search the index for items most similar to the query.
        
        Args:
            query: Search query (can be text, image, or vector)
            results_count: Number of results to return
            search_params: Additional search parameters
            
        Returns:
            Search results with distances and indices
        """
        # Handle single-item queries
        if isinstance(query, str) or isinstance(query, Image.Image):
            query = [query]

        # Convert query to vector embeddings if needed
        if not isinstance(query, np.ndarray):
            query_vectors = self._embed(query)
        else:
            query_vectors = np.asarray(query, dtype=np.float32)

        # Ensure index is loaded
        if self.vector_index is None:
            raise ValueError("Index not loaded. Call initialize_from_index() first.")

        # Perform search
        distances, indices = self.vector_index.search(query_vectors, results_count)
        return RMOutput(distances=distances, indices=indices)

    @abstractmethod
    def _embed(self, docs: Union[pd.Series, List]) -> NDArray[np.float64]:
        """
        Convert documents to vector embeddings.
        
        Args:
            docs: Documents to embed
            
        Returns:
            Document embedding vectors
        """
        pass
