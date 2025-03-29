from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from polyflow.types import RMOutput


class RetrieverEngine(ABC):
    """
    Abstract base class for all vector retrieval models.
    Provides interface for indexing, loading, and retrieving vectors.
    """

    def __init__(self) -> None:
        """Initialize the retriever engine."""
        self.index_location: Optional[str] = None

    @abstractmethod
    def create_index(self, documents: pd.Series, storage_directory: str, **index_params: Dict[str, Any]) -> None:
        """
        Process documents and create a searchable vector index.

        Args:
            documents: Series containing the documents to index
            storage_directory: Location to store the index
            index_params: Additional parameters for index construction
        """
        pass

    @abstractmethod
    def initialize_from_index(self, storage_directory: str) -> None:
        """
        Load an existing index into memory for searching.

        Args:
            storage_directory: Location where the index is stored
        """
        pass

    @abstractmethod
    def retrieve_vectors(self, storage_directory: str, vector_ids: List[int]) -> NDArray[np.float64]:
        """
        Retrieve specific vectors from an index by their IDs.

        Args:
            storage_directory: Directory containing the index
            vector_ids: List of vector IDs to retrieve

        Returns:
            Array of vectors corresponding to the requested IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Union[pd.Series, str, Image.Image, List, NDArray[np.float64]],
        results_count: int,
        **search_params: Dict[str, Any],
    ) -> RMOutput:
        """
        Search the index for the most similar items to the query.

        Args:
            query: Search query (text, image, or vector)
            results_count: Number of results to return
            search_params: Additional parameters to customize the search

        Returns:
            Object containing search results (distances and indices)
        """
        pass
        
    # Alias methods to maintain compatibility with existing code
    def index(self, docs: pd.Series, index_dir: str, **kwargs: Dict[str, Any]) -> None:
        return self.create_index(docs, index_dir, **kwargs)
        
    def load_index(self, index_dir: str) -> None:
        return self.initialize_from_index(index_dir)
        
    def get_vectors_from_index(self, index_dir: str, ids: List[int]) -> NDArray[np.float64]:
        return self.retrieve_vectors(index_dir, ids)
        
    def __call__(
        self,
        queries: Union[pd.Series, str, Image.Image, List, NDArray[np.float64]],
        K: int,
        **kwargs: Dict[str, Any],
    ) -> RMOutput:
        return self.search(queries, K, **kwargs)

# Define RM as alias for RetrieverEngine for backward compatibility
RM = RetrieverEngine
