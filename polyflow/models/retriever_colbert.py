import pickle
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from polyflow.models.retriever_base import RetrieverEngine
from polyflow.types import RMOutput

# Flag to track if ColBERT is available
COLBERT_AVAILABLE = False

try:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
    COLBERT_AVAILABLE = True
except ImportError:
    warnings.warn(
        "ColBERT is not installed. To use ColBERTEngine, install it with:\n"
        "pip install colbert-ai\n"
        "or:\n"
        "pip install polyflow[colbert]"
    )

class ColBERTEngine(RetrieverEngine):
    """
    Advanced retrieval engine based on ColBERT v2 late-interaction model.
    
    This retriever provides state-of-the-art dense retrieval capabilities
    with token-level interactions between queries and documents.
    
    Requires the colbert-ai package to be installed.
    """
    
    def __init__(
        self,
        document_max_length: int = 300,
        quantization_bits: int = 2,
        model_checkpoint: str = "colbert-ir/colbertv2.0"
    ) -> None:
        """
        Initialize the ColBERT retrieval engine.
        
        Args:
            document_max_length: Maximum document length in tokens
            quantization_bits: Number of bits for vector quantization
            model_checkpoint: Path or name of the model checkpoint
        """
        super().__init__()
        
        if not COLBERT_AVAILABLE:
            raise ImportError(
                "ColBERT is required but not installed. "
                "Install it with: pip install colbert-ai"
            )
            
        self.document_collection: Optional[List[str]] = None
        self.colbert_parameters = {
            "doc_maxlen": document_max_length,
            "nbits": quantization_bits
        }
        self.model_checkpoint = model_checkpoint
        self.index_location: Optional[str] = None

    def create_index(
        self, 
        documents: pd.Series, 
        storage_directory: str, 
        **index_params: Dict[str, Any]
    ) -> None:
        """
        Create a ColBERT index from documents.
        
        Args:
            documents: Collection of documents to index
            storage_directory: Directory to store the index
            index_params: Additional indexing parameters
        """
        document_list = documents.tolist()
        combined_params = {**self.colbert_parameters, **index_params}

        # Initialize ColBERT environment
        with Run().context(RunConfig(nranks=1, experiment="polyflow")):
            # Configure ColBERT indexer
            config = ColBERTConfig(
                doc_maxlen=combined_params["doc_maxlen"],
                nbits=combined_params["nbits"],
                kmeans_niters=4
            )
            
            # Create and run indexer
            indexer = Indexer(checkpoint=self.model_checkpoint, config=config)
            indexer.index(
                name=f"{storage_directory}/index",
                collection=document_list,
                overwrite=True
            )

        # Store document collection for retrieval
        with open(f"experiments/polyflow/indexes/{storage_directory}/index/docs", "wb") as file:
            pickle.dump(documents, file)

        # Update instance state
        self.document_collection = documents
        self.index_location = storage_directory

    def initialize_from_index(self, storage_directory: str) -> None:
        """
        Load a previously created ColBERT index.
        
        Args:
            storage_directory: Directory containing the index
        """
        self.index_location = storage_directory
        
        # Load document collection
        with open(f"experiments/polyflow/indexes/{storage_directory}/index/docs", "rb") as file:
            self.document_collection = pickle.load(file)

    def retrieve_vectors(
        self, 
        storage_directory: str, 
        vector_ids: List[int]
    ) -> NDArray[np.float64]:
        """
        Method not supported by ColBERT's architecture.
        
        ColBERT uses a late-interaction approach that doesn't expose document vectors.
        
        Raises:
            NotImplementedError: Always raises this exception
        """
        raise NotImplementedError(
            "retrieve_vectors is not supported by ColBERTEngine due to its late-interaction architecture"
        )

    def search(
        self,
        query: Union[str, Image.Image, List, NDArray[np.float64]],
        results_count: int,
        **search_params: Dict[str, Any],
    ) -> RMOutput:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text or list of queries
            results_count: Number of results to return
            search_params: Additional search parameters
            
        Returns:
            Search results with distances and indices
        """
        # Handle single query case
        if isinstance(query, str):
            query = [query]

        # Initialize ColBERT searcher in the appropriate environment
        with Run().context(RunConfig(experiment="polyflow")):
            searcher = Searcher(
                index=f"{self.index_location}/index",
                collection=self.document_collection
            )

        # Prepare queries in the format expected by ColBERT
        assert isinstance(query, list)
        query_mapping = {i: q for i, q in enumerate(query)}
        
        # Execute search
        search_results = searcher.search_all(query_mapping, k=results_count).todict()

        # Extract and format results
        result_indices = [
            [result[0] for result in search_results[qid]] 
            for qid in search_results.keys()
        ]
        similarity_scores = [
            [result[2] for result in search_results[qid]] 
            for qid in search_results.keys()
        ]

        return RMOutput(distances=similarity_scores, indices=result_indices)
