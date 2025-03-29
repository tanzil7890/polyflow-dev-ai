import faiss
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

from polyflow.dtype_extensions import convert_to_base_data
from polyflow.models.retriever_faiss import FaissRetriever


class SentenceVectorizer(FaissRetriever):
    """
    Vector retrieval engine using sentence-transformers models for embedding generation.
    Built on top of FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 64,
        normalize_vectors: bool = True,
        device: Optional[str] = None,
        index_type: str = "Flat",
        similarity_metric: int = faiss.METRIC_INNER_PRODUCT,
    ):
        """
        Initialize the sentence transformer retriever.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Number of items to process at once
            normalize_vectors: Whether to normalize embeddings (recommended for cosine similarity)
            device: Device to run the model on (cpu, cuda, etc)
            index_type: FAISS index type (Flat, IVF, HNSW, etc)
            similarity_metric: Distance metric for similarity calculation
        """
        super().__init__(index_type, similarity_metric)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_vectors = normalize_vectors
        self.transformer = SentenceTransformer(model_name, device=device)

    def generate_embeddings(self, items: Union[pd.Series, List]) -> NDArray[np.float64]:
        """
        Convert text or other data into vector embeddings using the transformer model.
        
        Args:
            items: Data items to embed
            
        Returns:
            Array of embedding vectors
        """
        all_embeddings = []
        for i in tqdm(range(0, len(items), self.batch_size), desc="Generating embeddings"):
            batch = items[i : i + self.batch_size]
            processed_batch = convert_to_base_data(batch)
            
            # Generate embeddings with the transformer model
            embedding_tensor = self.transformer.encode(
                processed_batch, 
                convert_to_tensor=True, 
                normalize_embeddings=self.normalize_vectors, 
                show_progress_bar=False
            )
            
            # Convert to numpy arrays and store
            assert isinstance(embedding_tensor, torch.Tensor)
            batch_embeddings = embedding_tensor.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)
        
    # Alias for compatibility with base class
    def _embed(self, docs: pd.Series) -> NDArray[np.float64]:
        return self.generate_embeddings(docs)
