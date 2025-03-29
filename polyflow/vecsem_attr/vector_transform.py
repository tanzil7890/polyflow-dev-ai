from typing import Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray

import polyflow
from polyflow.models.rm import RM
from polyflow.utils import compute_cosine_similarity

@pd.api.extensions.register_dataframe_accessor("vector_transform")
class VectorTransformDataframe:
    """DataFrame accessor for vector-based transformations."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        column: str,
        transform_query: str,
        K: int = 5,
        threshold: float = 0.7,
        suffix: str = "_transformed",
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """Transform column values using vector similarity.
        
        Args:
            column: Column to transform
            transform_query: Query to guide the transformation
            K: Number of similar vectors to consider
            threshold: Similarity threshold for filtering
            suffix: Suffix for the transformed column
            return_scores: Whether to return similarity scores
            
        Returns:
            pd.DataFrame: DataFrame with transformed column
        """
        if polyflow.settings.rm is None:
            raise ValueError(
                "The retrieval model must be configured using polyflow.settings.configure()"
            )

        rm = polyflow.settings.rm
        if not isinstance(rm, RM):
            raise ValueError("Invalid retrieval model type")

        # Get vector embeddings
        if column in self._obj.attrs.get("index_dirs", {}):
            index_dir = self._obj.attrs["index_dirs"][column]
            rm.load_index(index_dir)
            vectors = rm.get_vectors_from_index(index_dir, list(self._obj.index))
        else:
            vectors = rm._embed(self._obj[column])

        # Get query vector using _embed
        query_vector = rm._embed([transform_query])
        
        # Compute similarities
        similarities = compute_cosine_similarity(vectors, query_vector).flatten()
        
        # Apply threshold
        mask = similarities >= threshold
        
        # Create transformed DataFrame
        transformed = self._obj.copy()
        
        # Handle transformed column
        transformed[f"{column}{suffix}"] = pd.Series(
            np.where(mask, self._obj[column], None),
            index=self._obj.index,
            dtype="string[python]"  # Changed dtype to handle NA properly
        )
        
        # Handle scores column
        if return_scores:
            scores = np.where(mask, similarities, np.nan)  # Use np.nan instead of pd.NA
            transformed[f"{column}_score"] = pd.Series(
                scores,
                index=self._obj.index,
                dtype="float64"
            ).round(2)
            
        return transformed
