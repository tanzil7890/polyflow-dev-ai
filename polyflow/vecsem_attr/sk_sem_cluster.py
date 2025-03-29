from typing import Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize

# Import polyflow and its models
import polyflow
import polyflow.settings
from polyflow.models import SentenceVectorizer

@pd.api.extensions.register_dataframe_accessor("sk_sem_cluster")
class SemClusterDataframe:
    """DataFrame accessor for enhanced semantic clustering."""
    
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    @staticmethod
    def _validate(obj: pd.DataFrame):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        text_column: str,
        n_clusters: int,
        method: str = "kmeans",
        return_scores: bool = False,
        return_centroids: bool = False,
        n_iter: int = 20,
        min_cluster_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        """
        Perform semantic clustering on text data.
        
        Args:
            text_column: Column containing text to cluster
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans' or 'hierarchical')
            return_scores: Return similarity scores to centroids
            return_centroids: Return cluster centroids
            n_iter: Number of clustering iterations
            min_cluster_size: Minimum cluster size (for hierarchical)
            verbose: Print progress information
            
        Returns:
            DataFrame with cluster assignments and optionally scores/centroids
        """
        if polyflow.settings.rm is None:
            raise ValueError(
                "Retrieval model not configured. Use polyflow.settings.configure(rm=...)"
            )

        # Get or create embeddings
        if text_column not in self._obj.attrs.get("index_dirs", {}):
            self._obj = self._obj.vector_index(text_column, f"{text_column}_index")
        
        rm = polyflow.settings.rm
        index_dir = self._obj.attrs["index_dirs"][text_column]
        rm.load_index(index_dir)
        
        # Get vectors from index
        vectors = rm.get_vectors_from_index(index_dir, self._obj.index.tolist())
        vectors = normalize(vectors)

        # Perform clustering
        if method == "kmeans":
            clusters, scores, centroids = self._kmeans_cluster(
                vectors, n_clusters, n_iter, verbose
            )
        elif method == "hierarchical":
            clusters, scores, centroids = self._hierarchical_cluster(
                vectors, n_clusters, min_cluster_size, verbose
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Add results to dataframe
        result_df = self._obj.copy()
        result_df["cluster_id"] = clusters
        
        if return_scores:
            result_df["centroid_similarity"] = scores
            
        if return_centroids:
            return result_df, centroids
            
        return result_df

    def _kmeans_cluster(
        self, vectors: np.ndarray, n_clusters: int, n_iter: int, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform k-means clustering using FAISS."""
        d = vectors.shape[1]
        kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose)
        kmeans.train(vectors)
        
        # Get cluster assignments and distances
        distances, labels = kmeans.index.search(vectors, 1)
        
        return labels.ravel(), distances.ravel(), kmeans.centroids

    def _hierarchical_cluster(
        self, vectors: np.ndarray, n_clusters: int, min_size: Optional[int], verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform hierarchical clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(vectors)
        
        # Calculate distances to cluster centers
        centroids = np.zeros((n_clusters, vectors.shape[1]))
        for i in range(n_clusters):
            mask = labels == i
            centroids[i] = vectors[mask].mean(axis=0)
        
        distances = np.zeros(len(vectors))
        for i, vec in enumerate(vectors):
            distances[i] = np.dot(vec, centroids[labels[i]])
            
        return labels, distances, centroids
