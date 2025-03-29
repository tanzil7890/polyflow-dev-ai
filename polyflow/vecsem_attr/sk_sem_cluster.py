from typing import Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize

# Import polyflow and its models
import polyflow
import polyflow.settings
from polyflow.models import SentenceVectorizer

def validate_clustering_feasibility(vectors, n_clusters):
    """
    Validate if clustering is feasible based on data distribution.
    
    Args:
        vectors: Data vectors to cluster
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple of (is_feasible, message)
    """
    # Check basic requirements
    if len(vectors) < n_clusters:
        return False, f"Cannot create {n_clusters} clusters from {len(vectors)} data points. Need at least as many data points as clusters."
    
    # Check for minimal size requirement
    min_points_per_cluster = 5
    if len(vectors) < n_clusters * min_points_per_cluster:
        return False, f"For stable clustering, having at least {min_points_per_cluster} points per cluster is recommended. " \
                      f"Current data has {len(vectors)} points for {n_clusters} clusters."
    
    # Check for degenerate data (all vectors too similar)
    if len(vectors) > 1:
        # Calculate pairwise cosine similarities to check data distribution
        # Take a sample if dataset is large
        sample_size = min(100, len(vectors))
        if sample_size < len(vectors):
            indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[indices]
        else:
            sample_vectors = vectors
            
        # Normalize vectors
        sample_vectors = normalize(sample_vectors)
        
        # Compute similarities
        similarities = np.dot(sample_vectors, sample_vectors.T)
        
        # Check if all vectors are too similar (would create degenerate clusters)
        mean_sim = np.mean(similarities) 
        if mean_sim > 0.95:  # High similarity threshold
            return False, "Data points are too similar to each other for effective clustering. " \
                         "Consider transforming or enriching your data to increase variance."
    
    return True, "Clustering is feasible with the current data."

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
        min_samples_per_cluster: int = 5,
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
            min_samples_per_cluster: Minimum number of samples required per cluster
            
        Returns:
            DataFrame with cluster assignments and optionally scores/centroids
        """
        if polyflow.settings.rm is None:
            raise ValueError(
                "Retrieval model not configured. Use polyflow.settings.configure(rm=...)"
            )

        # Data validation to prevent crashes
        total_samples = len(self._obj)
        min_required_samples = n_clusters * min_samples_per_cluster
        
        if total_samples < min_required_samples:
            raise ValueError(
                f"Insufficient data for clustering: {total_samples} samples provided, but at least "
                f"{min_required_samples} samples are required for {n_clusters} clusters "
                f"(minimum {min_samples_per_cluster} samples per cluster). "
                f"Either reduce the number of clusters or provide more data."
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

        # Additional validation to ensure vectors are valid
        if np.isnan(vectors).any() or np.isinf(vectors).any():
            # Clean up any invalid vectors
            vectors = np.nan_to_num(vectors)
            
        # If vectors are all zeros, it would cause problems
        if np.all(np.abs(vectors) < 1e-10):
            raise ValueError("Invalid vectors: all embeddings are near zero")
            
        # Check if clustering is feasible based on data distribution
        is_feasible, message = validate_clustering_feasibility(vectors, n_clusters)
        if not is_feasible:
            raise ValueError(message)

        # Perform clustering
        if method == "kmeans":
            try:
                clusters, scores, centroids = self._kmeans_cluster(
                    vectors, n_clusters, n_iter, verbose
                )
            except Exception as e:
                raise ValueError(f"K-means clustering failed: {str(e)}. Consider reducing the number of clusters or using hierarchical clustering.")
        elif method == "hierarchical":
            try:
                clusters, scores, centroids = self._hierarchical_cluster(
                    vectors, n_clusters, min_cluster_size, verbose
                )
            except Exception as e:
                raise ValueError(f"Hierarchical clustering failed: {str(e)}. Consider reducing the number of clusters or using k-means clustering.")
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
        
        # Safety check - FAISS has issues with very small datasets
        if vectors.shape[0] < n_clusters * 2:
            # Fall back to simpler k-means for very small datasets
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            labels = kmeans.fit_predict(vectors)
            centroids = kmeans.cluster_centers_
            
            # Calculate distances
            distances = np.zeros(len(vectors))
            for i, vec in enumerate(vectors):
                # Cosine similarity to centroid
                distances[i] = np.dot(vec, centroids[labels[i]])
                
            return labels, distances, centroids
        
        # Use FAISS for larger datasets
        try:
            kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose)
            kmeans.train(vectors)
            
            # Get cluster assignments and distances
            distances, labels = kmeans.index.search(vectors, 1)
            
            return labels.ravel(), distances.ravel(), kmeans.centroids
        except RuntimeError as e:
            # Handle common FAISS errors
            if "empty cluster" in str(e).lower():
                # Try again with more iterations and different seed
                kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter*2, verbose=verbose, seed=1234)
                kmeans.train(vectors)
                distances, labels = kmeans.index.search(vectors, 1)
                return labels.ravel(), distances.ravel(), kmeans.centroids
            else:
                raise

    def _hierarchical_cluster(
        self, vectors: np.ndarray, n_clusters: int, min_size: Optional[int], verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform hierarchical clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(vectors)
            
            # Calculate distances to cluster centers
            centroids = np.zeros((n_clusters, vectors.shape[1]))
            for i in range(n_clusters):
                cluster_vectors = vectors[labels == i]
                if len(cluster_vectors) > 0:
                    centroids[i] = cluster_vectors.mean(axis=0)
                else:
                    # Handle empty clusters
                    centroids[i] = np.zeros(vectors.shape[1])
            
            distances = np.zeros(len(vectors))
            for i, vec in enumerate(vectors):
                distances[i] = np.dot(vec, centroids[labels[i]])
                
            return labels, distances, centroids
            
        except Exception as e:
            # Fall back to a simpler method if AgglomerativeClustering fails
            if verbose:
                print(f"Hierarchical clustering failed, falling back to k-means: {str(e)}")
            return self._kmeans_cluster(vectors, n_clusters, 20, verbose)
