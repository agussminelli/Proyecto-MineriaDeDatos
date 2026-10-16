import numpy as np
from sklearn.decomposition import TruncatedSVD

def create_hybrid_embeddings(sparse_tfidf_matrix, dense_embeddings, sparse_dimensions=50):
    """
    Merges semantic understanding with precise clinical keyword focus by concatenating 
    dense embeddings with reduced sparse TF-IDF vectors.
    
    Parameters:
    sparse_tfidf_matrix (scipy.sparse matrix): The output from the n-grams/TF-IDF step.
    dense_embeddings (numpy.ndarray): The dense vectors (e.g., from SentenceTransformers).
    sparse_dimensions (int): The number of dimensions to retain from the sparse matrix.
    
    Returns:
    numpy.ndarray: The concatenated hybrid feature matrix ready for DBSCAN.
    """
    
    # 1. Reduce the dimensionality of the sparse TF-IDF matrix 
    # TruncatedSVD is used because PCA does not support scipy sparse matrices directly
    svd_reducer = TruncatedSVD(n_components=sparse_dimensions, random_state=42)
    reduced_sparse_matrix = svd_reducer.fit_transform(sparse_tfidf_matrix)
    
    # 2. Concatenate the dense embeddings with the reduced TF-IDF vectors
    # np.hstack joins the arrays horizontally (combining the features column-wise)
    hybrid_feature_matrix = np.hstack((dense_embeddings, reduced_sparse_matrix))
    
    return hybrid_feature_matrix