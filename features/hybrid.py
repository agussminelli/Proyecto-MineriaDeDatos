import numpy as np
from sklearn.decomposition import TruncatedSVD


def create_hybrid_embeddings(
    sparse_tfidf_matrix, dense_embeddings, sparse_dimensions=50
):
    """Concatenate dense vectors and a fitted SVD reduction of TF-IDF.

    Parameters
    ----------
    sparse_tfidf_matrix : sparse matrix of shape (n_samples, n_terms)
        Training TF-IDF features.
    dense_embeddings : ndarray of shape (n_samples, n_dense_features)
        Dense vectors for exactly the same records in the same row order.
    sparse_dimensions : int
        Requested number of SVD components; random_state is fixed at 42.

    Returns
    -------
    ndarray
        Dense vectors followed by the reduced TF-IDF coordinates.

    Notes
    -----
    This exploratory helper neither normalizes nor weights the two blocks,
    and does not return the fitted SVD. The final hybrid representation in
    run_experiment.build_representation uses explicit normalization and
    square-root weights; do not treat the two constructions as equivalent.
    """

    # Reduce the dimensionality of the sparse TF-IDF matrix
    # TruncatedSVD is used because PCA does not support scipy sparse matrices directly
    svd_reducer = TruncatedSVD(n_components=sparse_dimensions, random_state=42)
    reduced_sparse_matrix = svd_reducer.fit_transform(sparse_tfidf_matrix)

    # Concatenate the dense embeddings with the reduced TF-IDF vectors
    hybrid_feature_matrix = np.hstack(
        (dense_embeddings, reduced_sparse_matrix)
    )

    return hybrid_feature_matrix
