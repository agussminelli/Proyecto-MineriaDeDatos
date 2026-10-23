"""Combine dense embeddings and normalized LSA features."""

import numpy as np
from sklearn.preprocessing import normalize


def create_hybrid_embeddings(
    dense_embeddings,
    lsa_embeddings,
    alpha=0.5,
):
    """Combine aligned dense and normalized LSA representations.

    Both inputs must contain the same records in the same row order.
    LSA rows must already be L2-normalized, as returned by lsa().

    alpha weights the LSA block; 1 - alpha weights the dense block.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    if len(dense_embeddings) != len(lsa_embeddings):
        raise ValueError("Dense and LSA row counts must match")

    return np.hstack((
        np.sqrt(1 - alpha)
        * normalize(dense_embeddings, norm="l2"),
        np.sqrt(alpha) * lsa_embeddings,
    ))
