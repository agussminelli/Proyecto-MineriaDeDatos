
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_silhouette(model, eps_values, min_samples_values):
    """Evaluate Silhouette using all clustered points."""

    results = []

    # Use the distance matrix already computed by DBSCAN
    D = model.distance_matrix_

    if D is None:
        raise ValueError("Fit DBSCAN with precompute=True first.")

    for eps in eps_values:
        for min_samples in min_samples_values:

            model.eps = eps
            model.min_samples = min_samples

            labels = model.fit_predict()

            # Exclude noise points
            valid_indices = np.flatnonzero(labels != -1)
            valid_labels = labels[valid_indices]

            n_clusters = len(np.unique(valid_labels))
            n_noise = np.sum(labels == -1)

            # Proportion of clustered points
            coverage = len(valid_indices) / len(labels)

            silhouette = np.nan
            adjusted_silhouette = np.nan

            # Silhouette requires at least two clusters
            if 2 <= n_clusters < len(valid_indices):

                # Distance matrix for all clustered points
                D_valid = D[np.ix_(valid_indices, valid_indices)]

                silhouette = silhouette_score(
                    D_valid,
                    valid_labels,
                    metric="precomputed"
                )

                # Penalize configurations with high noise
                adjusted_silhouette = silhouette * coverage

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_fraction": n_noise / len(labels),
                "silhouette": silhouette,
                "adjusted_silhouette": adjusted_silhouette
            })

    return pd.DataFrame(results)
