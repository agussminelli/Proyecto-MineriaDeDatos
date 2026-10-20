#DBVC implemented for python in hdbscan
#Silhouette from sklearn, corrected with noise after SWC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan.validity as dbcv_module
from scipy.special import logsumexp
from sklearn.metrics import silhouette_score, adjusted_rand_score

def stable_all_points_core_distance(distance_matrix, d=2.0):
    """Compute DBCV core distances without inverse-power overflow.

    Preserves hdbscan's convention of ignoring zero distances.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distances must form a square matrix")
    if not np.isfinite(D).all() or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if not np.isfinite(d) or d <= 0:
        raise ValueError("d must be finite and positive")

    n = len(D)
    if n < 2:
        return np.zeros(n, dtype=float)

    positive = D > 0

    # log(distance ** -d) = -d * log(distance)
    log_terms = np.full(D.shape, -np.inf)
    log_terms[positive] = -d * np.log(D[positive])

    log_sums = logsumexp(log_terms, axis=1)

    result = np.zeros(n, dtype=float)
    has_positive_distance = positive.any(axis=1)

    result[has_positive_distance] = np.exp(
        (
            np.log(n - 1)
            - log_sums[has_positive_distance]
        ) / d
    )

    return result

dbcv_module.all_points_core_distance = (
    stable_all_points_core_distance
)

def prepare_dbcv_labels(labels):
    labels = np.asarray(labels, dtype=int)

    if np.any(labels == 0):
        raise ValueError("Clustering still contains unassigned points")

    converted = np.full(labels.shape, -1, dtype=int)
    cluster_ids = np.unique(labels[labels > 0])

    for new_id, original_id in enumerate(cluster_ids):
        converted[labels == original_id] = new_id

    return converted

def evaluate_parameters(model, eps_values, min_samples_values, y_true=None):
    """Evaluate Silhouette and DBCV for each parameter combination.
    ARI for Comparison, if GT is provided.

    Silhouette excludes noise; DBCV includes noise.
    Restores the model's original parameters and labels afterward.
    """
    if y_true is not None:
        y_true = np.asarray(y_true)

        if y_true.ndim != 1 or len(y_true) != len(model.X_):
            raise ValueError("y_true must contain one outcome per input row")

        # Exclude missing outcomes from external evaluation.
        labelled = pd.notna(y_true)
        labelled &= ~pd.Series(y_true).astype(str).str.strip().str.lower().isin(
            ["", "na", "nan"]
        ).to_numpy()

    model._ensure_distance_matrix()
    D = np.asarray(model.distance_matrix_, dtype=np.float64)

    results = []

    original_eps = model.eps
    original_min_samples = model.min_samples
    original_labels = (
        model.labels_.copy() if hasattr(model, "labels_") else None
    )

    try:
        for eps in eps_values:
            for min_samples in min_samples_values:
                model.eps = float(eps)
                model.min_samples = int(min_samples)

                labels = model.fit_predict()
                if np.any(labels == 0):
                    raise ValueError("Some points remain unassigned")
                ari_all = np.nan
                ari_clustered = np.nan

                if y_true is not None:
                    if labelled.sum() >= 2:
                        ari_all = adjusted_rand_score(
                            y_true[labelled],
                            labels[labelled],
                        )

                    selected = labelled & (labels > 0)

                    if selected.sum() >= 2:
                        ari_clustered = adjusted_rand_score(
                            y_true[selected],
                            labels[selected],
                        )

                # Your DBSCAN uses positive IDs for clusters.
                clustered_indices = np.flatnonzero(labels > 0)
                clustered_labels = labels[clustered_indices]

                n_clustered = len(clustered_indices)
                n_clusters = np.unique(clustered_labels).size
                coverage = n_clustered / len(labels)

                silhouette = np.nan
                coverage_weighted_silhouette = np.nan
                dbcv = np.nan

                if 2 <= n_clusters < n_clustered:
                    D_clustered = D[
                        np.ix_(clustered_indices, clustered_indices)
                    ]

                    silhouette = silhouette_score(
                        D_clustered,
                        clustered_labels,
                        metric="precomputed",
                    )

                    coverage_weighted_silhouette = (
                        silhouette * coverage
                    )

                # Avoid undefined singleton-cluster cases.
                _, cluster_sizes = np.unique(
                    clustered_labels,
                    return_counts=True,
                )

                if n_clusters >= 2 and np.all(cluster_sizes >= 2):
                    evaluation_labels = prepare_dbcv_labels(labels)

                    dbcv = dbcv_module.validity_index(
                        D,
                        evaluation_labels,
                        metric="precomputed",
                        d=model.X_.shape[1],
                    )

                results.append({
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "metric": model.metric,
                    "n_clusters": int(n_clusters),
                    "coverage": coverage,
                    "noise_fraction": 1 - coverage,
                    "silhouette": silhouette,
                    "coverage_weighted_silhouette":
                        coverage_weighted_silhouette,
                    "dbcv": dbcv,
                    "ari_all": ari_all,
                    "ari_clustered": ari_clustered,
                })

    finally:
        model.eps = original_eps
        model.min_samples = original_min_samples

        if original_labels is not None:
            model.labels_ = original_labels
        elif hasattr(model, "labels_"):
            del model.labels_

    return pd.DataFrame(results)

def plot_evaluation(results):
    metrics = [
        ("silhouette", "Silhouette"),
        ("dbcv", "DBCV"),
        ("ari_all", "ARI — all labelled cases"),
        ("ari_clustered", "ARI — clustered cases"),
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    for ax, (column, title) in zip(axes.flat, metrics):
        table = results.pivot(
            index="min_samples",
            columns="eps",
            values=column,
        ).sort_index().sort_index(axis=1)

        image = ax.imshow(
            np.ma.masked_invalid(table.to_numpy(dtype=float)),
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(
            range(len(table.columns)),
            labels=[f"{value:g}" for value in table.columns],
            rotation=45,
        )
        ax.set_yticks(
            range(len(table.index)),
            labels=table.index,
        )
        ax.set_xlabel("eps")
        ax.set_ylabel("min_samples")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    plt.show()


#Use
if __name__ == "__main__":
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    
    from loading_dataset import load_dataset
    from preprocess import preprocess_dataset
    from vectorization.dense import load_or_embed_texts, apply_pca
    from vectorization.sparse import apply_ngrams_and_vectorization, lsa
    from dbscan import DBSCAN


    train,_, train_headers,_, data_dir= load_dataset()
    output_dir = data_dir/"output"
    train_df=pd.DataFrame(train, columns=train_headers)

    prep_train_df = preprocess_dataset(
        df=train_df,
        text_cols=['open_response'],
        download=False
    )

    # X_train_dense = load_or_embed_texts(
    #     texts=prep_train_df["open_response_clean"].fillna("").tolist(),
    #     file_path=output_dir / "train_embeddings.npy", 
    #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #     recalculate=False,
    # )

    # X_train_dense_pca = apply_pca(X_train_dense, variance_threshold=0.8)

    tfidf_matrix_train, ngrams_train, _= lsa(
        prep_train_df["open_response_clean"].fillna(""),
        sparse_dimensions=100,
    )

    model = DBSCAN(
        eps=0.6,         
        min_samples=4,
        metric="euclidean",
    )

    X = np.asarray(tfidf_matrix_train, dtype=np.float64)
    #labels = model.fit_predict(X,precompute=True)

    # evaluation_labels = prepare_dbcv_labels(labels)
    # n_clusters = np.unique(
    #     evaluation_labels[evaluation_labels >= 0]
    # ).size

    # noise_fraction = np.mean(evaluation_labels == -1)

    # if n_clusters < 2:
    #     score = np.nan
    #     print("DBCV not evaluated: fewer than two clusters.")
    # else: ## Use of precomputed distances
    #     score = validity_index(
    #             np.asarray(model.distance_matrix_, dtype=np.float64),
    #             evaluation_labels,
    #             metric="precomputed",
    #             d=model.X_.shape[1],
    #         )
    #         # ## Recomputing distances
    #         #     score = validity_index(
    #         #         X,
    #         #         evaluation_labels,
    #         #         metric="euclidean",
    #         #     )

    # print("Clusters:", n_clusters)
    # print(f"Noise: {noise_fraction:.1%}")
    # print("DBCV:", score)

    ### Or with Silhouette and DBVC and ARI
    model.fit(tfidf_matrix_train, precompute=True)

    # eps_values=[0.3,0.5, 0.7,0.9,1.1]
    # min_samples_values=[3, 4, 5, 6, 7,]
    eps_values=[0.4,0.45,0.5,0.55, 0.6]
    min_samples_values=[3, 4, 5, 7, 9, 12]

    # model.examine_parameters(
    #     eps_values,
    #     min_samples_values
    # )

    results = evaluate_parameters(
        model,
        eps_values,
        min_samples_values,
        y_true=prep_train_df["gs_text34"].to_numpy(),
    )

    print(results.sort_values("dbcv", ascending=False))

    results.to_csv(
        output_dir / "clustering_evaluation.csv",
        index=False,
    )

    from sklearn.cluster import DBSCAN as ReferenceDBSCAN
    from sklearn.metrics import adjusted_rand_score

    your_labels = model.fit_predict(tfidf_matrix_train)

    reference_labels = ReferenceDBSCAN(
        eps=model.eps,
        min_samples=model.min_samples,
        metric="euclidean",
    ).fit_predict(tfidf_matrix_train)

    print(
        "Agreement between implementations:",
        adjusted_rand_score(your_labels, reference_labels),
    )

    print(
        "Same noise assignments:",
        np.array_equal(your_labels == -1, reference_labels == -1),
    )

    #plot_evaluation(results)