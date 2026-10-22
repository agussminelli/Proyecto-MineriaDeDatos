#DBVC implemented for python in hdbscan
#Silhouette from sklearn, corrected with noise after SWC
from pathlib import Path
from time import perf_counter
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

def calculate_ari(labels, y_true):
    """Return ARI for all labelled cases and clustered labelled cases."""
    if y_true is None:
        return np.nan, np.nan

    y = pd.Series(y_true).reset_index(drop=True)

    if len(y) != len(labels):
        raise ValueError("Reference labels must match clustering row order")

    text = y.astype("string").str.strip()

    valid = (
        text.notna()
        & ~text.str.casefold().isin(
            ["", "na", "nan", "n/a", "none", "null"]
        )
    ).to_numpy(dtype=bool)

    # Strings are valid reference labels for ARI.
    values = text.fillna("").to_numpy(dtype=str)
    clustered = valid & (labels > 0)

    ari_all = (
        adjusted_rand_score(values[valid], labels[valid])
        if valid.sum() >= 2 else np.nan
    )

    ari_clustered = (
        adjusted_rand_score(values[clustered], labels[clustered])
        if clustered.sum() >= 2 else np.nan
    )

    return ari_all, ari_clustered

def evaluate_parameters(
    model,
    eps_values,
    min_samples_values,
    y_true=None,
    y_true_grouped=None,
    return_assignments=False,
):
    """Evaluate Silhouette and DBCV for each parameter combination.
    ARI for Comparison, if GT is provided.

    Silhouette excludes noise; DBCV includes noise.
    Restores the model's original parameters and labels afterward.
    """
    eps_values = list(eps_values)
    min_samples_values = list(min_samples_values)

    if not eps_values or not min_samples_values:
        raise ValueError("Parameter ranges must not be empty")

    for eps in eps_values:
        if not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps values must be finite positive numbers")

    for value in min_samples_values:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError("min_samples values must be positive integers")

    assignments = {}

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
                
                fit_started = perf_counter()
                labels = model.fit_predict()
                fit_seconds = perf_counter() - fit_started
                
                run_key = f"grid_{len(results):04d}"
                if return_assignments:
                    assignments[run_key] = labels.copy()

                if np.any(labels == 0):
                    raise ValueError("Some points remain unassigned")

                ari_all, ari_clustered = calculate_ari(labels, y_true)

                ari_grouped_all, ari_grouped_clustered = calculate_ari(
                    labels,
                    y_true_grouped,
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
                    "run_key": run_key,
                    "p": float(model.p) if model.metric == "minkowski" else np.nan,
                    "fit_seconds_cached": fit_seconds,
                    "n_clusters": int(n_clusters),
                    "coverage": coverage,
                    "noise_fraction": 1 - coverage,
                    "silhouette": silhouette,
                    "coverage_weighted_silhouette":
                        coverage_weighted_silhouette,
                    "dbcv": dbcv,
                    "ari_all": ari_all,
                    "ari_clustered": ari_clustered,
                    "ari_grouped_all": ari_grouped_all,
                    "ari_grouped_clustered": ari_grouped_clustered,
                })

    finally:
        model.eps = original_eps
        model.min_samples = original_min_samples

        if original_labels is not None:
            model.labels_ = original_labels
        elif hasattr(model, "labels_"):
            del model.labels_

    table = pd.DataFrame(results)

    if return_assignments:
        return table, assignments

    return table

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

def save_cluster_examples(
    df,
    labels,
    distance_matrix,
    output_path,
    clusters_per_group=3,
    random_per_cluster=2,
    noise_examples=5,
    seed=42,
):
    """Save cluster examples as a UTF-8 text report.

    Preconditions:
        df, labels and distance_matrix use exactly the same row order.
        Labels are -1 for noise and positive integers for clusters.
        Distances match the representation and metric used for clustering.

    Every example prints original (gs_text34) and grouped (gs_grouped)
    labels; absent or empty labels are shown as <missing>.

    Representatives:
        Medoid: smallest mean distance to other cluster members.
        Peripheral: largest distance from the medoid.
        Random: sampled from remaining members without replacement.
    """
    labels = np.asarray(labels)
    D = np.asarray(distance_matrix)
    n = len(df)

    if labels.shape != (n,) or D.shape != (n, n):
        raise ValueError("Data, labels and distances must have matching shapes")

    if np.any(labels < -1) or np.any(labels == 0):
        raise ValueError("Expected -1 for noise and positive cluster IDs")

    if "open_response" not in df.columns:
        raise ValueError("df must contain open_response")

    if clusters_per_group < 1 or random_per_cluster < 0 or noise_examples < 0:
        raise ValueError("Invalid example counts")

    rng = np.random.default_rng(seed)
    lines = []

    def add_distribution(group, column, top=None):
        if column not in group.columns:
            return

        counts = group[column].fillna("<missing>").value_counts()
        if top is not None:
            counts = counts.head(top)

        lines.append(f"{column}:")
        for value, count in counts.items():
            lines.append(
                f"  {value}: {count} ({count / len(group):.1%})"
            )

    def add_text(index, role):
        row = df.iloc[index]
        
        # identifier = row["newid"] if "newid" in df.columns else index
        lines.append(f"\n[{role}]") # ID: {identifier}")

        if "module" in df.columns:
            lines.append(f"module: {row['module']}")
        # Always display both reference fields, including for noise examples.
        # Missing labels are explicit; never infer one from the other here.
        for column in ["gs_text34", "gs_grouped"]:
            value = row[column] if column in df.columns else None
            shown = "<missing>" if pd.isna(value) or str(value).strip() == "" else str(value)
            lines.append(f"{column}: {shown}")

        text = row["open_response"]
        lines.append("" if pd.isna(text) else str(text))

    cluster_ids, sizes = np.unique(
        labels[labels > 0],
        return_counts=True,
    )
    size_map = dict(zip(cluster_ids, sizes))

    lines.extend([
        "CLUSTER INSPECTION",
        f"Random seed: {seed}",
        f"Total narratives: {n}",
        f"Clusters: {len(cluster_ids)}",
        f"Noise: {np.sum(labels == -1)}",
        f"Noise fraction: {np.mean(labels == -1):.1%}",
    ])

    # Noise examples
    noise_indices = np.flatnonzero(labels == -1)
    sampled_noise = rng.choice(
        noise_indices,
        size=min(noise_examples, len(noise_indices)),
        replace=False,
    )

    lines.append("\n" + "=" * 60)
    lines.append("RANDOM NOISE EXAMPLES")

    for index in sampled_noise:
        add_text(index, "Random noise")

    # Break size ties by cluster ID.
    largest = sorted(
        cluster_ids,
        key=lambda c: (-size_map[c], c),
    )[:clusters_per_group]

    # Avoid printing the same cluster in both categories.
    largest_set = set(largest)
    smallest = sorted(
        [c for c in cluster_ids if c not in largest_set],
        key=lambda c: (size_map[c], c),
    )[:clusters_per_group]

    for heading, selected_clusters in [
        ("LARGEST CLUSTERS", largest),
        ("SMALLEST REMAINING CLUSTERS", smallest),
    ]:
        lines.append("\n" + "=" * 60)
        lines.append(heading)

        if not selected_clusters:
            lines.append("No additional clusters available.")
            continue

        for cluster_id in selected_clusters:
            members = np.flatnonzero(labels == cluster_id)
            group = df.iloc[members]

            lines.extend([
                "\n" + "-" * 60,
                f"Cluster {cluster_id}: {len(members)} narratives",
                f"Share of dataset: {len(members) / n:.1%}",
            ])

            add_distribution(group, "module")
            add_distribution(group, "gs_text34", top=5)
            add_distribution(group, "gs_grouped", top=5)

            # Compute mean within-cluster distances in row batches,
            # avoiding a full extra cluster-by-cluster matrix.
            distance_sums = np.empty(len(members), dtype=float)

            for start in range(0, len(members), 128):
                stop = min(start + 128, len(members))
                block = members[start:stop]

                distance_sums[start:stop] = D[
                    np.ix_(block, members)
                ].sum(axis=1)

            medoid = int(members[np.argmin(distance_sums)])
            add_text(medoid, "Medoid")

            selected = {medoid}

            other_members = members[members != medoid]

            if len(other_members):
                peripheral = int(
                    other_members[
                        np.argmax(D[medoid, other_members])
                    ]
                )
                add_text(peripheral, "Furthest from medoid")
                selected.add(peripheral)

            remaining = np.array(
                [i for i in members if i not in selected],
                dtype=int,
            )

            random_indices = rng.choice(
                remaining,
                size=min(random_per_cluster, len(remaining)),
                replace=False,
            )

            for index in random_indices:
                add_text(index, "Random member")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path


# Example Use
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

    ### With Silhouette and DBVC and ARI
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
        y_true_grouped=prep_train_df["gs_grouped"].to_numpy(),
    )

    print(results.sort_values("dbcv", ascending=False))

    results.to_csv(
        output_dir / "clustering_evaluation.csv",
        index=False,
    )

    #plot_evaluation(results)