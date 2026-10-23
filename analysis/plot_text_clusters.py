"""Earlier raw-text/PCA visualization with experiment-specific parameters."""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import json

from features.loading_dataset import load_dataset
from features.preprocess import preprocess_dataset
from features.dense import embed_texts, apply_pca
from dbscan import DBSCAN

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NOISE_COLOR = "#888888"
OTHER_COLOR = "#BCC5CC"


def plot_dbscan_clusters_2d(X, labels, title, output_path):
    """
    Project data to 2D with PCA and plot DBSCAN clusters.
    Noise points (-1) are shown separately.
    This projection is a visualization, not the clustering space.
    """
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    unique_labels = sorted(set(labels))

    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        mask = labels == label

        if label == -1:
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=10,
                alpha=0.4,
                label="Noise"
            )
        else:
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=18,
                alpha=0.8,
                label=f"Cluster {label}"
            )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(title)
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_text_clusters(
    features,
    labels,
    output_dir,
    *,
    title="Text clustering",
    metadata=None,
    max_clusters=12,
    filename="text_clusters",
):
    """Save a PCA view, cluster-size chart and aggregate JSON summary.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Finite vectors in the clustering space, with at least two columns.
    labels : array-like of shape (n_samples,)
        Integer cluster labels in matching row order. -1 denotes noise;
        nonnegative IDs (including sklearn's zero) are accepted as clusters.
    output_dir : str or Path
        Destination directory. Matching filenames are overwritten.
    title : str
        Figure title, independent of the dataset or experiment loader.
    metadata : dict, optional
        JSON-compatible provenance. Optional eps and min_samples are shown.
        Do not include narrative text or other information meant to stay private.
    max_clusters : int
        Show the largest 1 to 20 clusters separately; aggregate the rest.
    filename : str
        Plain filename stem for PNG, SVG and JSON outputs.

    Returns
    -------
    dict
        Paths keyed by png, svg and summary. No clustering is fitted;
        PCA is fitted for display only. Input metadata is not modified.
    """
    output_dir = Path(output_dir)
    metadata = dict(metadata or {})
    labels = np.asarray(labels)

    if not 1 <= max_clusters <= 20:
        raise ValueError("max_clusters must be between 1 and 20")
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a plain filename stem")
    features = np.asarray(features, dtype=np.float64)

    if (
        features.ndim != 2
        or min(features.shape) < 2
        or not np.isfinite(features).all()
    ):
        raise ValueError("Need a finite feature matrix with >=2 rows and columns")

    if (
        labels.ndim != 1
        or len(labels) != len(features)
        or not np.isfinite(labels).all()
        or np.any(labels != np.floor(labels))
        or np.any(labels < -1)
    ):
        raise ValueError("Labels must be aligned integers, with -1 for noise")

    labels = labels.astype(int)

    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(features)
    explained = float(pca.explained_variance_ratio_.sum())

    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    ranked = sorted(zip(ids, counts), key=lambda pair: (-pair[1], pair[0]))
    displayed = ranked[:max_clusters]
    remaining = ranked[max_clusters:]
    palette = plt.get_cmap("tab20")
    colors = {
        int(cluster): palette(index)
        for index, (cluster, _) in enumerate(displayed)
    }
    noise_count = int(np.sum(labels == -1))
    assigned_count = int(np.sum(labels >= 0))

    fig, (scatter_ax, size_ax) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.subplots_adjust(top=0.79, bottom=0.19, wspace=0.32)
    fig.suptitle(title, fontsize=17, y=0.97)
    fig.text(
        0.5,
        0.89,
        f"{len(labels):,} records | {features.shape[1]} features | "
        + (
            f"eps = {metadata['eps']:g} | "
            f"min_samples = {metadata['min_samples']}\n"
            if "eps" in metadata and "min_samples" in metadata
            else "\n"
        )
        + f"{len(ids)} clusters | coverage = {assigned_count / len(labels):.1%} | "
        f"noise = {noise_count / len(labels):.1%}",
        ha="center",
        fontsize=10,
    )

    noise = labels == -1
    scatter_ax.scatter(*coordinates[noise].T, color=NOISE_COLOR, marker="x", s=22, alpha=0.6)

    if remaining:
        other_ids = [cluster for cluster, _ in remaining]
        other = np.isin(labels, other_ids)
        scatter_ax.scatter(*coordinates[other].T, color=OTHER_COLOR, s=15, alpha=0.55)

    for cluster, _ in displayed:
        members = labels == cluster
        scatter_ax.scatter(
            *coordinates[members].T,
            color=colors[int(cluster)],
            s=42 if len(labels) < 50 else 12,
            alpha=0.7,
            edgecolors="none",
        )

    scatter_ax.set(
        title=f"PCA view only ({explained:.1%} variance retained)",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    scatter_ax.grid(alpha=0.15)

    names = [f"Cluster {cluster}" for cluster, _ in displayed]
    sizes = [int(size) for _, size in displayed]
    bar_colors = [colors[int(cluster)] for cluster, _ in displayed]

    if remaining:
        names.append(f"Other {len(remaining)} clusters")
        sizes.append(int(sum(size for _, size in remaining)))
        bar_colors.append(OTHER_COLOR)

    names.append("Noise (crosses)")
    sizes.append(noise_count)
    bar_colors.append(NOISE_COLOR)
    size_ax.barh(names, sizes, color=bar_colors)
    size_ax.invert_yaxis()
    size_ax.set(
        title="Cluster sizes, including noise", xlabel="Number of records"
    )
    size_ax.set_xlim(0, max(sizes) * 1.3 + 1)

    for index, size in enumerate(sizes):
        size_ax.text(
            size + max(sizes) * 0.02,
            index,
            f"{size:,} ({size / len(labels):.1%})",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.085,
        "Clustering uses the original feature space. The 2D projection can "
        "distort distances and apparent separation.",
        ha="center",
        fontsize=9,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for extension in ("png", "svg"):
        path = output_dir / f"{filename}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        paths[extension] = path

    plt.close(fig)
    metadata.update(
        {
            "n_samples": len(labels),
            "n_features": features.shape[1],
            "n_clusters": len(ids),
            "noise_count": noise_count,
            "coverage": assigned_count / len(labels),
            "projection": "PCA, 2 components, full SVD; display only",
            "projection_variance_retained": explained,
            "cluster_sizes": {str(int(i)): int(n) for i, n in ranked},
        }
    )
    (output_dir / f"{filename}_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    paths["summary"] = output_dir / f"{filename}_summary.json"
    return paths



def main():
    # Load data
    train, _, train_headers, _, data_dir = load_dataset(apply_grouping=False)
    train_df = pd.DataFrame(train, columns=train_headers)

    # Preprocess
    prep_train_df = preprocess_dataset(train_df,text_cols=["open_response"])

    texts = prep_train_df["open_response"].fillna("").tolist()

    # Dense embeddings
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=True
    )

    X_dense = embed_texts(texts, embedding_model)

    # PCA for clustering
    X_pca = apply_pca(X_dense, variance_threshold=0.90)

    # Use the parameter combination you want to visualize
    eps = 0.625
    min_samples = 8

    model = DBSCAN(eps=eps,min_samples=min_samples)
    labels = model.fit_predict(X_pca)

    n_clusters = len(set(labels) - {-1})
    noise_fraction = np.mean(labels == -1)

    print("Clusters:", n_clusters)
    print(f"Noise fraction: {noise_fraction:.2%}")

    filename = f"dbscan_dense_eps{eps}_min{min_samples}.png"
    output_path = Path("outputs/figures") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"DBSCAN clusters in 2D PCA space\n"
        f"eps={eps}, min_samples={min_samples}, "
        f"clusters={n_clusters}, noise={noise_fraction:.2%}"
    )

    plot_dbscan_clusters_2d(
        X_pca,
        labels,
        title,
        output_path
    )

    print("Plot saved to:", output_path)


if __name__ == "__main__":
    main()