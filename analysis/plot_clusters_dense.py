"""Earlier raw-text/PCA visualization with experiment-specific parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from features.loading_dataset import load_dataset
from features.preprocess import preprocess_dataset
from features.dense import embed_texts, apply_pca
from dbscan import DBSCAN

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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