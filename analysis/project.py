"""Earlier dense/PCA exploration; use run_experiment.py for final experiments.

This script embeds raw text, applies PCA retaining 90% variance, and explores
parameters using the historical coverage-weighted silhouette heuristic.
"""

from pathlib import Path

import pandas as pd

from dbscan import DBSCAN
from features.loading_dataset import load_dataset
from features.preprocess import preprocess_dataset
from analysis.silhouette_dense import evaluate_silhouette
from features.dense import apply_pca, load_or_embed_texts


VERBOSE = True
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EPS_VALUES = [0.3, 0.5, 0.7, 0.9, 1.1]
MIN_SAMPLES_VALUES = [2, 3, 4, 5]


def main():
    train, _, train_headers, _, data_dir = load_dataset()
    output_dir = Path("outputs/exploration/project")
    train_df = pd.DataFrame(train, columns=train_headers)

    # Cleaning is retained for the historical side-by-side inspection.
    # The embeddings below intentionally use the original text column.
    prepared = preprocess_dataset(
        train_df,
        text_cols=["open_response"],
    )

    if VERBOSE:
        print(prepared[["open_response", "open_response_clean"]].head())

    dense = load_or_embed_texts(
        texts=prepared["open_response"].fillna("").tolist(),
        file_path=output_dir / "train_embeddings.npy",
        model_name=MODEL_NAME,
        recalculate=False,
    )
    reduced = apply_pca(dense, variance_threshold=0.90)

    model = DBSCAN()
    model.fit(reduced, precompute=True)

    results = evaluate_silhouette(
        model,
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
    )

    print("\nSilhouette results:")
    print(results.to_string(index=False))

    print("\nTop configurations under the historical heuristic:")
    ranked = results.sort_values("adjusted_silhouette", ascending=False)
    print(ranked.head(5).to_string(index=False))

    model.examine_parameters(
        eps_values=EPS_VALUES,
        min_samples_values=MIN_SAMPLES_VALUES,
    )


if __name__ == "__main__":
    main()
