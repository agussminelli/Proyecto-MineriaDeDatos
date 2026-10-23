"""Earlier cleaning comparison; the active variant remains Original only.

To reproduce all three variants, explicitly enable the commented dictionary
entries below. Doing so runs additional embedding and clustering experiments.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer

from features.loading_dataset import load_dataset
from features.dense import embed_texts, apply_pca
from dbscan import DBSCAN
from analysis.silhouette_dense import evaluate_silhouette
from analysis.inspect_narratives import (
    remove_irrelevant_phrases,
    remove_narrative_introductions
)


def main():

    train, _, train_headers, _, _ = load_dataset()
    train_df = pd.DataFrame(train, columns=train_headers)

    original_texts = (train_df["open_response"].fillna("").astype(str).tolist())
    administrative_texts = remove_irrelevant_phrases(original_texts)
    fully_cleaned_texts = remove_narrative_introductions(administrative_texts)

    datasets = {
        "Original": original_texts,
        # "Administrative": administrative_texts,
        # "Fully cleaned": fully_cleaned_texts,
    }

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True
    )

    results = []

    for name, texts in datasets.items():
        print(f"\nEvaluating: {name}")

        X = embed_texts(texts, embedding_model)
        X_pca = apply_pca(X,variance_threshold=0.90)

        model = DBSCAN()
        model.fit(X_pca, precompute=True)

        scores = evaluate_silhouette(
            model,
            eps_values=[0.55, 0.575, 0.60, 0.625, 0.65],
            min_samples_values=[3, 4, 5, 6, 8],
        )

        scores["dataset"] = name
        results.append(scores)

    comparison = pd.concat(results, ignore_index=True)

    print("\nComparison results:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
