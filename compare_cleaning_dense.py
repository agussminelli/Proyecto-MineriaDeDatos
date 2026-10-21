
import pandas as pd
from sentence_transformers import SentenceTransformer

from loading_dataset import load_dataset
from vectorization.dense import embed_texts, apply_pca
from dbscan import DBSCAN
from silhouette_dense import evaluate_silhouette
from inspect_narratives import (
    remove_irrelevant_phrases,
    remove_narrative_introductions
)


def main():

    train, _, train_headers, _, _ = load_dataset()
    train_df = pd.DataFrame(train, columns=train_headers)

    original_texts = (
        train_df["open_response"].fillna("").astype(str).tolist()
    )

    administrative_texts = remove_irrelevant_phrases(
        original_texts
    )

    fully_cleaned_texts = remove_narrative_introductions(
        administrative_texts
    )

    datasets = {
        "Original": original_texts,
        "Administrative": administrative_texts,
        "Fully cleaned": fully_cleaned_texts
    }

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True
    )

    results = []

    for name, texts in datasets.items():

        print(f"\nEvaluating: {name}")

        X = embed_texts(texts, embedding_model)

        X_pca = apply_pca(
            X,
            variance_threshold=0.90
        )

        model = DBSCAN()
        model.fit(X_pca, precompute=True)

        scores = evaluate_silhouette(
            model,
            eps_values=[0.5, 0.6, 0.7],
            min_samples_values=[3, 5]
        )

        scores["dataset"] = name
        results.append(scores)

    comparison = pd.concat(results, ignore_index=True)

    print("\nComparison results:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
