from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_or_embed_texts(
    texts,
    file_path,
    model_name,
    recalculate=False,
    local_files_only=True,
):
    """Load cached embeddings, or calculate and save new ones.

    Set recalculate=True if texts, their order, preprocessing,
    or the embedding model have changed.
    """
    file_path = Path(file_path)
    texts = list(texts)

    if file_path.exists() and not recalculate:
        X = np.load(file_path, allow_pickle=False)

        if X.ndim != 2 or X.shape[0] != len(texts):
            raise ValueError(
                "Cached embeddings do not match the text count. "
                "Run with recalculate=True."
            )

        print(f"Loaded embeddings: {file_path}, shape={X.shape}")
        return X

    embedding_model = SentenceTransformer(
        model_name,
        local_files_only=local_files_only,
    )

    X = embed_texts(texts, embedding_model)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, X)

    print(f"Saved embeddings: {file_path}")
    return X

def embed_texts(texts, model, batch_size=32):
    """Return normalised vectors in the same order as the input texts."""
    if not texts:
        raise ValueError("No texts provided")

    print("Number of texts:", len(texts))

    X = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding matrix shape:", X.shape)

    return X

def save_embeddings(X, headers, rows, output_dir, prefix):
    """Save embeddings and matching metadata in identical row order."""
    import csv

    if len(X) != len(rows):
        raise ValueError("Embeddings and metadata row counts differ")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_embeddings.npy", X)

    # with (output_dir / f"{prefix}_metadata.csv").open(
    #     "w", newline="", encoding="utf-8"
    # ) as file:
    #     writer = csv.writer(file)
    #     writer.writerow(headers)
    #     writer.writerows(rows)


def show_similarity(index, X, texts, num_results=5):

    similarities = cosine_similarity(
        X[index].reshape(1, -1),
        X
    )[0]
    # Embeddings are normalised, so dot product = cosine similarity.
    # similarities = X @ X[index]

    sorted_indices = np.argsort(similarities)[::-1]

    print("Reference text:")
    print(texts[index])
    print()

    print(f"{num_results} most similar texts:")

    for i in sorted_indices[1:num_results + 1]:
        print(f"\nSimilarity: {similarities[i]:.4f}")
        print(texts[i])



if __name__ == "__main__":
    import sys
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))

    from loading_dataset import load_dataset
    ## data_dir = r"C:\Users\Agustina\semestre7\intercambio\Mineria de datos\DatosProyecto
    train, _, train_headers, _, data_dir = load_dataset() #if it doesnt automatically refer to the right dir, add dir here
    
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=True, #Set to False, for first time downloaded
    )

    output_dir = data_dir / "Output"

    for prefix, headers, rows in [
        ("train", train_headers, train),
    ]:
        text_index = headers.index("open_response")
        texts = [row[text_index] for row in rows]

        X = embed_texts(texts, embedding_model)
        save_embeddings(X, headers, rows, output_dir, prefix)

        show_similarity(0, X, texts, num_results=5)
