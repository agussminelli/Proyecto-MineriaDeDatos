
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
)


from pathlib import Path

def vectorize_dataset(file_path, output_dir):

    df = pd.read_csv(file_path)

    df_vectorized = df.dropna(subset=["open_response"]).copy()

    texts = df_vectorized["open_response"].astype(str).tolist()

    print("Number of texts:", len(texts))

    X = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding matrix shape:", X.shape)

    # Create output directory if it does not exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_path / "embeddings.npy", X)

    # Save metadata
    df_vectorized[
        ["newid", "gs_text34"]
    ].to_csv(
        output_path / "embedding_metadata.csv",
        index=False
    )

    return X, df_vectorized


## This is my file path, you can change it to your own file path to where the dataset is located
## and to where you want to save the embeddings and metadata

file_path = r"C:\Users\Agustina\semestre7\intercambio\Mineria de datos\DatosProyecto\cleaned_PHMRC_VAI_redacted_free_text.train.csv"
output_dir = r"C:\Users\Agustina\semestre7\intercambio\Mineria de datos\DatosProyecto\LocalData"

X, df_vectorized = vectorize_dataset(file_path, output_dir)



def show_similarity(index, X, texts, num_results=5):

    similarities = cosine_similarity(
        X[index].reshape(1, -1),
        X
    )[0]

    sorted_indices = np.argsort(similarities)[::-1]

    print("Reference text:")
    print(texts.iloc[index])
    print()

    print(f"{num_results} most similar texts:")

    for i in sorted_indices[1:num_results + 1]:
        print(f"\nSimilarity: {similarities[i]:.4f}")
        print(texts.iloc[i])



# ## Show similarity for a specific text
# show_similarity(
#     0,
#     X,
#     df_vectorized["open_response"],
#     num_results=5
# )

# show_similarity(
#     25,
#     X,
#     df_vectorized["open_response"],
#     num_results=10
# )
