from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def vectorization_dataset(ruta):

    df = pd.read_csv(ruta)

    df_vectorizacion = df.dropna(subset=["open_response"]).copy()

    textos = df_vectorizacion["open_response"].astype(str).tolist()

    print("Cantidad de textos:", len(textos))

    X = model.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Dimensión matriz embeddings:", X.shape)

    np.save("embeddings.npy", X)

    df_vectorizacion[
        ["newid", "gs_text34"]
    ].to_csv(
        "embedding_metadata.csv",
        index=False
    )

    return X, df_vectorizacion


ruta = r"C:\Users\Agustina\semestre7\intercambio\Mineria de datos\DatosProyecto\cleaned_PHMRC_VAI_redacted_free_text.train.csv" 
X, df_vectorizacion = vectorization_dataset(ruta)

from sklearn.metrics.pairwise import cosine_similarity

def show_similarity(indice, X, textos, cantidad=5):
    similitudes = cosine_similarity(
        X[indice].reshape(1, -1),
        X
    )[0]

    indices = np.argsort(similitudes)[::-1]

    print("Texto de referencia:")
    print(textos.iloc[indice])
    print()

    print(f"{cantidad} textos más similares:")

    for i in indices[:cantidad]:
        print(f"\nSimilitud: {similitudes[i]:.4f}")
        print(textos.iloc[i])