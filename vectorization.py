from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# Modelo multilingüe
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Dataset
ruta = r"C:\Users\Agustina\semestre7\intercambio\Mineria de datos\DatosProyecto\cleaned_PHMRC_VAI_redacted_free_text.train.csv"

df = pd.read_csv(ruta)


# Miramos estructura
print(df.columns)
#print(df.head())


# Tomamos solo registros que tengan texto
df_vectorizacion = df.dropna(subset=["open_response"]).copy()


# Convertimos el texto a string
textos = df_vectorizacion["open_response"].astype(str).tolist()


print("Cantidad de textos:", len(textos))


# Generar embeddings
X = model.encode(
    textos,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


print("Dimensión matriz embeddings:", X.shape)


# Guardar embeddings
np.save("embeddings.npy", X)

df_vectorizacion[
    ["newid", "gs_text34"]
].to_csv(
    "embedding_metadata.csv",
    index=False
)