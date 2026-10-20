##Proyecto de minería de datos

import pandas as pd
from sentence_transformers import SentenceTransformer

from loading_dataset import load_dataset
from preprocess import preprocess_dataset
from vectorization.sparse import apply_ngrams_and_vectorization
from vectorization.dense import load_or_embed_texts, apply_pca
from vectorization.hybrid import create_hybrid_embeddings
from dbscan import DBSCAN
from silhouette_dense import evaluate_silhouette

VERBOSE=True #set to false to skip terminal outputs

#Example of how to investigate Params
def main():
    train, test, train_headers,_, data_dir = load_dataset()

    output_dir = data_dir/"output"

    train_df=pd.DataFrame(train, columns=train_headers)

    prep_train_df=preprocess_dataset(train_df, text_cols=['open_response'])

    if VERBOSE:
        print(prep_train_df[['open_response', 'open_response_clean']].head())

    # tfidf_matrix_train, ngrams_train, _= apply_ngrams_and_vectorization(
    #     prep_train_df["open_response_clean"].fillna("")
    # )

    # # #for testdata
    # # tfidf_matrix_test = vectorizer.transform(
    # #     prep_test_df["open_response_clean"].fillna("")
    # # )

    # if VERBOSE:
    #     print("Train:", tfidf_matrix_train.shape)
    #     #print(tfidf_matrix_train, ngrams_train)

    X_train_dense = load_or_embed_texts(
        texts=prep_train_df["open_response"].fillna("").tolist(),
        file_path=output_dir / "train_embeddings.npy",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        recalculate=False,
    )
    ## FOR PCA ON DENSE EMBEDDINGS
    X_train_pca = apply_pca(
    X_train_dense,
    variance_threshold=0.90
    )
  
    # X_hybrid_train = create_hybrid_embeddings(tfidf_matrix_train,X_train_dense)

   # DBSCAN evaluation
    model1 = DBSCAN()
    model1.fit(X_train_pca, precompute=True)

    results = evaluate_silhouette(
        model1,
        eps_values=[0.3, 0.5, 0.7, 0.9, 1.1],
        min_samples_values=[2, 3, 4, 5]
    )

    print("\nSilhouette results:")
    print(results.to_string(index=False))

    print("\nBest configurations:")
    print(
        results.sort_values(
            "adjusted_silhouette",
            ascending=False
        ).head(5).to_string(index=False)
    )

    counts, noise, candidates = model1.examine_parameters(
        eps_values=[0.3, 0.5, 0.7, 0.9, 1.1],
        min_samples_values=[2,3,4,5],
    )



if __name__=="__main__":
    main()