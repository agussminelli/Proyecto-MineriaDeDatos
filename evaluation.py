#DBVC implemented for python in hdbscan

import numpy as np
from hdbscan.validity import validity_index

def prepare_dbcv_labels(labels):
    labels = np.asarray(labels, dtype=int)

    if np.any(labels == 0):
        raise ValueError("Clustering still contains unassigned points")

    converted = np.full(labels.shape, -1, dtype=int)
    cluster_ids = np.unique(labels[labels > 0])

    for new_id, original_id in enumerate(cluster_ids):
        converted[labels == original_id] = new_id

    return converted

#Use
if __name__ == "__main__":
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    
    from loading_dataset import load_dataset
    from preprocess import preprocess_dataset
    from vectorization.dense import load_or_embed_texts
    from dbscan import DBSCAN


    train,_, train_headers,_, data_dir= load_dataset()
    output_dir = data_dir/"output"
    train_df=pd.DataFrame(train, columns=train_headers)

    prep_train_df = preprocess_dataset(
        df=train_df,
        text_cols=['open_response'] 
    )

    X_train_dense = load_or_embed_texts(
        texts=prep_train_df["open_response"].fillna("").tolist(),
        file_path=output_dir / "train_embeddings.npy",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        recalculate=False,
    )

    model = DBSCAN(
        eps=0.7,         
        min_samples=3,
        metric="euclidean",
    )
    X = np.asarray(X_train_dense, dtype=np.float64)
    labels = model.fit_predict(X,precompute=True)

    evaluation_labels = prepare_dbcv_labels(labels)
    n_clusters = np.unique(
        evaluation_labels[evaluation_labels >= 0]
    ).size

    noise_fraction = np.mean(evaluation_labels == -1)

    if n_clusters < 2:
        score = np.nan
        print("DBCV not evaluated: fewer than two clusters.")
    else: ## Use of precomputed distances
        score = validity_index(
                np.asarray(model.distance_matrix_, dtype=np.float64),
                evaluation_labels,
                metric="precomputed",
                d=model.X_.shape[1],
            )
            # ## Recomputing distances
            #     score = validity_index(
            #         X,
            #         evaluation_labels,
            #         metric="euclidean",
            #     )

    print("Clusters:", n_clusters)
    print(f"Noise: {noise_fraction:.1%}")
    print("DBCV:", score)