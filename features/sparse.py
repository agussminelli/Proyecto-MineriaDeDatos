from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def apply_ngrams_and_vectorization(
    cleaned_text_series,
    n_min=1,
    n_max=2,
    min_docs=5,
    max_feat=1000,
):
    """
    Implements n-grams and sparse TF-IDF vectorization in a single step.
    Assumes the input text has already been preprocessed (lowercased,
    lemmatized, and stripped of stop words).

    Parameters:
    cleaned_text_series (pd.Series): The preprocessed text column.
    n_min (int): Minimum n-gram size (1 = unigrams).
    n_max (int): Maximum n-gram size (2 = bigrams).
    min_docs (int): Minimum frequency threshold to ignore rare terms.
    max_feat (int): Maximum dimensions of the resulting sparse matrix.
    """

    # Initialize the vectorizer to handle both n-grams and sparse vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(n_min, n_max),
        stop_words=None,
        min_df=min_docs,
        max_features=max_feat,
    )

    # Implement sparse vectorization (Learn vocabulary and transform to TF-IDF)
    sparse_tfidf_matrix = vectorizer.fit_transform(cleaned_text_series)

    # Retrieve the names of the generated n-grams
    ngram_features = vectorizer.get_feature_names_out()

    return sparse_tfidf_matrix, ngram_features, vectorizer


def lsa(
    cleaned_text_series,
    n_min=1,
    n_max=2,
    min_docs=5,
    max_feat=1000,
    sparse_dimensions=50,
):
    """Fit TF-IDF, reduce with TruncatedSVD and L2-normalize each row.

    Parameters
    ----------
    cleaned_text_series : iterable of str
        Training texts; cleaning must already have been applied.
    n_min, n_max : int
        Inclusive n-gram range for TF-IDF.
    min_docs : int or float
        Minimum document count or fraction accepted by TfidfVectorizer.
    max_feat : int
        Maximum TF-IDF vocabulary size.
    sparse_dimensions : int
        Requested number of SVD components; must suit the fitted matrix.

    Returns
    -------
    X_train_lsa : ndarray
        Dense, row-normalized training representation.
    n_grams : ndarray of str
        Original TF-IDF vocabulary names, not names of SVD components.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF transformer.

    Notes
    -----
    Uses random_state=42 and prints shape and retained variance. This legacy
    helper does not return its fitted SVD, so its returned objects alone cannot
    transform unseen texts into the same space. The final experiment pipeline
    separately saves both fitted transformers.
    """

    sparse_tfidf_matrix, n_grams, vectorizer = apply_ngrams_and_vectorization(
        cleaned_text_series,
        n_min,
        n_max,
        min_docs,
        max_feat,
    )

    svd_reducer = TruncatedSVD(n_components=sparse_dimensions, random_state=42)
    X_train_lsa = svd_reducer.fit_transform(sparse_tfidf_matrix)

    # Normalise after dimensionality reduction.
    X_train_lsa = normalize(X_train_lsa, norm="l2")

    print("LSA shape:", X_train_lsa.shape)
    print("Retained variance:", svd_reducer.explained_variance_ratio_.sum())

    return X_train_lsa, n_grams, vectorizer
