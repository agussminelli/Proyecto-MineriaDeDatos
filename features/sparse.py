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
    texts,
    n_min=1,
    n_max=2,
    min_docs=5,
    max_feat=1000,
    sparse_dimensions=50,
    random_state=42,
):
    """Fit TF-IDF and SVD, then L2-normalize the reduced vectors.

    Returns
    -------
    vectors : ndarray
        Normalized LSA representation in input row order.
    vectorizer : TfidfVectorizer
        Fitted vocabulary and TF-IDF transformation.
    svd : TruncatedSVD
        Fitted dimensionality reduction for reuse on unseen texts.
    """
    tfidf, _, vectorizer = apply_ngrams_and_vectorization(
        texts,
        n_min=n_min,
        n_max=n_max,
        min_docs=min_docs,
        max_feat=max_feat,
    )

    if not 1 <= sparse_dimensions < min(tfidf.shape):
        raise ValueError(
            "LSA components must be positive and smaller than "
            "both the number of rows and TF-IDF features"
        )

    svd = TruncatedSVD(n_components=sparse_dimensions,random_state=random_state,)

    vectors = normalize(svd.fit_transform(tfidf),norm="l2",)

    return vectors, vectorizer, svd