import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def apply_ngrams_and_vectorization(cleaned_text_series, n_min=1, n_max=2, min_docs=5, max_feat=1000):
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
        max_features=max_feat        
    )
    
    # Implement sparse vectorization (Learn vocabulary and transform to TF-IDF)
    sparse_tfidf_matrix = vectorizer.fit_transform(cleaned_text_series)
    
    # Retrieve the names of the generated n-grams
    ngram_features = vectorizer.get_feature_names_out()
    
    return sparse_tfidf_matrix, ngram_features, vectorizer


