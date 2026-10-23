import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

from features.loading_dataset import load_dataset

# Solve  SSL-certificate problems (usual for Mac )
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def preprocess_dataset(
    df,
    text_cols=None,
    cat_cols=None,
    num_cols=None,
    download=True,
):
    """Return a copy with cleaned columns for each requested text field.

    Parameters
    ----------
    df : pandas.DataFrame
        Input records; the original DataFrame is not modified.
    text_cols : list of str, optional
        Existing fields to clean. Each produces a <column>_clean field.
        Missing requested columns are skipped.
    cat_cols, num_cols : optional
        Reserved arguments; no categorical or numeric processing is done.
    download : bool
        Download the required NLTK resources before cleaning when True.

    Returns
    -------
    pandas.DataFrame
        Original columns plus cleaned text. Non-string values become empty
        text. Cleaning lowercases, removes markup/URLs, expands selected
        contractions, removes punctuation and digits, tokenizes, removes
        English stopwords and applies WordNet lemmatization.

    Notes
    -----
    English stopword removal can remove negation tokens. This documents the
    current experiment pipeline; it does not change its behavior.
    """
    # Download required packages
    if download:
        packages_to_download = [
            "punkt",
            "punkt_tab",
            "stopwords",
            "wordnet",
            "omw-1.4",
        ]

        for paket in packages_to_download:
            nltk.download(paket)

        print("All required packages downloaded")

    def clean_single_text(text):

        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Remove HTML tags and URLS, leaving Space
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Expand common negative contractions before removing punctuation.
        text = text.replace("’", "'")
        text = re.sub(r"\bcan't\b", "can not", text)
        text = re.sub(r"\bwon't\b", "will not", text)
        text = re.sub(r"n't\b", " not", text)

        # Remove punctuation and numbers (?)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)

        tokens = word_tokenize(text)

        # Stop word removal
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [w for w in tokens if not w in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

        return " ".join(lemmatized_tokens)

    df_clean = df.copy()

    # Ensure arguments are lists if None is passed
    text_cols = text_cols or []

    # --- PROCESS TEXT COLUMNS ---
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[f"{col}_clean"] = df_clean[col].apply(clean_single_text)

    return df_clean


if __name__ == "__main__":

    train, _, train_headers, _, _ = load_dataset()
    train_df = pd.DataFrame(train, columns=train_headers)

    ##HOW TO USE THE FUNCTION
    processed_df = preprocess_dataset(
        df=train_df, text_cols=["open_response"]  # replace with our data
    )
    # voluntary display check
    print(processed_df[["open_response", "open_response_clean"]].head())
