import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl


# Solve  SSL-certificate problems (usual for Mac hehe)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required packages
paket_att_ladda_ner = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']

for paket in paket_att_ladda_ner:
    print(f"Laddar ner {paket}...")
    nltk.download(paket)

print("Nedladdning klar! Prova att köra din preprocess-funktion igen.")


def preprocess_dataset(df, text_cols=None, cat_cols=None, num_cols=None):
    """
    Preprocesses a pandas DataFrame by cleaning text columns, 
    imputing missing numerical values, and casting categorical columns.
    """
    
    # Define the text cleaning logic as an inner function
    def clean_single_text(text):
        # Handle empty or missing values safely
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase the text
        text = text.lower()
        
        # 2. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 3. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 4. Remove punctuation and numbers
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        # 5. Tokenization
        tokens = word_tokenize(text)
        
        # 6. Stop word removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if not w in stop_words]
        
        # 7. Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
        
        return " ".join(lemmatized_tokens)

    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Ensure arguments are lists if None is passed
    text_cols = text_cols or []
    cat_cols = cat_cols or []
    num_cols = num_cols or []
    
    # --- 1. PROCESS TEXT COLUMNS ---
    for col in text_cols:
        if col in df_clean.columns:
            # Apply the inner text cleaning function to each row in the column
            # We save it to a new column with the suffix '_clean'
            df_clean[f"{col}_clean"] = df_clean[col].apply(clean_single_text)
            
    # --- 2. PROCESS NUMERICAL COLUMNS ---
    for col in num_cols:
        if col in df_clean.columns:
            # Impute missing numerical values with the column median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # --- 3. PROCESS CATEGORICAL COLUMNS ---
    for col in cat_cols:
        if col in df_clean.columns:
            # Convert to pandas categorical data type for memory efficiency
            df_clean[col] = df_clean[col].astype('category')
            
    return df_clean

##HOW TO USE THE FUNCTION
# 1. Define the columns to be processed
text_columns = ['open_response']

# 3. Apply the preprocessing function
# cat_cols and num_cols are left out, so they default to None and are ignored
processed_df = preprocess_dataset(
    df="" , #replace with our data
    text_cols=text_columns
)

# 4. Display the first few rows to verify the results
# This will print the original text and the newly created cleaned text column
print(processed_df[['open_response', 'open_response_clean']].head())