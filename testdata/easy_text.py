from sklearn.feature_extraction.text import TfidfVectorizer

def generate_easy_text():
    texts = [
        "fever cough difficulty breathing",
        "high fever cough difficulty breathing",
        "fever persistent cough breathing difficulty",
        "cough fever severe difficulty breathing",

        "diarrhea vomiting dehydration",
        "severe diarrhea vomiting dehydration",
        "diarrhea persistent vomiting dehydration",
        "vomiting diarrhea severe dehydration",

        "injury bleeding road accident",
        "severe injury bleeding road accident",
        "road accident injury heavy bleeding",
        "injury bleeding after road accident",

        "routine vaccination appointment",
    ]

    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(texts).toarray()
    return X_text, texts