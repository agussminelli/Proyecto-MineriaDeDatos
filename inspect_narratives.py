
import pandas as pd
import re

from loading_dataset import load_dataset


def remove_irrelevant_phrases(texts):

    patterns = [
        r"\bthe client thanked for services?\b",
        r"\bthe client thanked for services? which provided by nurses and doctors in the hospital\b"
    ]

    cleaned_texts = []

    for text in texts:
        cleaned = str(text)

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
                flags=re.IGNORECASE
            )

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned_texts.append(cleaned)

    return cleaned_texts


def main():

    train, _, train_headers, _, _ = load_dataset()

    train_df = pd.DataFrame(train, columns=train_headers)

    texts = train_df["open_response"].fillna("").astype(str)

    # Apply cleaning
    original_texts = texts.tolist()
    cleaned_texts = remove_irrelevant_phrases(original_texts)

    # Compare original and cleaned narratives
    n_modified = 0

    for original, cleaned in zip(original_texts, cleaned_texts):
        if original != cleaned:

            n_modified += 1

            print("\nORIGINAL:")
            print(original[:500])

            print("\nCLEANED:")
            print(cleaned[:500])

    print("\nModified narratives:", n_modified)
    print("Total narratives:", len(original_texts))


if __name__ == "__main__":
    main()
