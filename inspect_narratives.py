
import pandas as pd

from loading_dataset import load_dataset


def inspect_phrases(texts, phrases):

    for phrase in phrases:

        mask = texts.str.contains(
            phrase,
            case=False,
            regex=False,
            na=False
        )

        print(f"\nPhrase: {phrase}")
        print(f"Occurrences: {mask.sum()}")
        print(f"Percentage: {mask.mean() * 100:.2f}%")

        for text in texts[mask].head(3):
            print(f"\n{text[:500]}")

def main():

    train, _, train_headers, _, _ = load_dataset()

    train_df = pd.DataFrame(train, columns=train_headers)

    texts = train_df["open_response"].fillna("").astype(str)

    phrases = [
    "thank",
    "hospital",
    "client"
 ]

    inspect_phrases(texts, phrases)


if __name__ == "__main__":
    main()
