
import pandas as pd
import re

from features.loading_dataset import load_dataset

def remove_irrelevant_phrases(texts):
    """Remove repetitive administrative phrases from narratives."""

    patterns = [
        # Gratitude and comments about hospital services
        r"\bthe client thanked for servi(?:ce|ces|ces|es)\b"
        r"(?:\s+(?:which|provided|given|by|in|the|hospital|"
        r"especial|especially|nurses?|doctors?|and))*",

        # Misplaced death certificate
        r"\bthe client misplaced death certificate\b",

        # Transfer of death certificate
        r"\bthe client (?:transfer|trnsfer) death certificate "
        r"to their original home(?:\s+\[?place\]?)?",

        # Complaints about services
        r"\bthe client claimed about services\b"
        r"(?:\s+(?:which|provided|by|nurses?|doctors?|"
        r"and|in|the|hospital|especial))*",
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


def remove_narrative_introductions(texts):
    """Remove repetitive narrative introductions."""

    patterns = [
        r"\brespondent explained that\b",
        r"\bthe client said that\b"
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


def inspect_remaining_phrases(texts):

    phrases = [
        "the client thanked",
        "the client misplaced death certificate",
        "the client transfer death certificate",
        "the client said that"
    ]

    for phrase in phrases:

        matches = [
            text for text in texts
            if phrase in text.lower()
        ]

        print(f"\n{'=' * 50}")
        print(f"Phrase: {phrase}")
        print(f"Remaining occurrences: {len(matches)}")

        for text in matches[:5]:
            print("\nExample:")
            print(text)


def main():

    # Load dataset
    train, _, train_headers, _, _ = load_dataset()

    train_df = pd.DataFrame(train, columns=train_headers)

    texts = train_df["open_response"].fillna("").astype(str)

    # Generate three versions
    original_texts = texts.tolist()

    administrative_texts = remove_irrelevant_phrases(
        original_texts
    )

    fully_cleaned_texts = remove_narrative_introductions(
        administrative_texts
    )

    # Count modified narratives
    n_administrative = sum(
        a != b
        for a, b in zip(original_texts, administrative_texts)
    )

    n_fully_cleaned = sum(
        a != b
        for a, b in zip(original_texts, fully_cleaned_texts)
    )

    print("Total narratives:", len(original_texts))
    print("Administrative cleaning:", n_administrative)
    print("Administrative + introductions:", n_fully_cleaned)

    # Check empty narratives
    empty_texts = sum(
        not text.strip()
        for text in fully_cleaned_texts
    )

    print("\nEmpty narratives after cleaning:", empty_texts)

    # Check remaining phrases
    remaining_phrases = [
        "the client thanked",
        "the client misplaced death certificate",
        "the client transfer death certificate",
        "respondent explained that",
        "the client said that"
    ]

    print("\nRemaining phrases after cleaning:")

    for phrase in remaining_phrases:
        count = sum(
            phrase in text.lower()
            for text in fully_cleaned_texts
        )

        print(f"{phrase}: {count}")

    # Show examples of modified narratives
    print("\nExamples of cleaned narratives:")

    shown = 0

    for original, cleaned in zip(original_texts, fully_cleaned_texts):

        if original != cleaned:

            print("\nORIGINAL:")
            print(original)

            print("\nCLEANED:")
            print(cleaned)

            shown += 1

            if shown >= 10:
                break

    inspect_remaining_phrases(fully_cleaned_texts)

if __name__ == "__main__":
    main()
