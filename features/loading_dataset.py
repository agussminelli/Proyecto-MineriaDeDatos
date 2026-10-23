import csv
import re
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        headers = next(reader)
        rows = list(reader)

    return headers, rows


def convert_numeric_columns(headers, rows):
    converters = {
        "newid": int,
        "age": float,
        "sex": int,
    }

    for column, convert in converters.items():
        index = headers.index(column)
        for row in rows:
            value = row[index].strip()
            row[index] = convert(value) if value else None


def load_dataset(dataset_dir=None, apply_grouping=True):
    if dataset_dir is None:
        repository_root = Path(__file__).resolve().parents[1]
        dataset_dir = repository_root.parent / "verbal_autopsy"
    else:
        dataset_dir = Path(dataset_dir)
    testdata_path = (dataset_dir / "cleaned_PHMRC_VAI_redacted_free_text.test_blind.csv")
    traindata_path = (dataset_dir / "cleaned_PHMRC_VAI_redacted_free_text.train.csv")
    mapping_path = dataset_dir / "VA_ConsistentRelabelling_updated.txt"

    train_headers, train = read_csv(traindata_path)
    test_headers, test = read_csv(testdata_path)

    if apply_grouping:
        train_headers, train = add_broad_labeling_to_ds(
            train_headers, train, mapping_path
        )

    convert_numeric_columns(train_headers, train)
    convert_numeric_columns(test_headers, test)

    return train, test, train_headers, test_headers, dataset_dir


def add_broad_labeling_to_ds(headers, rows, mapping_path):
    """Return headers and rows with a freshly calculated gs_grouped column.

    Preserves original columns, values and row order.
    Missing reference outcomes receive an empty grouped label.
    Raises for unmapped outcomes.
    """
    mapping = {}
    current_group = None

    def normalize_label(value):
        return " ".join(str(value).split()).casefold()

    # Read the proposed grouping.
    text = Path(mapping_path).read_text(encoding="utf-8-sig")

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.match(r"^\[[^\]]+\]", line):
            current_group = line
            continue

        if current_group is None:
            continue

        key = normalize_label(line)

        if key in mapping and mapping[key] != current_group:
            raise ValueError(f"Outcome appears in multiple groups: {line}")

        mapping[key] = current_group

    if not mapping:
        raise ValueError("No chapter mappings found")

    if "gs_text34" not in headers:
        raise ValueError("Missing column: gs_text34")

    new_headers = list(headers)
    source_index = new_headers.index("gs_text34")

    if "gs_grouped" in new_headers:
        target_index = new_headers.index("gs_grouped")
    else:
        target_index = len(new_headers)
        new_headers.append("gs_grouped")

    new_rows = []
    unknown = set()
    missing_values = {"", "na", "nan", "n/a", "none", "null"}

    for row in rows:
        new_row = list(row)

        if len(new_row) != len(headers):
            raise ValueError("CSV row length does not match headers")

        value = new_row[source_index]
        key = "" if value is None else normalize_label(value)

        if key in missing_values:
            group = ""
        elif key in mapping:
            group = mapping[key]
        else:
            unknown.add(str(value))
            group = ""

        if target_index == len(new_row):
            new_row.append(group)
        else:
            new_row[target_index] = group

        new_rows.append(new_row)

    if unknown:
        raise ValueError(
            "Outcomes missing from the proposed mapping: "
            + ", ".join(sorted(unknown))
        )

    return new_headers, new_rows


if __name__ == "__main__":
    train, test, _, _, _ = load_dataset()
