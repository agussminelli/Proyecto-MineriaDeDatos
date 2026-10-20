import csv
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

def load_dataset(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent.parent / "verbal_autopsy"
    else:
        dataset_dir = Path(dataset_dir)
    testdata_path = dataset_dir / "cleaned_PHMRC_VAI_redacted_free_text.test_blind.csv"
    traindata_path = dataset_dir / "cleaned_PHMRC_VAI_redacted_free_text.train.csv"

    train_headers, train = read_csv(traindata_path)
    test_headers, test = read_csv(testdata_path)
    convert_numeric_columns(train_headers, train)
    convert_numeric_columns(test_headers, test)

    return train, test, train_headers, test_headers, dataset_dir

if __name__ == "__main__":
    train, test = load_dataset()