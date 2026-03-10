from pathlib import Path
import json
import pandas as pd


def load_raw_dataset(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # if dataset is stored as dict of split -> records
            for key, value in data.items():
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    df["source_split"] = key
                    return df
            return pd.json_normalize(data)

    if suffix == ".jsonl":
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported file type: {suffix}")


def inspect_dataframe(df: pd.DataFrame, n: int = 3):
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.head(n).to_dict(orient="records"))
    print("\nDtypes:")
    print(df.dtypes)