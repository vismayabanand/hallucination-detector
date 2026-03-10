import pandas as pd


KEYS = ["sample_id", "sentence_id", "label"]


def process_split(split_name):
    base_df = pd.read_csv(f"data/processed/{split_name}.csv")
    retr_df = pd.read_csv(f"data/features/{split_name}_retrieval_nli.csv")
    self_df = pd.read_csv(f"data/features/{split_name}_selfcheck.csv")

    merged = base_df.merge(retr_df, on=KEYS, how="inner")
    merged = merged.merge(self_df, on=KEYS, how="inner")

    out_path = f"data/features/{split_name}_phase2_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Shape:", merged.shape)


def main():
    for split_name in ["train", "val", "test"]:
        process_split(split_name)


if __name__ == "__main__":
    main()