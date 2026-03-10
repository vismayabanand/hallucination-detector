from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def map_label(label: str):
    """
    Convert string annotation into binary label.
    0 = factual / accurate
    1 = hallucinated / inaccurate
    """
    if label is None:
        return None

    label = str(label).strip().lower()

    if label == "accurate":
        return 0
    if label in {"minor_inaccurate", "major_inaccurate", "inaccurate"}:
        return 1

    return None


def build_sentence_level_df(dataset_split):
    rows = []

    for row_idx, sample in enumerate(dataset_split):
        sample_id = sample.get("wiki_bio_test_idx", row_idx)
        gpt3_text = sample.get("gpt3_text", "")
        wiki_bio_text = sample.get("wiki_bio_text", "")
        sentences = sample.get("gpt3_sentences", [])
        annotations = sample.get("annotation", [])
        sampled_generations = sample.get("gpt3_text_samples", [])

        # safety check
        if len(sentences) != len(annotations):
            print(f"Skipping sample_id={sample_id} because sentence/annotation lengths mismatch.")
            continue

        for sent_id, (sentence, ann) in enumerate(zip(sentences, annotations)):
            binary_label = map_label(ann)

            rows.append({
                "sample_id": sample_id,
                "sentence_id": sent_id,
                "gpt3_text": gpt3_text,
                "wiki_bio_text": wiki_bio_text,
                "sentence": sentence,
                "annotation_raw": ann,
                "label": binary_label,
                "gpt3_text_samples": sampled_generations,
            })

    df = pd.DataFrame(rows)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    return df


def split_by_sample_id(df):
    unique_ids = df["sample_id"].drop_duplicates().tolist()

    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )

    train_df = df[df["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["sample_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["sample_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    eval_split = dataset["evaluation"]

    print("Loaded split:", eval_split)
    df = build_sentence_level_df(eval_split)

    print("\nSentence-level dataframe shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nRaw annotation distribution:")
    print(df["annotation_raw"].value_counts(dropna=False))
    print("\nBinary label distribution:")
    print(df["label"].value_counts(dropna=False))

    train_df, val_df, test_df = split_by_sample_id(df)

    print("\nTrain shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("\nSaved:")
    print("data/processed/train.csv")
    print("data/processed/val.csv")
    print("data/processed/test.csv")


if __name__ == "__main__":
    main()