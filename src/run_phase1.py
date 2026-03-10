from pathlib import Path
import pandas as pd

from src.dataset import load_raw_dataset, inspect_dataframe
from src.preprocessing import (
    build_sentence_level_dataframe,
    clean_labels,
    split_dataset,
)
from src.baselines import random_baseline, sentence_length_baseline
from src.evaluate import classification_metrics, correlation_metrics
from src.config import RAW_DIR, PROCESSED_DIR


def main():
    raw_file = RAW_DIR / "wikibio_hallucination.json"

    print(f"Loading dataset from: {raw_file}")
    raw_df = load_raw_dataset(raw_file)
    inspect_dataframe(raw_df)

    print("\nBuilding sentence-level dataframe...")
    sentence_df = build_sentence_level_dataframe(raw_df)
    print(sentence_df.head())
    print("Sentence-level shape:", sentence_df.shape)

    print("\nCleaning labels...")
    sentence_df = clean_labels(sentence_df)
    print("After cleaning labels:", sentence_df.shape)
    print(sentence_df["label"].value_counts(dropna=False))

    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(sentence_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\nSaved:")
    print(PROCESSED_DIR / "train.csv")
    print(PROCESSED_DIR / "val.csv")
    print(PROCESSED_DIR / "test.csv")

    print("\nRunning simple baselines on test set...")

    y_true = test_df["label"].values

    rand_scores = random_baseline(len(test_df))
    rand_metrics = classification_metrics(y_true, rand_scores)
    rand_corr = correlation_metrics(y_true, rand_scores)

    len_scores = sentence_length_baseline(test_df["sentence"].tolist())
    len_metrics = classification_metrics(y_true, len_scores)
    len_corr = correlation_metrics(y_true, len_scores)

    print("\nRandom baseline:")
    print({**rand_metrics, **rand_corr})

    print("\nSentence-length baseline:")
    print({**len_metrics, **len_corr})


if __name__ == "__main__":
    main()