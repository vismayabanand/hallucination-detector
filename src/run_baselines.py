import pandas as pd

from src.baselines import (
    random_baseline,
    sentence_length_baseline,
    entity_overlap_baseline,
    sample_count_baseline,
)
from src.evaluate import classification_metrics, correlation_metrics


def evaluate_one(name, y_true, scores):
    metrics = classification_metrics(y_true, scores)
    corr = correlation_metrics(y_true, scores)
    row = {"model": name}
    row.update(metrics)
    row.update(corr)
    return row


def main():
    test_df = pd.read_csv("data/processed/test.csv")
    y_true = test_df["label"].values

    results = []

    rand_scores = random_baseline(len(test_df))
    results.append(evaluate_one("random", y_true, rand_scores))

    len_scores = sentence_length_baseline(test_df["sentence"].tolist())
    results.append(evaluate_one("sentence_length", y_true, len_scores))

    overlap_scores = entity_overlap_baseline(
        test_df["sentence"].tolist(),
        test_df["wiki_bio_text"].tolist(),
    )
    results.append(evaluate_one("wiki_overlap", y_true, overlap_scores))

    sample_scores = sample_count_baseline(test_df["gpt3_text_samples"].tolist())
    results.append(evaluate_one("sample_count", y_true, sample_scores))

    results_df = pd.DataFrame(results)
    print("\n=== BASELINE RESULTS ===")
    print(results_df)

    results_df.to_csv("results/baseline_results.csv", index=False)
    print("\nSaved: results/baseline_results.csv")


if __name__ == "__main__":
    main()