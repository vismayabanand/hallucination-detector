from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("=== DATASET OVERVIEW ===")
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Full shape:", full_df.shape)

    print("\n=== LABEL DISTRIBUTION ===")
    print(full_df["label"].value_counts())
    print("\nLabel proportions:")
    print(full_df["label"].value_counts(normalize=True))

    print("\n=== RAW ANNOTATION DISTRIBUTION ===")
    print(full_df["annotation_raw"].value_counts())

    full_df["sentence_word_count"] = full_df["sentence"].astype(str).apply(lambda x: len(x.split()))
    full_df["response_word_count"] = full_df["gpt3_text"].astype(str).apply(lambda x: len(x.split()))

    sample_sentence_counts = full_df.groupby("sample_id")["sentence_id"].count()

    print("\n=== LENGTH STATS ===")
    print("Average sentence word count:", round(full_df["sentence_word_count"].mean(), 2))
    print("Average response word count:", round(full_df["response_word_count"].mean(), 2))
    print("Average sentences per biography:", round(sample_sentence_counts.mean(), 2))

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    full_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["Accurate (0)", "Hallucinated (1)"], rotation=0)
    plt.ylabel("Count")
    plt.title("Binary Label Distribution")
    plt.tight_layout()
    plt.savefig("results/figures/label_distribution.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    full_df["sentence_word_count"].plot(kind="hist", bins=30)
    plt.xlabel("Sentence word count")
    plt.ylabel("Frequency")
    plt.title("Sentence Length Distribution")
    plt.tight_layout()
    plt.savefig("results/figures/sentence_length_distribution.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    sample_sentence_counts.plot(kind="hist", bins=20)
    plt.xlabel("Sentences per biography")
    plt.ylabel("Frequency")
    plt.title("Sentences per Biography")
    plt.tight_layout()
    plt.savefig("results/figures/sentences_per_biography.png")
    plt.close()

    print("\nSaved figures:")
    print("results/figures/label_distribution.png")
    print("results/figures/sentence_length_distribution.png")
    print("results/figures/sentences_per_biography.png")


if __name__ == "__main__":
    main()