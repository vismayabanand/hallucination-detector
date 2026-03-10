import ast
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_sample_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def split_sample_into_sentences(text):
    parts = str(text).replace("\n", " ").split(".")
    return [p.strip() for p in parts if len(p.strip()) > 5]


def sentence_similarity_stats(sentence, sampled_texts, model):
    if not sampled_texts:
        return {
            "selfcheck_mean_sim": 0.0,
            "selfcheck_max_sim": 0.0,
            "selfcheck_support_count": 0.0,
            "selfcheck_support_ratio": 0.0,
        }

    candidate_sents = []
    for txt in sampled_texts:
        candidate_sents.extend(split_sample_into_sentences(txt))

    if not candidate_sents:
        return {
            "selfcheck_mean_sim": 0.0,
            "selfcheck_max_sim": 0.0,
            "selfcheck_support_count": 0.0,
            "selfcheck_support_ratio": 0.0,
        }

    sent_emb = model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)
    cand_emb = model.encode(candidate_sents, convert_to_numpy=True, normalize_embeddings=True)

    sims = cosine_similarity(sent_emb, cand_emb)[0]

    support_threshold = 0.70
    support_count = int(np.sum(sims >= support_threshold))

    return {
        "selfcheck_mean_sim": float(np.mean(sims)),
        "selfcheck_max_sim": float(np.max(sims)),
        "selfcheck_support_count": float(support_count),
        "selfcheck_support_ratio": float(support_count / max(len(candidate_sents), 1)),
    }


def process_split(split_name, model):
    df = pd.read_csv(f"data/processed/{split_name}.csv")

    rows = []
    parsed_samples = [parse_sample_list(x) for x in df["gpt3_text_samples"]]

    for (_, row), samples in tqdm(zip(df.iterrows(), parsed_samples), total=len(df), desc=f"Self-check {split_name}"):
        feats = sentence_similarity_stats(row["sentence"], samples, model)

        rows.append({
            "sample_id": row["sample_id"],
            "sentence_id": row["sentence_id"],
            "label": row["label"],
            **feats
        })

    out_df = pd.DataFrame(rows)
    out_path = f"data/features/{split_name}_selfcheck.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    model = SentenceTransformer(MODEL_NAME)

    for split_name in ["train", "val", "test"]:
        process_split(split_name, model)


if __name__ == "__main__":
    main()