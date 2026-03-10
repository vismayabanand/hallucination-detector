import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME = "potsawee/deberta-v3-large-mnli"


def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text).strip()) if s.strip()]


def make_windows(sentences, window_size=3, stride=1):
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + window_size >= len(sentences):
            break
    return chunks


def build_example_id_to_reference():
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    eval_ds = dataset["evaluation"]

    id_to_ref = {}
    for ex in eval_ds:
        sample_id = ex["wiki_bio_test_idx"]
        id_to_ref[sample_id] = ex["wiki_bio_text"]
    return id_to_ref


def build_evidence_df(sample_ids, id_to_ref, window_size=3, stride=1):
    evidence_rows = []

    for sample_id in sample_ids:
        reference_text = id_to_ref[sample_id]
        sents = split_into_sentences(reference_text)
        chunks = make_windows(sents, window_size=window_size, stride=stride)

        for chunk_id, chunk in enumerate(chunks):
            evidence_rows.append({
                "sample_id": sample_id,
                "chunk_id": chunk_id,
                "passage_text": chunk
            })

    return pd.DataFrame(evidence_rows)


def build_local_indices(evidence_df, retriever):
    sample_to_index = {}

    for sample_id in tqdm(evidence_df["sample_id"].unique(), desc="Building FAISS indices"):
        local_df = evidence_df[evidence_df["sample_id"] == sample_id].reset_index(drop=True)

        passages = local_df["passage_text"].tolist()
        embeddings = retriever.encode(
            passages,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))

        sample_to_index[sample_id] = {
            "index": index,
            "passages": passages
        }

    return sample_to_index


def retrieve_top_k(sentence, sample_id, sample_to_index, retriever, k=5):
    entry = sample_to_index[sample_id]

    query_emb = retriever.encode(
        [sentence],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = entry["index"].search(query_emb, min(k, len(entry["passages"])))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append((float(score), entry["passages"][idx]))

    return results


def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


def nli_probs(premise, hypothesis, tokenizer, model):
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Gilbert’s code assumes contradiction=0, entailment=1 for this model
    return {
        "entailment": float(probs[1]),
        "contradiction": float(probs[0]),
        "neutral": float(probs[2]) if len(probs) > 2 else 0.0
    }


def build_features_for_df(df, sample_to_index, retriever, tokenizer, nli_model, k=5):
    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring sentences"):
        retrieved = retrieve_top_k(
            sentence=row["sentence"],
            sample_id=row["sample_id"],
            sample_to_index=sample_to_index,
            retriever=retriever,
            k=k
        )

        retrieval_scores = [score for score, _ in retrieved]
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []

        for _, passage in retrieved:
            probs = nli_probs(passage, row["sentence"], tokenizer, nli_model)
            entailment_scores.append(probs["entailment"])
            contradiction_scores.append(probs["contradiction"])
            neutral_scores.append(probs["neutral"])

        feature_rows.append({
            "sample_id": row["sample_id"],
            "sentence_id": row["sentence_id"],
            "label": row["label"],

            "top1_retrieval_score": retrieval_scores[0] if retrieval_scores else 0.0,
            "retrieval_mean": float(np.mean(retrieval_scores)) if retrieval_scores else 0.0,
            "retrieval_max": float(np.max(retrieval_scores)) if retrieval_scores else 0.0,

            "entailment_mean": float(np.mean(entailment_scores)) if entailment_scores else 0.0,
            "entailment_max": float(np.max(entailment_scores)) if entailment_scores else 0.0,

            "contradiction_mean": float(np.mean(contradiction_scores)) if contradiction_scores else 0.0,
            "contradiction_max": float(np.max(contradiction_scores)) if contradiction_scores else 0.0,

            "neutral_mean": float(np.mean(neutral_scores)) if neutral_scores else 0.0,
            "neutral_max": float(np.max(neutral_scores)) if neutral_scores else 0.0,
        })

    return pd.DataFrame(feature_rows)


def process_split(split_name, retriever, tokenizer, nli_model, id_to_ref, k=5):
    df = pd.read_csv(f"data/processed/{split_name}.csv")
    sample_ids = sorted(df["sample_id"].unique().tolist())

    evidence_df = build_evidence_df(sample_ids, id_to_ref)
    sample_to_index = build_local_indices(evidence_df, retriever)

    features_df = build_features_for_df(
        df=df,
        sample_to_index=sample_to_index,
        retriever=retriever,
        tokenizer=tokenizer,
        nli_model=nli_model,
        k=k
    )

    out_path = f"data/features/{split_name}_retrieval_nli.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    print("Device:", DEVICE)

    retriever = SentenceTransformer(RETRIEVER_NAME, device=DEVICE)
    tokenizer, nli_model = load_nli_model()
    id_to_ref = build_example_id_to_reference()

    for split_name in ["train", "val", "test"]:
        process_split(split_name, retriever, tokenizer, nli_model, id_to_ref, k=5)


if __name__ == "__main__":
    main()