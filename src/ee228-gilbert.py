import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import faiss
from tqdm.auto import tqdm

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print("Device:", device)

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
eval_ds = dataset["evaluation"]

print("Examples:", len(eval_ds))
print(eval_ds[0].keys())

rows = []

for ex_id, ex in enumerate(eval_ds):
    reference_text = ex["wiki_bio_text"]
    sentences = ex["gpt3_sentences"]
    annotations = ex["annotation"]

    for sent_id, sent in enumerate(sentences):
        label = annotations[sent_id] if sent_id < len(annotations) else None

        rows.append({
            "example_id": ex_id,
            "sentence_id": sent_id,
            "sentence_text": sent,
            "label": label,
            "reference_text": reference_text
        })

sent_df = pd.DataFrame(rows)
sent_df.head(60)

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def make_windows(sentences, window_size=3, stride=1):
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i:i+window_size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + window_size >= len(sentences):
            break
    return chunks

evidence_rows = []

for ex_id, ex in enumerate(eval_ds):
    sents = split_into_sentences(ex["wiki_bio_text"])
    chunks = make_windows(sents, window_size=3, stride=1)

    for chunk_id, chunk in enumerate(chunks):
        evidence_rows.append({
            "example_id": ex_id,
            "chunk_id": chunk_id,
            "passage_text": chunk
        })

evidence_df = pd.DataFrame(evidence_rows)
print(evidence_df.shape)
evidence_df.head(60)

retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

example_to_index = {}

for ex_id in evidence_df["example_id"].unique():
    local_df = evidence_df[evidence_df["example_id"] == ex_id].reset_index(drop=True)

    embeddings = retriever.encode(
        local_df["passage_text"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    example_to_index[ex_id] = {
        "index": index,
        "passages": local_df["passage_text"].tolist()
    }

def retrieve_top_k(sentence, example_id, k=5):
    entry = example_to_index[example_id]

    query_emb = retriever.encode(
        [sentence],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = entry["index"].search(query_emb, min(k, len(entry["passages"])))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append((float(score), entry["passages"][idx]))

    return results

model_name = "potsawee/deberta-v3-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
nli_model.eval()

print(nli_model.config.id2label)

def nli_probs(premise, hypothesis):
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    return {
        "entailment": float(probs[1]),
        "contradiction": float(probs[0]),
    }

def build_features(row, k=5):
    retrieved = retrieve_top_k(row["sentence_text"], row["example_id"], k=k)

    retrieval_scores = [score for score, _ in retrieved]
    entailment_scores = []
    contradiction_scores = []

    for _, passage in retrieved:
        probs = nli_probs(passage, row["sentence_text"])
        entailment_scores.append(probs["entailment"])
        contradiction_scores.append(probs["contradiction"])

    return {
        "example_id": row["example_id"],
        "sentence_id": row["sentence_id"],
        "sentence_text": row["sentence_text"],
        "label": row["label"],

        "top1_retrieval_score": retrieval_scores[0] if retrieval_scores else 0.0,
        "retrieval_mean": float(np.mean(retrieval_scores)) if retrieval_scores else 0.0,

        "entailment_mean": float(np.mean(entailment_scores)) if entailment_scores else 0.0,
        "entailment_max": float(np.max(entailment_scores)) if entailment_scores else 0.0,

        "contradiction_mean": float(np.mean(contradiction_scores)) if contradiction_scores else 0.0,
        "contradiction_max": float(np.max(contradiction_scores)) if contradiction_scores else 0.0,
    }

feature_rows = []

for _, row in tqdm(sent_df.iterrows(), total=len(sent_df)):
    feature_rows.append(build_features(row, k=5))

features_df = pd.DataFrame(feature_rows)
features_df.head()

features_df.to_csv("sentence_features.csv", index=False)
print("Saved: sentence_features.csv")

features_df.groupby("label")[["entailment_mean", "contradiction_mean"]].mean()