import re
import ast
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE

nltk.download("punkt", quiet=True)


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str):
    text = normalize_text(text)
    if not text:
        return []
    return nltk.sent_tokenize(text)


def safe_parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [value]

    return [value]


def infer_response_column(df: pd.DataFrame):
    candidates = [
        "response",
        "generated_text",
        "model_output",
        "output",
        "passage",
        "text",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find response column. Available columns: {df.columns.tolist()}")


def infer_prompt_column(df: pd.DataFrame):
    candidates = [
        "prompt",
        "question",
        "input",
        "instruction",
        "context",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_sentence_labels_column(df: pd.DataFrame):
    candidates = [
        "sentence_labels",
        "labels",
        "annotation",
        "annotations",
        "sent_labels",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_sentence_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw examples into sentence-level rows.
    Tries to align sentence labels if present.
    """
    response_col = infer_response_column(df)
    prompt_col = infer_prompt_column(df)
    label_col = infer_sentence_labels_column(df)

    rows = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"] if "sample_id" in df.columns else idx
        prompt = normalize_text(row[prompt_col]) if prompt_col else ""
        response = normalize_text(row[response_col])

        sentences = split_into_sentences(response)
        labels = safe_parse_list(row[label_col]) if label_col else []

        for sent_idx, sentence in enumerate(sentences):
            label = None
            if sent_idx < len(labels):
                label = labels[sent_idx]

            rows.append(
                {
                    "sample_id": sample_id,
                    "sentence_id": sent_idx,
                    "prompt": prompt,
                    "response": response,
                    "sentence": sentence,
                    "label": label,
                }
            )

    sentence_df = pd.DataFrame(rows)

    if sentence_df.empty:
        raise ValueError("Sentence-level dataframe is empty. Check raw dataset format.")

    return sentence_df


def clean_labels(sentence_df: pd.DataFrame) -> pd.DataFrame:
    df = sentence_df.copy()

    label_map = {
        "hallucination": 1,
        "factual": 0,
        "supported": 0,
        "not_supported": 1,
        "yes": 1,
        "no": 0,
        True: 1,
        False: 0,
    }

    def convert_label(x):
        if pd.isna(x):
            return None
        if isinstance(x, str):
            x_norm = x.strip().lower()
            if x_norm in label_map:
                return label_map[x_norm]
            try:
                return int(x_norm)
            except Exception:
                return None
        if isinstance(x, (int, float)):
            if x in [0, 1]:
                return int(x)
        if x in label_map:
            return label_map[x]
        return None

    df["label"] = df["label"].apply(convert_label)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def split_dataset(sentence_df: pd.DataFrame):
    unique_ids = sentence_df["sample_id"].drop_duplicates()

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    relative_val_size = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - relative_val_size),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_df = sentence_df[sentence_df["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = sentence_df[sentence_df["sample_id"].isin(val_ids)].reset_index(drop=True)
    test_df = sentence_df[sentence_df["sample_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df