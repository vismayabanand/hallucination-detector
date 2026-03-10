import ast
import numpy as np


def random_baseline(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.random(n)


def sentence_length_baseline(sentences):
    lengths = np.array([len(str(s).split()) for s in sentences], dtype=float)
    if lengths.max() == lengths.min():
        return np.zeros_like(lengths)
    return (lengths - lengths.min()) / (lengths.max() - lengths.min())


def entity_overlap_baseline(sentences, references):
    """
    Very simple lexical overlap baseline:
    higher overlap with wiki_bio_text -> more likely factual
    convert to hallucination score = 1 - overlap
    """
    scores = []

    for sent, ref in zip(sentences, references):
        sent_tokens = set(str(sent).lower().split())
        ref_tokens = set(str(ref).lower().split())

        if not sent_tokens:
            scores.append(1.0)
            continue

        overlap = len(sent_tokens & ref_tokens) / len(sent_tokens)
        halluc_score = 1.0 - overlap
        scores.append(halluc_score)

    return np.array(scores)


def sample_count_baseline(gpt3_text_samples_col):
    """
    Placeholder baseline from number of sampled generations available.
    This won't be very meaningful, but gives a sanity-check feature shape.
    """
    counts = []

    for item in gpt3_text_samples_col:
        try:
            if isinstance(item, str):
                parsed = ast.literal_eval(item)
            else:
                parsed = item
            counts.append(len(parsed))
        except Exception:
            counts.append(0)

    counts = np.array(counts, dtype=float)
    if counts.max() == counts.min():
        return np.zeros_like(counts)
    return (counts - counts.min()) / (counts.max() - counts.min())