import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_prob),
    }


def correlation_metrics(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if len(np.unique(y_score)) <= 1:
        return {"pearson": 0.0, "spearman": 0.0}

    return {
        "pearson": pearsonr(y_true, y_score)[0],
        "spearman": spearmanr(y_true, y_score)[0],
    }