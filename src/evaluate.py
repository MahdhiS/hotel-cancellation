import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate(model, X_val, y_val, threshold=None):
    y_true = np.asarray(y_val)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_val)[:, 1]
        default_threshold = 0.5
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_val)
        default_threshold = 0.0
    else:
        raise ValueError("model needs predict_proba or decision_function")

    thr = default_threshold if threshold is None else threshold
    pred = (score >= thr).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]),
    }
    