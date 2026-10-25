"""Detection metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob).ravel() > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
