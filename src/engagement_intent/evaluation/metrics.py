"""Binary-classification metrics shared by every model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def binary_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    labels = np.asarray(targets, dtype=float)
    return float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())


def binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute positive-class metrics with confusion order [[TN, FP], [FN, TP]]."""

    labels = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"Target/probability shapes differ: {labels.shape} vs {scores.shape}")
    predictions = (scores >= threshold).astype(np.uint8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "positive_label": 1,
        "confusion_matrix_order": [["true_negative", "false_positive"], ["false_negative", "true_positive"]],
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "loss": binary_log_loss(labels, scores),
        "samples": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
    }


def select_f1_threshold(
    targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Select the threshold with maximum validation F1, breaking ties toward 0.5."""

    candidates = [binary_metrics(targets, probabilities, float(value)) for value in thresholds]
    best = max(candidates, key=lambda item: (item["f1"], -abs(item["threshold"] - 0.5)))
    return float(best["threshold"]), best

