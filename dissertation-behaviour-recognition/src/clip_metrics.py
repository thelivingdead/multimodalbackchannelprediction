"""Sklearn-free clip-level P/R/F1 (Mac venv has no sklearn)."""
from __future__ import annotations

import numpy as np


def clip_binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def always_predict(y_true, value: int) -> dict:
    y_true = np.asarray(y_true).astype(int)
    return clip_binary_metrics(y_true, np.full(len(y_true), int(value)))
