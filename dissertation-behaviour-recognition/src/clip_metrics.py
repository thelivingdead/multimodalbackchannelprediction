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
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    return clip_binary_metrics(y_true, np.full(len(y_true), int(value)))


COLLAPSE_POS_RATE = 0.85


def collapse_diagnostics(
    y_pred,
    tn: int,
    pos_rate_thr: float = COLLAPSE_POS_RATE,
) -> dict:
    """Flag always-shake style collapse on DEV (sklearn-free).

    Reject if predicted-positive rate ≳ 0.85 **or** TN=0 on DEV
    (always-shake on the negative class). Always-1 F1 is 0.80 on the
    10 pos / 5 neg GOLD DEV split.

    ``tn`` must be the DEV confusion TN (never TEST).
    """
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    n = int(len(y_pred))
    n_pred_pos = int((y_pred == 1).sum())
    rate = float(n_pred_pos / n) if n else 0.0
    reasons = []
    if rate >= pos_rate_thr:
        reasons.append(f"predicted_positive_rate={rate:.3f}>={pos_rate_thr}")
    if n > 0 and int(tn) == 0:
        reasons.append("tn=0 (always-shake on the DEV negatives)")
    collapsed = bool(reasons)
    return {
        "predicted_positive_rate": rate,
        "pred_pos_rate": rate,
        "n_pred_pos": n_pred_pos,
        "n": n,
        "collapse": collapsed,
        "collapsed": collapsed,
        "collapse_reason": "; ".join(reasons),
        "pos_rate_threshold": float(pos_rate_thr),
    }


def choose_dev_threshold(y_true, prob, criterion: str = "f1"):
    """Sweep ``np.linspace(0.2, 0.8, 13)`` on GOLD DEV only.

    Pass DEV labels and DEV probabilities. There is no TEST argument;
    callers must not feed gold TEST into this function.

    Ties: balanced accuracy, then precision, then the lower threshold.
    ``criterion='balanced_accuracy'`` prefers specificity-aware thresholds
    over recall-only F1 (always-shake F1 is 0.80 on this DEV split).
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.shape != prob.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {prob.shape}")
    criterion = str(criterion).strip().lower()
    best = None
    for t in np.linspace(0.2, 0.8, 13):
        m = clip_binary_metrics(y_true, (prob >= t).astype(int))
        if criterion in ("balanced_accuracy", "bacc", "bal"):
            key = (m["balanced_accuracy"], m["precision"], m["f1"], -float(t))
        else:
            key = (m["f1"], m["balanced_accuracy"], m["precision"], -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), m)
    assert best is not None
    return best[1], best[2]
