"""Reusable frame-level and event-level metrics. Headline is not accuracy."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .events import Event, greedy_match


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_frames": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def event_metrics(pred: list[Event], gold: list[Event], iou_thresholds: tuple[float, ...] = (0.10, 0.30, 0.50)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr in iou_thresholds:
        tp, fp, fn = greedy_match(pred, gold, thr)
        m = prf(tp, fp, fn)
        key = f"iou_{thr:.2f}".replace(".", "p")
        out[key] = m
    primary = out["iou_0p30"]
    out["primary_event_f1"] = primary["f1"]
    out["primary_iou"] = 0.30
    return out
