"""Event matching and window labelling for the nod pipeline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def greedy_match(
    pred: list[tuple[float, float]],
    gold: list[tuple[float, float]],
    iou_thr: float = 0.2,
) -> tuple[int, int, int]:
    """Return tp, fp, fn using greedy IoU matching (each gold used once)."""
    used = set()
    tp = 0
    for p in pred:
        best_j, best = -1, 0.0
        for j, g in enumerate(gold):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_j = v, j
        if best >= iou_thr and best_j >= 0:
            tp += 1
            used.add(best_j)
    fp = len(pred) - tp
    fn = len(gold) - len(used)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def events_from_df(df: pd.DataFrame, video_id: str | None = None, person: str | None = None) -> list[tuple[float, float]]:
    sub = df
    if video_id is not None:
        sub = sub[sub["video_id"].astype(str) == str(video_id)]
    if person is not None and "person" in sub.columns:
        sub = sub[sub["person"].astype(str) == str(person)]
    out = []
    for _, r in sub.iterrows():
        try:
            out.append((float(r.start_time), float(r.end_time)))
        except Exception:
            continue
    return out


def window_overlaps_event(t0: float, t1: float, events: Iterable[tuple[float, float]], min_overlap: float = 0.25) -> int:
    for a, b in events:
        ov = max(0.0, min(t1, b) - max(t0, a))
        if ov >= min_overlap or iou((t0, t1), (a, b)) >= 0.3:
            return 1
    return 0
