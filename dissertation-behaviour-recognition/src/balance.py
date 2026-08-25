"""TRAIN-set rebalancing for the 75-pos / 5-neg shake collapse test.

Does not touch gold labels. Operates on integer 0/1 arrays already built
from frozen-rule pseudo-labels.
"""
from __future__ import annotations

import numpy as np


def balance_indices(
    y,
    *,
    mode: str = "none",
    ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Return indices into ``y`` after none / subsample / oversample.

    * ``none`` — identity (all rows, original order).
    * ``subsample`` — keep every minority-class row; draw
      ``round(n_minority * ratio)`` majority rows without replacement
      (``ratio=1`` → 1:1). Seeded.
    * ``oversample`` — keep every majority-class row; sample minority
      rows with replacement until the two classes are the same size.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    idx_all = np.arange(n)
    mode = str(mode).strip().lower()
    if mode in ("", "none", "off"):
        return idx_all
    if ratio <= 0:
        raise SystemExit(f"STOP: balance ratio must be > 0, got {ratio}")
    pos = idx_all[y == 1]
    neg = idx_all[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise SystemExit(
            f"STOP: cannot balance TRAIN with a single class "
            f"({int(len(pos))} pos / {int(len(neg))} neg)."
        )
    rng = np.random.RandomState(int(seed))
    if mode == "subsample":
        n_min = min(len(pos), len(neg))
        n_maj_keep = int(round(n_min * float(ratio)))
        n_maj_keep = max(n_maj_keep, 1)
        if len(pos) >= len(neg):
            n_maj_keep = min(n_maj_keep, len(pos))
            keep_pos = rng.choice(pos, size=n_maj_keep, replace=False)
            keep_neg = neg
        else:
            n_maj_keep = min(n_maj_keep, len(neg))
            keep_neg = rng.choice(neg, size=n_maj_keep, replace=False)
            keep_pos = pos
        idx = np.concatenate([keep_pos, keep_neg])
        rng.shuffle(idx)
        return idx
    if mode == "oversample":
        n_maj = max(len(pos), len(neg))
        if len(pos) < n_maj:
            extra = rng.choice(pos, size=n_maj - len(pos), replace=True)
            keep_pos = np.concatenate([pos, extra])
            keep_neg = neg
        elif len(neg) < n_maj:
            extra = rng.choice(neg, size=n_maj - len(neg), replace=True)
            keep_neg = np.concatenate([neg, extra])
            keep_pos = pos
        else:
            keep_pos, keep_neg = pos, neg
        idx = np.concatenate([keep_pos, keep_neg])
        rng.shuffle(idx)
        return idx
    raise SystemExit(
        f"STOP: unknown balance mode {mode!r} "
        "(use none, subsample, or oversample)."
    )


def apply_index_list(items: list, idx: np.ndarray) -> list:
    return [items[int(i)] for i in idx]


def boosted_pos_weight(n_pos: int, n_neg: int, boost: float = 1.0) -> float:
    """BCE ``pos_weight = n_neg / n_pos``, then emphasise the minority class.

    ``boost=1`` is the existing recipe. ``boost>1`` further up-weights the
    rarer class (divides ``pos_weight`` when positives are the majority,
    as in shake 75/5).
    """
    pos = max(int(n_pos), 1)
    neg = max(int(n_neg), 1)
    w = neg / pos
    boost = float(boost)
    if boost <= 0:
        raise SystemExit(f"STOP: pos-weight boost must be > 0, got {boost}")
    if boost == 1.0:
        return float(w)
    if pos >= neg:
        return float(w / boost)
    return float(w * boost)
