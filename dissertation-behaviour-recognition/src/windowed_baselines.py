"""Shared machinery for 3 s sliding-window baselines (nod and shake).

Balanced accuracy is the selection criterion and headline metric; intervals
resample whole clips. Rationale is in reports/methods_chapter_draft.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.clip_metrics import clip_binary_metrics

ROOT = Path(__file__).resolve().parents[1]
WINDOW_FRAMES = 75
WINDOWS_PER_CLIP = 29
CLIPS_PER_SPLIT = 15
HEADLINE = "balanced_accuracy"
CRITERIA = ("balanced_accuracy", "f1")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def rule_score_function():
    """Load the frozen amplitude scorer used by the 60 s study."""
    path = ROOT / "scripts" / "run_full_experiment.py"
    spec = importlib.util.spec_from_file_location("run_full_experiment", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"STOP: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rule_score


def load_windows(path: Path, split: str, allowed: set[str]) -> pd.DataFrame:
    """Load one split's window labels and refuse anything off-protocol."""
    df = pd.read_csv(path)
    needed = {
        "window_id",
        "sample_id",
        "split",
        "start_frame_relative",
        "end_frame_relative",
        "label",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"STOP: {path.name} missing {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.upper()
    if (df["split"] != split).any():
        raise SystemExit(f"STOP: {path.name} contains a non-{split} row")
    ids = set(df["sample_id"])
    if ids != allowed:
        raise SystemExit(
            f"STOP: {path.name} {split} ids differ: "
            f"extra={sorted(ids - allowed)}, missing={sorted(allowed - ids)}"
        )
    counts = df.groupby("sample_id").size()
    expected = CLIPS_PER_SPLIT * WINDOWS_PER_CLIP
    if len(df) != expected or not (counts == WINDOWS_PER_CLIP).all():
        raise SystemExit(
            f"STOP: {path.name} is not {CLIPS_PER_SPLIT} clips "
            f"x {WINDOWS_PER_CLIP} windows"
        )
    return df.reset_index(drop=True)


def score_windows(
    df: pd.DataFrame, axis: int, rule_score, pose_dir: Path, cache: dict | None = None
) -> np.ndarray:
    """Amplitude score per window on one rotation axis."""
    from src.pose_cnn import load_npz

    cache = {} if cache is None else cache
    scores: list[float] = []
    for row in df.itertuples(index=False):
        sid = str(row.sample_id)
        if sid not in cache:
            pose_path = pose_dir / f"{sid}.npz"
            if not pose_path.exists():
                raise SystemExit(f"STOP: missing pose file {pose_path}")
            cache[sid] = np.asarray(
                load_npz(pose_path)["rotation_xyz"], dtype=np.float32
            )
        start = int(row.start_frame_relative)
        end = int(row.end_frame_relative)
        chunk = cache[sid][start:end]
        if chunk.shape != (WINDOW_FRAMES, 3):
            raise SystemExit(
                f"STOP: {sid} window {start}:{end} has shape {chunk.shape}"
            )
        scores.append(float(rule_score(chunk, axis)))
    return np.asarray(scores, dtype=float)


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    values = np.unique(np.asarray(scores, dtype=float))
    if not len(values):
        raise SystemExit("STOP: no rule scores to sweep")
    thresholds = [float(np.nextafter(values[0], -np.inf))]
    thresholds.extend(float((a + b) / 2.0) for a, b in zip(values[:-1], values[1:]))
    thresholds.append(float(np.nextafter(values[-1], np.inf)))
    return thresholds


def selection_key(metrics: dict, threshold: float, criterion: str) -> tuple:
    if criterion == "balanced_accuracy":
        return (
            metrics["balanced_accuracy"],
            metrics["precision"],
            metrics["f1"],
            -threshold,
        )
    return (
        metrics["f1"],
        metrics["balanced_accuracy"],
        metrics["precision"],
        -threshold,
    )


def select_dev_threshold(
    y_dev: np.ndarray, scores_dev: np.ndarray, criterion: str = HEADLINE
) -> tuple[float, dict, pd.DataFrame]:
    """Select a threshold using DEV only. Criterion is fixed before TEST."""
    if criterion not in CRITERIA:
        raise SystemExit(f"STOP: unknown selection criterion {criterion!r}")
    y_dev = np.asarray(y_dev, dtype=int)
    scores_dev = np.asarray(scores_dev, dtype=float)
    rows: list[dict] = []
    best = None
    for threshold in candidate_thresholds(scores_dev):
        metrics = clip_binary_metrics(y_dev, (scores_dev >= threshold).astype(int))
        rows.append({"threshold": threshold, **metrics})
        key = selection_key(metrics, threshold, criterion)
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return float(best[1]), dict(best[2]), pd.DataFrame(rows)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR AUC by the step-wise definition. Threshold-free ranking quality."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = y_true[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    previous = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous) * precision))


def clip_bootstrap(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile intervals from resampling whole clips with replacement."""
    sample_ids = np.asarray(sample_ids)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    clips = np.unique(sample_ids)
    index_by_clip = {clip: np.flatnonzero(sample_ids == clip) for clip in clips}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "balanced_accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    n_degenerate = 0
    for _ in range(n_resamples):
        drawn = rng.choice(clips, size=len(clips), replace=True)
        idx = np.concatenate([index_by_clip[clip] for clip in drawn])
        labels = y_true[idx]
        if labels.min() == labels.max():
            n_degenerate += 1
            continue
        metrics = clip_binary_metrics(labels, y_pred[idx])
        for name in collected:
            collected[name].append(float(metrics[name]))
    out: dict = {
        "n_resamples": int(n_resamples),
        "n_clips": int(len(clips)),
        "n_skipped_single_class_resamples": int(n_degenerate),
        "resampling_unit": "clip",
    }
    for name, values in collected.items():
        array = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(array.mean()) if array.size else float("nan"),
            "ci_lower_95": (
                float(np.percentile(array, 2.5)) if array.size else float("nan")
            ),
            "ci_upper_95": (
                float(np.percentile(array, 97.5)) if array.size else float("nan")
            ),
        }
    return out
