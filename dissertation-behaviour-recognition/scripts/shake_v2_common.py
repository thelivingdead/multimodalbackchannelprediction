"""Shared helpers for the head-shake v2 (DEV-only) protocol.

Does not touch locked nod or shake TEST artefacts. Collapse is defined on
GOLD DEV predictions only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "results" / "shake" / "v2"
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
LOCKED_RULE = ROOT / "results" / "shake" / "rule_selected_config.json"
PSEUDO_DIR = ROOT / "features" / "pseudo"
GOLD_NPZ = ROOT / "features" / "gold"

# Reject near-always-shake DEV predictors even if F1 looks high
# (always-1 F1 is 0.80 on 10 pos / 5 neg DEV).
COLLAPSE_POS_RATE = 0.85


def clip_binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    n = int(len(y_true))
    return {
        "accuracy": float((tp + tn) / n) if n else 0.0,
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "pred_pos": tp + fp,
        "pred_pos_rate": float((tp + fp) / n) if n else 0.0,
    }


def collapse_verdict(metrics: dict) -> dict:
    """Reject near-always-positive DEV predictors even if F1 looks high.

    GOLD DEV is 10 pos / 5 neg, so always-1 F1 is 0.80. Collapse if
    predicted-positive rate ≳ 0.85 **or** TN=0 (no true negative).
    """
    n = int(metrics.get("n") or (
        int(metrics.get("tp", 0))
        + int(metrics.get("fp", 0))
        + int(metrics.get("tn", 0))
        + int(metrics.get("fn", 0))
    ))
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    tn = int(metrics.get("tn", 0))
    pred_pos_rate = float((tp + fp) / n) if n else 0.0
    reasons: list[str] = []
    if n > 0 and pred_pos_rate >= COLLAPSE_POS_RATE:
        reasons.append(
            f"predicted_positive_rate={pred_pos_rate:.3f}>={COLLAPSE_POS_RATE}"
        )
    if n > 0 and tn == 0:
        reasons.append("tn=0 (always-shake on the DEV negatives)")
    collapsed = bool(reasons)
    return {
        "collapse": collapsed,
        "collapsed": collapsed,
        "pred_pos_rate": pred_pos_rate,
        "predicted_positive_rate": pred_pos_rate,
        "tn": tn,
        "n": n,
        "collapse_reason": "; ".join(reasons),
        "rule": (
            f"reject if pred_pos_rate >= {COLLAPSE_POS_RATE} or tn == 0"
        ),
    }


def write_dev_json_and_preds(out_dir: Path, metrics: dict, pred_df) -> None:
    """Write both filename conventions; never ``metrics.json`` (TEST-style)."""
    from src.utils import dump_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "dev_metrics.json", metrics)
    dump_json(out_dir / "metrics_dev.json", metrics)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pred_df.to_csv(out_dir / "predictions_dev.csv", index=False)


def as_str(value) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, OSError):
            pass
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip()


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def npz_video_id(path: Path) -> str:
    z = load_npz(path)
    if "video_id" not in z:
        raise SystemExit(f"STOP: {path.name} has no video_id")
    return as_str(z["video_id"])


def n_direction_changes(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0
    d = np.diff(x)
    s = np.sign(d)
    s[s == 0] = np.nan
    # ignore flat segments; count sign flips
    valid = s[np.isfinite(s)]
    if valid.size < 2:
        return 0
    return int(np.sum(np.diff(valid) != 0))


def max_abs_step(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(x))))


def savgol_smooth(x: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Centered Savitzky–Golay (numpy). Interior matches scipy savgol_filter."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return x.copy()
    win = int(window)
    if win % 2 == 0:
        win -= 1
    win = min(win, n if n % 2 == 1 else n - 1)
    win = max(3, win)
    if win % 2 == 0:
        win -= 1
    poly = min(int(poly), win - 1)
    half = win // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, N=poly + 1, increasing=True)
    kernel = np.linalg.pinv(A)[0]
    out = np.convolve(x, kernel[::-1], mode="same")
    for i in list(range(half)) + list(range(n - half, n)):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sl = x[lo:hi]
        tt = np.arange(lo, hi, dtype=float) - i
        Ai = np.vander(tt, N=poly + 1, increasing=True)
        coef = np.linalg.lstsq(Ai, sl, rcond=None)[0]
        out[i] = float(coef[0])
    return out


def rule_score(rot: np.ndarray, axis: int, min_frames: int = 5, max_frames: int = 50) -> float:
    """Same half-cycle amplitude detector as ``run_full_experiment.rule_score``.

    Uses a numpy Savitzky–Golay so Mac venv (no scipy) can still audit / label.
    On otter, scipy is not required.
    """
    try:
        from scipy.signal import savgol_filter as _sg
    except ImportError:
        _sg = None
    x = np.asarray(rot[:, axis], dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 11:
        return 0.0
    win = min(11, x.size if x.size % 2 == 1 else x.size - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    if _sg is not None:
        sm = _sg(x, win, 2)
    else:
        sm = savgol_smooth(x, win, 2)
    d = np.diff(sm)
    turns = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    best = 0.0
    n_turns = int(len(turns))
    for i in range(n_turns):
        a = int(turns[i])
        for j in range(i + 1, n_turns):
            b = int(turns[j])
            span = b - a
            if span > max_frames:
                break
            if span < min_frames:
                continue
            amp = float(abs(sm[b] - sm[a]))
            if amp > best:
                best = amp
    if best == 0.0:
        best = float(np.ptp(sm))
    return best
