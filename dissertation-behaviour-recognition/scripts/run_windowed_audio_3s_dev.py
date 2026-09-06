#!/usr/bin/env python3
"""DEV 3 s audio-only nod experiment. TEST is never read.

Reuses the existing 30-D MFCC summary (13 mean + 13 std + RMS + centroid)
and a leave-one-clip-out logistic regression. Same windows and labels as
nod_windows_dev.csv.

    python3 scripts/run_windowed_audio_3s_dev.py --audit-only
    python3 scripts/run_windowed_audio_3s_dev.py --extract --train --figures

Otter::

    bash scripts/run_windowed_audio_3s_otter.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.audio_io import (  # noqa: E402
    FEATURE_DIM,
    TARGET_SR,
    extract_audio_features,
    extract_window_wav,
    inventory_clip,
    load_shard_index,
    load_wav_mono,
    refuse_test_scoring,
    resample_mono,
    resolve_video_file,
)
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    load_windows,
)

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
OUT = ROOT / "results" / "windowed_dev" / "audio_3s"
CACHE = ROOT / "data" / "features" / "audio_windowed_dev"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
SEED = 42
FIXED_THRESHOLD = 0.5
L2 = 1e-2
NEWTON_STEPS = 40
FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(13)]
    + [f"mfcc{i}_std" for i in range(13)]
    + ["rms_mean", "rms_std", "centroid_mean", "centroid_std"]
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(x: np.ndarray, y: np.ndarray, pos_weight: float, l2: float = L2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    xb = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    n, d = xb.shape
    w = np.zeros(d, dtype=float)
    sw = np.where(y == 1.0, float(pos_weight), 1.0)
    sw = sw / max(float(sw.mean()), 1e-8)
    eye = np.eye(d)
    eye[0, 0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = sigmoid(xb @ w)
        resid = sw * (p - y)
        grad = xb.T @ resid / n
        grad[1:] += l2 * w[1:]
        s = sw * p * (1.0 - p)
        hess = (xb.T * s) @ xb / n
        hess = hess + l2 * eye + 1e-8 * np.eye(d)
        try:
            w = w - np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            w = w - 0.1 * grad
    return w


def scale_train_only(train: np.ndarray, held: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (held - mean) / std, mean, std


def _refuse_test_frame(frame: pd.DataFrame) -> None:
    ids = set(frame["sample_id"].astype(str))
    if ids & TEST_IDS:
        raise SystemExit("STOP: TEST id in DEV audio table")
    if ids != DEV_IDS:
        raise SystemExit("STOP: audio table is not the 15 DEV clips")


def load_dev_windows() -> pd.DataFrame:
    if "test" in WINDOWS.name.lower():
        raise SystemExit("STOP: refused TEST window file")
    frame = load_windows(WINDOWS, "DEV", DEV_IDS)
    _refuse_test_frame(frame)
    return frame.reset_index(drop=True)


def write_audit(frame: pd.DataFrame, statuses: list[str], out_dir: Path) -> pd.DataFrame:
    audit = pd.DataFrame(
        {
            "clip_id": frame["sample_id"].astype(str),
            "window_id": frame["window_id"].astype(str),
            "window_start": frame["start_sec"].astype(float),
            "window_end": frame["end_sec"].astype(float),
            "human_nod_label": frame["label"].astype(int),
            "split": "DEV",
            "audio_status": statuses,
        }
    )
    if set(audit["clip_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id leaked into audio audit")
    n_ok = int((audit["audio_status"] == "ok").sum())
    n_fail = int((audit["audio_status"] == "failed").sum())
    n_miss = int((audit["audio_status"] == "not_extracted").sum())
    pos = int(audit["human_nod_label"].sum())
    n = len(audit)
    summary = {
        "development_only": True,
        "test_read": False,
        "n_clips": int(audit["clip_id"].nunique()),
        "total_windows": n,
        "positive_windows": pos,
        "negative_windows": n - pos,
        "prevalence": float(pos / n) if n else 0.0,
        "missing_or_failed_windows": n_fail + n_miss,
        "n_ok": n_ok,
        "n_failed": n_fail,
        "n_not_extracted": n_miss,
        "duration_s": {
            "min": float((audit["window_end"] - audit["window_start"]).min()),
            "max": float((audit["window_end"] - audit["window_start"]).max()),
            "mean": float((audit["window_end"] - audit["window_start"]).mean()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False)
    dump_json(out_dir / "dataset_audit.json", summary)
    table = pd.DataFrame(
        [
            {
                "split": "DEV",
                "clips": summary["n_clips"],
                "windows": n,
                "positive": pos,
                "negative": n - pos,
                "prevalence": summary["prevalence"],
                "failed": summary["missing_or_failed_windows"],
            }
        ]
    )
    table.to_csv(out_dir / "tables" / "table1_dataset.csv", index=False)
    latex = (
        "split & clips & windows & positive & negative & prevalence & failed \\\\\n"
        f"DEV & {summary['n_clips']} & {n} & {pos} & {n - pos} & "
        f"{summary['prevalence']:.3f} & {summary['missing_or_failed_windows']} \\\\\n"
    )
    (out_dir / "tables" / "table1_dataset.tex").write_text(latex)
    print(
        f"AUDIT DEV clips={summary['n_clips']} windows={n} "
        f"pos={pos} neg={n - pos} prev={summary['prevalence']:.3f} "
        f"failed_or_missing={summary['missing_or_failed_windows']}"
    )
    return audit


def extract_features(frame: pd.DataFrame, keep_video: bool) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    index = load_shard_index()
    statuses = ["not_extracted"] * len(frame)
    tmp_dir = CACHE / "tmp_video"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clip_cache: dict[str, tuple[np.ndarray, int, dict]] = {}
    for i, rec in enumerate(frame.itertuples(index=False)):
        sid = str(rec.sample_id)
        if sid in TEST_IDS:
            raise SystemExit(f"STOP: TEST id {sid}")
        feat_path = CACHE / f"{rec.window_id}.npz"
        if feat_path.exists():
            statuses[i] = "ok"
            continue
        try:
            if sid not in clip_cache:
                meta = inventory_clip(sid)
                if str(meta.get("split", "")).upper() == "TEST":
                    raise SystemExit(f"STOP: inventory returned TEST {sid}")
                video_path, _, delete_video = resolve_video_file(
                    meta["video_id"], index=index, tmp_dir=tmp_dir, keep=keep_video
                )
                wav_path = CACHE / f"{sid}_clip.wav"
                extract_window_wav(
                    video_path,
                    wav_path,
                    t0_s=float(meta["t0_s"]),
                    duration_s=float(meta["duration_s"]),
                )
                y, sr = load_wav_mono(wav_path)
                y16 = resample_mono(y, sr, TARGET_SR)
                clip_cache[sid] = (y16, TARGET_SR, meta)
                if delete_video and Path(video_path).exists() and not keep_video:
                    try:
                        Path(video_path).unlink()
                    except OSError:
                        pass
            y16, sr, _meta = clip_cache[sid]
            s0 = int(round(float(rec.start_sec) * sr))
            s1 = int(round(float(rec.end_sec) * sr))
            chunk = y16[s0:s1]
            if chunk.size < sr // 4:
                raise RuntimeError(f"short slice {chunk.size} samples")
            feat, stats = extract_audio_features(chunk, sr)
            np.savez(
                feat_path,
                features=feat.astype(np.float32),
                window_id=str(rec.window_id),
                sample_id=sid,
                backend=np.asarray(stats.get("backend", "")),
            )
            statuses[i] = "ok"
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: {rec.window_id} audio failed ({exc})")
            statuses[i] = "failed"
        if (i + 1) % 29 == 0:
            print(f"extract {sid} done", flush=True)
    return statuses


def load_feature_matrix(frame: pd.DataFrame, statuses: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    keep = []
    for rec, status in zip(frame.itertuples(index=False), statuses):
        if status != "ok":
            keep.append(False)
            rows.append(np.full(FEATURE_DIM, np.nan, dtype=np.float32))
            continue
        path = CACHE / f"{rec.window_id}.npz"
        with np.load(path, allow_pickle=True) as z:
            feat = np.asarray(z["features"], dtype=np.float32).reshape(-1)
        if feat.size != FEATURE_DIM:
            raise SystemExit(f"STOP: {rec.window_id} is {feat.size}-D, expected {FEATURE_DIM}")
        rows.append(feat)
        keep.append(True)
    return np.stack(rows), np.asarray(keep, dtype=bool)


def train_loco(frame: pd.DataFrame, x: np.ndarray, keep: np.ndarray) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["sample_id"].astype(str).to_numpy()
    if set(ids) & TEST_IDS:
        raise SystemExit("STOP: TEST id at train time")
    usable = keep & np.isfinite(x).all(axis=1)
    oof = np.full(len(y), np.nan)
    folds = []
    for held in sorted(set(ids)):
        train = usable & (ids != held)
        held_m = usable & (ids == held)
        if train.sum() < 8 or held_m.sum() == 0 or len(set(y[train])) < 2:
            folds.append({"held_out_clip": held, "skipped": True})
            continue
        x_tr, x_h, mean, std = scale_train_only(x[train], x[held_m])
        n_pos = max(int((y[train] == 1).sum()), 1)
        n_neg = max(int((y[train] == 0).sum()), 1)
        w = fit_logreg(x_tr, y[train], n_neg / n_pos)
        xb = np.column_stack([np.ones(x_h.shape[0]), x_h])
        oof[held_m] = sigmoid(xb @ w)
        folds.append(
            {
                "held_out_clip": held,
                "skipped": False,
                "n_train": int(train.sum()),
                "n_held": int(held_m.sum()),
                "train_mean": [float(v) for v in mean],
                "train_std": [float(v) for v in std],
            }
        )
        print(f"fold {held} train={train.sum()} held={held_m.sum()}", flush=True)
    scored = np.isfinite(oof)
    if scored.sum() < 30:
        raise SystemExit("STOP: too few OOF audio windows")
    pred = np.zeros(len(y), dtype=int)
    pred[scored] = (oof[scored] >= FIXED_THRESHOLD).astype(int)
    metrics = clip_binary_metrics(y[scored], pred[scored])
    boot = clip_bootstrap(ids[scored], y[scored], pred[scored])
    pr_auc = average_precision(y[scored], oof[scored])
    pred_df = pd.DataFrame(
        {
            "window_id": frame["window_id"].astype(str),
            "sample_id": ids,
            "split": "DEV",
            "label": y,
            "audio_ok": usable.astype(int),
            "oof_probability": oof,
            "pred_at_0.5": pred,
        }
    )
    return {
        "pred_df": pred_df,
        "metrics": metrics,
        "boot": boot,
        "pr_auc": pr_auc,
        "folds": folds,
        "n_scored": int(scored.sum()),
        "always_no": always_predict(y[scored], 0),
        "always_yes": always_predict(y[scored], 1),
    }


def write_tables(pack: dict, out_dir: Path) -> None:
    m = pack["metrics"]
    b = pack["boot"]["balanced_accuracy"]
    row = {
        "model": "audio MFCC LR",
        "ba": m["balanced_accuracy"],
        "ci_lo": b["ci_lower_95"],
        "ci_hi": b["ci_upper_95"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "pr_auc": pack["pr_auc"],
        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"],
    }
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(tables / "table2_dev_audio.csv", index=False)
    (tables / "table2_dev_audio.tex").write_text(
        "model & BA & 95\\% CI & F1 & P & R & PR-AUC & TP & FP & TN & FN \\\\\n"
        f"audio MFCC LR & {row['ba']:.3f} & [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] & "
        f"{row['f1']:.3f} & {row['precision']:.3f} & {row['recall']:.3f} & "
        f"{row['pr_auc']:.3f} & {row['tp']} & {row['fp']} & {row['tn']} & {row['fn']} \\\\\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    refuse_test_scoring(score_test=False, split="dev")
    if not (args.audit_only or args.extract or args.train or args.figures):
        args.audit_only = True
    out = assert_unlocked_out_dir(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    frame = load_dev_windows()
    print("WINDOWED AUDIO 3s  DEV only  TEST not read")
    print(f"representation: {FEATURE_DIM}-D MFCC summary at {TARGET_SR} Hz mono")
    print(
        "one sentence: each 3 s mix is 13 MFCC means, 13 MFCC stds, "
        "RMS mean/std, and spectral centroid mean/std."
    )
    statuses = ["not_extracted"] * len(frame)
    if args.extract:
        statuses = extract_features(frame, bool(args.keep_video))
    else:
        statuses = [
            "ok" if (CACHE / f"{wid}.npz").exists() else "not_extracted"
            for wid in frame["window_id"].astype(str)
        ]
    write_audit(frame, statuses, out)
    if args.audit_only and not args.train:
        print(f"wrote {out / 'dataset_audit.csv'}")
        print("STOP before TEST. Extract+train on Otter.")
        return
    if args.train:
        x, keep = load_feature_matrix(frame, statuses)
        pack = train_loco(frame, x, keep)
        pack["pred_df"].to_csv(out / "predictions.csv", index=False)
        write_tables(pack, out)
        dump_json(
            out / "metrics.json",
            {
                "protocol": "windowed_nod_3s_audio_mfcc_loco",
                "development_only": True,
                "test_scored": False,
                "test_read": False,
                "seed": args.seed,
                "threshold": FIXED_THRESHOLD,
                "threshold_selection": "fixed 0.5; not chosen on held-out clips",
                "feature_dim": FEATURE_DIM,
                "feature_names": FEATURE_NAMES,
                "sample_rate_hz": TARGET_SR,
                "normalisation": "train-fold z-score over windows",
                "model": "weighted logistic regression, L2=1e-2, no sklearn required",
                "input_shape": f"(n_windows, {FEATURE_DIM})",
                "representation": (
                    "30-D summary of 3 s container soundtrack: "
                    "13 MFCC means, 13 MFCC stds, RMS mean/std, centroid mean/std"
                ),
                "n_windows_scored": pack["n_scored"],
                "at_fixed_threshold_0.5": pack["metrics"],
                "pr_auc_out_of_fold": pack["pr_auc"],
                "clip_bootstrap_at_0.5": pack["boot"],
                "always_no": pack["always_no"],
                "always_yes": pack["always_yes"],
                "folds": pack["folds"],
                "chance_balanced_accuracy": 0.5,
            },
        )
        dump_json(
            out / "config.json",
            {
                "seed": args.seed,
                "target_sr": TARGET_SR,
                "n_mfcc": 13,
                "stft_window_s": 0.025,
                "stft_hop_s": 0.010,
                "feature_dim": FEATURE_DIM,
                "classifier": "logreg_newton",
                "class_weight": "n_neg/n_pos on train clips",
                "threshold": FIXED_THRESHOLD,
                "windows": str(WINDOWS.relative_to(ROOT)),
                "test_read": False,
            },
        )
        m = pack["metrics"]
        b = pack["boot"]["balanced_accuracy"]
        print(
            f"AUDIO-ONLY DEV BA {m['balanced_accuracy']:.3f} "
            f"[{b['ci_lower_95']:.3f}, {b['ci_upper_95']:.3f}]  "
            f"F1 {m['f1']:.3f}  PR {pack['pr_auc']:.3f}  "
            f"TP{m['tp']} FP{m['fp']} TN{m['tn']} FN{m['fn']}"
        )
        if m["balanced_accuracy"] < 0.53 and b["ci_lower_95"] <= 0.5:
            print("Audio is at chance on DEV. Do not fuse. Do not score TEST.")
        print("TEST not loaded.")
    if args.figures:
        from plot_windowed_audio_3s_dev import write_figures

        write_figures(out)
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
