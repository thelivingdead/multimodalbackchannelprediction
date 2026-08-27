#!/usr/bin/env python3
"""Step B: audio-only LogisticRegression on nod TRAIN + GOLD DEV (no TEST).

Features (clip-level): MFCC mean/std (13+13), RMS mean/std, spectral
centroid mean/std. Model: sklearn LogisticRegression, seed 42.

Requires Step A PASS (``results/audio_alignment_check.json`` status PASS)
unless ``--allow-without-alignment`` is set (debug only; still no TEST).

Writes ``results/audio_dev_results.csv`` labelled DEV ONLY. Refuses GOLD TEST
and locked ``results/videomae_finetuned/`` / shake TEST dirs.

    OMP_NUM_THREADS=1 python scripts/train_audio_baseline_dev.py
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_audio_baseline_dev.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_io import (  # noqa: E402
    AUDIO_FEAT_DIR,
    WAV_DIR,
    TARGET_SR,
    assert_not_locked_out_dir,
    extract_audio_features,
    extract_window_wav,
    feature_path,
    inventory_clip,
    load_features,
    load_shard_index,
    load_wav_mono,
    nod_train_dev_ids,
    refuse_test_scoring,
    resample_mono,
    resolve_video_file,
    save_features,
)
from src.clip_metrics import always_predict, choose_dev_threshold  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402

ALIGN_JSON = ROOT / "results" / "audio_alignment_check.json"
OUT_CSV = ROOT / "results" / "audio_dev_results.csv"
OUT_DIR = ROOT / "results" / "audio_dev"
SEED = 42
MIN_TRAIN = 8
MIN_DEV = 3


def require_step_a(allow: bool) -> dict:
    if allow:
        print("NOTE: --allow-without-alignment: Step A not required (still no TEST).")
        return {}
    if not ALIGN_JSON.exists():
        raise SystemExit(
            "STOP: Step A has not been run. "
            "Run scripts/audio_alignment_check.py first. No F1 invented."
        )
    payload = json.loads(ALIGN_JSON.read_text())
    if str(payload.get("status", "")).upper() != "PASS":
        raise SystemExit(
            "STOP: Step A status is "
            f"{payload.get('status')!r} (blocker: {payload.get('blocker')}). "
            "Do not train audio. No F1 invented."
        )
    return payload


def features_for(sample_id: str, index: dict, keep_wav: bool) -> np.ndarray:
    cached = load_features(sample_id)
    if cached is not None and cached.size:
        return cached
    meta = inventory_clip(sample_id)
    if str(meta.get("split", "")).upper() == "TEST":
        raise SystemExit(f"STOP: refused to extract audio for TEST clip {sample_id}.")
    tmp_dir = WAV_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path, _prov, delete_video = resolve_video_file(
        meta["video_id"], index=index, tmp_dir=tmp_dir, keep=False
    )
    wav_path = AUDIO_FEAT_DIR / f"{sample_id}.wav"
    try:
        extract_window_wav(
            video_path,
            wav_path,
            t0_s=float(meta["t0_s"]),
            duration_s=float(meta["duration_s"]),
        )
        y, sr = load_wav_mono(wav_path)
        y16 = resample_mono(y, sr, TARGET_SR)
        feat, stats = extract_audio_features(y16, TARGET_SR)
        save_features(
            sample_id,
            feat,
            {
                "backend": stats.get("backend", ""),
                "src_sr": sr,
                "rms": stats.get("rms", 0.0),
                "duration_s": stats.get("duration_s", 0.0),
            },
        )
        return feat
    finally:
        if delete_video and Path(video_path).exists():
            try:
                Path(video_path).unlink()
            except OSError:
                pass
        if not keep_wav and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass


def stack_split(ids: list[str], index: dict, keep_wav: bool, name: str):
    xs, ys, kept, missing = [], [], [], []
    for sid in ids:
        meta = inventory_clip(sid)
        if str(meta.get("split", "")).upper() == "TEST":
            raise SystemExit(f"STOP: {name} list contains TEST id {sid}.")
        try:
            feat = features_for(sid, index, keep_wav)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: {name} {sid} blocked ({exc}); not fabricating features.")
            missing.append(sid)
            continue
        xs.append(feat)
        ys.append(int(meta["label"]))
        kept.append(sid)
    if missing:
        print(f"NOTE: {name}: {len(missing)} clips have no audio features: {missing}")
    if not xs:
        return None, None, kept, missing
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), kept, missing


def row_for(name: str, metrics: dict, extra: dict) -> dict:
    out = {
        "split": "DEV",
        "protocol": "DEV_ONLY",
        "gold_test_scored": False,
        "model": name,
        "n": extra.get("n"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "note": extra.get("note", "DEV ONLY — not a TEST headline"),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-test", action="store_true", default=False)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument(
        "--allow-without-alignment",
        action="store_true",
        help="debug only; still refuses TEST",
    )
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite DEV-only outputs (never unlocks GOLD TEST dirs)",
    )
    args = parser.parse_args(argv)
    refuse_test_scoring(score_test=bool(args.score_test), split=args.split)
    require_step_a(bool(args.allow_without_alignment))
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_csv = args.out_csv if args.out_csv.is_absolute() else ROOT / args.out_csv
    assert_not_locked_out_dir(out_dir)
    assert_not_locked_out_dir(out_csv.parent)
    if (out_dir / "metrics.json").exists():
        raise SystemExit(
            f"STOP: {out_dir / 'metrics.json'} looks like a TEST artefact. "
            "Audio DEV writes dev_metrics.json only."
        )
    if out_csv.exists() and not args.force:
        raise SystemExit(
            f"STOP: {out_csv} already exists. Pass --force to replace DEV-only "
            "numbers (still will not score TEST)."
        )

    require_step_a(bool(args.allow_without_alignment))
    set_seed(int(args.seed))
    train_ids, dev_ids = nod_train_dev_ids()
    index = load_shard_index()
    AUDIO_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, tr_kept, tr_miss = stack_split(
        train_ids, index, bool(args.keep_wav), "TRAIN"
    )
    X_dv, y_dv, dv_kept, dv_miss = stack_split(
        dev_ids, index, bool(args.keep_wav), "DEV"
    )
    if X_tr is None or len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
        raise SystemExit(
            f"BLOCKED: TRAIN audio usable={0 if X_tr is None else len(y_tr)} "
            f"(need >= {MIN_TRAIN} with both classes). Missing={tr_miss}. "
            "No F1 invented."
        )
    if X_dv is None or len(y_dv) < MIN_DEV:
        raise SystemExit(
            f"BLOCKED: DEV audio usable={0 if X_dv is None else len(y_dv)} "
            f"(need >= {MIN_DEV}). Missing={dv_miss}. No F1 invented."
        )

    always = always_predict(y_dv, 1)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(args.seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_dv)[:, 1]
    pred_05 = (prob >= 0.5).astype(int)
    m05 = binary_metrics(y_dv, pred_05)
    thr, m_thr = choose_dev_threshold(y_dv, prob, criterion="f1")
    pred_thr = (prob >= thr).astype(int)
    m_thr = binary_metrics(y_dv, pred_thr)

    pred_df = pd.DataFrame(
        {
            "sample_id": dv_kept,
            "split": "DEV",
            "y_true": y_dv,
            "prob": prob,
            "pred_thr": pred_thr,
            "pred_0p5": pred_05,
        }
    )
    pred_df.to_csv(out_dir / "dev_predictions.csv", index=False)
    rows = [
        row_for("always_nod", always, {"n": int(len(y_dv)), "note": "trivial DEV baseline"}),
        row_for(
            "audio_lr_thr0.5",
            m05,
            {"n": int(len(y_dv)), "note": "LR default 0.5; DEV ONLY"},
        ),
        row_for(
            "audio_lr_dev_threshold",
            m_thr,
            {
                "n": int(len(y_dv)),
                "note": f"LR threshold {thr:.3f} chosen on DEV; DEV ONLY not TEST",
            },
        ),
    ]
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_csv, index=False)
    dump_json(
        out_dir / "dev_metrics.json",
        {
            "split": "DEV",
            "gold_test_scored": False,
            "seed": int(args.seed),
            "n_train": int(len(y_tr)),
            "n_dev": int(len(y_dv)),
            "train_ids": tr_kept,
            "dev_ids": dv_kept,
            "train_missing": tr_miss,
            "dev_missing": dv_miss,
            "threshold": float(thr),
            "always_nod": always,
            "audio_lr_thr0.5": m05,
            "audio_lr_dev_threshold": m_thr,
            "free_gb": shutil.disk_usage(Path.home()).free / 1024**3,
        },
    )
    try:
        import joblib

        joblib.dump(pipe, out_dir / "audio_lr.joblib")
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: did not save joblib ({exc})")
    print(
        f"DEV ONLY always-nod F1={always['f1']:.3f}  "
        f"audio LR@0.5 F1={m05['f1']:.3f}  "
        f"audio LR@thr={thr:.3f} F1={m_thr['f1']:.3f}"
    )
    print(f"wrote {out_csv} (DEV ONLY; TEST not scored)")


if __name__ == "__main__":
    main()
