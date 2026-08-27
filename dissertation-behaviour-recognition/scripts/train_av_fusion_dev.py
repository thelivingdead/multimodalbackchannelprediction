#!/usr/bin/env python3
"""Step C: frozen VideoMAE RGB + audio concat, GOLD DEV only (no TEST).

Reuses existing frozen VideoMAE embeddings under ``data/features/videomae/``.
Does **not** retrain VideoMAE. Compares RGB only / audio only / RGB+audio
with LogisticRegression (seed 42) on nod TRAIN pseudo-labels, scored on
GOLD DEV.

Requires Step A PASS and audio features from Step B (or extracts them).
If embeddings are missing, exits BLOCKED and writes no fabricated F1.

    OMP_NUM_THREADS=1 python scripts/train_av_fusion_dev.py
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_av_fusion_dev.py
"""
from __future__ import annotations

import argparse
import json
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
    EMB_DIR,
    assert_not_locked_out_dir,
    inventory_clip,
    load_features,
    load_videomae_embedding,
    nod_train_dev_ids,
    refuse_test_scoring,
)
from src.clip_metrics import always_predict, choose_dev_threshold  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402

ALIGN_JSON = ROOT / "results" / "audio_alignment_check.json"
OUT_CSV = ROOT / "results" / "audio_visual_fusion_dev.csv"
OUT_MD = ROOT / "results" / "tables" / "multimodal_ablation.md"
OUT_DIR = ROOT / "results" / "audio_visual_fusion_dev"
SEED = 42
MIN_TRAIN = 8
MIN_DEV = 3


def require_step_a(allow: bool) -> None:
    if allow:
        print("NOTE: --allow-without-alignment set (still no TEST).")
        return
    if not ALIGN_JSON.exists():
        raise SystemExit("STOP: Step A missing. Run audio_alignment_check.py. No F1 invented.")
    payload = json.loads(ALIGN_JSON.read_text())
    if str(payload.get("status", "")).upper() != "PASS":
        raise SystemExit(
            f"STOP: Step A is {payload.get('status')}. Blocker: {payload.get('blocker')}. "
            "Do not run fusion. No F1 invented."
        )


def collect(ids: list[str], name: str):
    rgb, aud, y, kept, miss_rgb, miss_aud = [], [], [], [], [], []
    for sid in ids:
        meta = inventory_clip(sid)
        if str(meta.get("split", "")).upper() == "TEST":
            raise SystemExit(f"STOP: {name} contains TEST id {sid}.")
        e = load_videomae_embedding(sid)
        a = load_features(sid)
        if e is None:
            miss_rgb.append(sid)
            continue
        if a is None:
            miss_aud.append(sid)
            continue
        rgb.append(e)
        aud.append(a)
        y.append(int(meta["label"]))
        kept.append(sid)
    if miss_rgb:
        print(f"NOTE: {name}: missing VideoMAE embeddings: {miss_rgb}")
    if miss_aud:
        print(f"NOTE: {name}: missing audio features: {miss_aud}")
    if not rgb:
        return None, None, None, kept, miss_rgb, miss_aud
    return (
        np.stack(rgb).astype(np.float32),
        np.stack(aud).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        kept,
        miss_rgb,
        miss_aud,
    )


def fit_eval(X_tr, y_tr, X_dv, y_dv, seed: int) -> dict:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_dv)[:, 1]
    pred05 = (prob >= 0.5).astype(int)
    m05 = binary_metrics(y_dv, pred05)
    thr, _ = choose_dev_threshold(y_dv, prob, criterion="f1")
    mthr = binary_metrics(y_dv, (prob >= thr).astype(int))
    return {
        "thr0.5": m05,
        "dev_threshold": mthr,
        "threshold": float(thr),
        "prob": prob,
    }


def metric_row(model: str, inputs: str, metrics: dict, n: int, note: str) -> dict:
    return {
        "split": "DEV",
        "protocol": "DEV_ONLY",
        "gold_test_scored": False,
        "model": model,
        "input": inputs,
        "n": n,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "note": note,
    }


def write_md(rows: list[dict], extra: str) -> None:
    lines = [
        "# RGB / audio ablation (GOLD DEV only)",
        "",
        "**DEV ONLY.** GOLD TEST was not scored. Nod TEST headline remains "
        "fine-tuned VideoMAE F1 **0.82** (locked RGB). Shake TEST headline "
        "remains pose rule F1 **0.70** (locked).",
        "",
        "Pose + RGB in the locked tables are **visual representation "
        "experiments** (two encodings of the camera stream), not two sensory "
        "modalities. This table is the first **auditory** stream on the nod "
        "task, fused by concatenating a *frozen* VideoMAE vector with clip "
        "audio statistics. VideoMAE was not retrained.",
        "",
        "| model | input | n | P | R | F1 | bal-acc | TP FP TN FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['input']} | {r['n']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | **{r['f1']:.3f}** | "
            f"{r['balanced_accuracy']:.3f} | "
            f"{r['tp']} {r['fp']} {r['tn']} {r['fn']} |"
        )
    lines += ["", extra, ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


def write_blocked(reason: str) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        "# RGB / audio ablation (GOLD DEV only)\n\n"
        "**BLOCKED — no F1 invented.** GOLD TEST was not scored.\n\n"
        f"{reason}\n"
    )
    pd.DataFrame(
        [
            {
                "split": "DEV",
                "protocol": "DEV_ONLY",
                "gold_test_scored": False,
                "model": "BLOCKED",
                "input": "",
                "n": 0,
                "precision": "",
                "recall": "",
                "f1": "",
                "balanced_accuracy": "",
                "note": reason,
            }
        ]
    ).to_csv(OUT_CSV, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-test", action="store_true", default=False)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--allow-without-alignment", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    refuse_test_scoring(score_test=bool(args.score_test), split=args.split)
    require_step_a(bool(args.allow_without_alignment))
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    assert_not_locked_out_dir(out_dir)
    assert_not_locked_out_dir(OUT_CSV.parent)
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"STOP: refusing TEST-style {out_dir / 'metrics.json'}")
    if OUT_CSV.exists() and not args.force:
        raise SystemExit(
            f"STOP: {OUT_CSV} exists. Pass --force to replace DEV-only fusion "
            "(will not score TEST)."
        )

    set_seed(int(args.seed))
    train_ids, dev_ids = nod_train_dev_ids()

    if not EMB_DIR.exists():
        reason = (
            f"BLOCKED: frozen VideoMAE embeddings dir missing: {EMB_DIR}. "
            "Step C reuses existing embeddings and will not retrain VideoMAE. "
            "On otter they live at data/features/videomae/<sample_id>.npz. "
            "No fusion F1 invented."
        )
        write_blocked(reason)
        raise SystemExit(reason)

    tr = collect(train_ids, "TRAIN")
    dv = collect(dev_ids, "DEV")
    Xr_tr, Xa_tr, y_tr, tr_kept, miss_rgb_tr, miss_aud_tr = tr
    Xr_dv, Xa_dv, y_dv, dv_kept, miss_rgb_dv, miss_aud_dv = dv
    if Xr_tr is None or y_tr is None or len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
        reason = (
            f"BLOCKED: TRAIN usable RGB+audio="
            f"{0 if y_tr is None else len(y_tr)} (need >= {MIN_TRAIN} both "
            f"classes). missing_rgb={miss_rgb_tr} missing_audio={miss_aud_tr}. "
            "No F1 invented."
        )
        write_blocked(reason)
        raise SystemExit(reason)
    if Xr_dv is None or y_dv is None or len(y_dv) < MIN_DEV:
        reason = (
            f"BLOCKED: DEV usable RGB+audio="
            f"{0 if y_dv is None else len(y_dv)} (need >= {MIN_DEV}). "
            f"missing_rgb={miss_rgb_dv} missing_audio={miss_aud_dv}. "
            "No F1 invented."
        )
        write_blocked(reason)
        raise SystemExit(reason)

    always = always_predict(y_dv, 1)
    rgb = fit_eval(Xr_tr, y_tr, Xr_dv, y_dv, int(args.seed))
    aud = fit_eval(Xa_tr, y_tr, Xa_dv, y_dv, int(args.seed))
    fus = fit_eval(
        np.concatenate([Xr_tr, Xa_tr], axis=1),
        y_tr,
        np.concatenate([Xr_dv, Xa_dv], axis=1),
        y_dv,
        int(args.seed),
    )
    n = int(len(y_dv))
    rows = [
        metric_row("always_nod", "none", always, n, "trivial DEV baseline"),
        metric_row(
            "rgb_lr_dev_threshold",
            "frozen VideoMAE 768-D",
            rgb["dev_threshold"],
            n,
            "DEV ONLY; frozen embeddings, no VideoMAE retrain",
        ),
        metric_row(
            "audio_lr_dev_threshold",
            "MFCC/RMS/centroid 30-D",
            aud["dev_threshold"],
            n,
            "DEV ONLY",
        ),
        metric_row(
            "rgb_audio_lr_dev_threshold",
            "concat 768-D + 30-D",
            fus["dev_threshold"],
            n,
            "DEV ONLY early fusion; TEST not scored",
        ),
    ]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    extra = (
        f"TRAIN n={len(y_tr)} (pseudo-labels). DEV n={n} (gold). "
        f"Thresholds chosen on DEV (RGB {rgb['threshold']:.3f}, "
        f"audio {aud['threshold']:.3f}, fusion {fus['threshold']:.3f}). "
        "Text/transcript models were not run (future work). "
        "This is **supervised prediction of the backchannel label associated "
        "with a conversational window**, not anticipatory forecasting."
    )
    write_md(rows, extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        out_dir / "dev_metrics.json",
        {
            "split": "DEV",
            "gold_test_scored": False,
            "n_train": int(len(y_tr)),
            "n_dev": n,
            "train_ids": tr_kept,
            "dev_ids": dv_kept,
            "always_nod": always,
            "rgb": {k: rgb[k] for k in ("thr0.5", "dev_threshold", "threshold")},
            "audio": {k: aud[k] for k in ("thr0.5", "dev_threshold", "threshold")},
            "fusion": {k: fus[k] for k in ("thr0.5", "dev_threshold", "threshold")},
        },
    )
    pd.DataFrame(
        {
            "sample_id": dv_kept,
            "split": "DEV",
            "y_true": y_dv,
            "prob_rgb": rgb["prob"],
            "prob_audio": aud["prob"],
            "prob_fusion": fus["prob"],
        }
    ).to_csv(out_dir / "dev_predictions.csv", index=False)
    print(f"DEV ONLY RGB F1={rgb['dev_threshold']['f1']:.3f} "
          f"audio F1={aud['dev_threshold']['f1']:.3f} "
          f"fusion F1={fus['dev_threshold']['f1']:.3f}")
    print(f"wrote {OUT_CSV} and {OUT_MD} (TEST not scored)")


if __name__ == "__main__":
    main()
