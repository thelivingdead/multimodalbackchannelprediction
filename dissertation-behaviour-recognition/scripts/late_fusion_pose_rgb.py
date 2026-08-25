#!/usr/bin/env python3
"""Late fusion of pose stream + VideoMAE probs (DEV weights, TEST once).

New directory only::

    results/shake/fusion_pose_rgb/

Pose stream
-----------
The locked shake 1D CNN ``predictions.csv`` is TEST-only and the CNN
checkpoint was never saved, so DEV CNN probabilities cannot be recovered
without retraining the locked run. Fusion therefore uses the **frozen
shake amplitude-rule scores** on DEV+TEST (already committed), min-max
scaled from DEV only, plus VideoMAE probabilities.

If a later CSV has DEV+TEST ``prob`` for the CNN, pass
``--pose-preds results/shake/cnn/predictions.csv`` (must include a split
column). Do not overwrite ``results/shake/cnn/``.

RGB stream: ``results/shake/videomae_finetuned/predictions.csv`` (has DEV).

Grid: fusion weight w ∈ {0.0, 0.1, …, 1.0} and threshold on DEV F1
(ties → balanced accuracy). TEST scored once.

Otter95 or Mac (CPU)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/late_fusion_pose_rgb.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.clip_metrics import clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
OUT_DIR = ROOT / "results" / "shake" / "fusion_pose_rgb"
RGB_PREDS = ROOT / "results" / "shake" / "videomae_finetuned" / "predictions.csv"
RULE_DEV = ROOT / "results" / "shake" / "rule_dev_predictions.csv"
RULE_TEST = ROOT / "results" / "shake" / "rule_test_predictions.csv"
CNN_PREDS = ROOT / "results" / "shake" / "cnn" / "predictions.csv"


def _idcol(df: pd.DataFrame) -> str:
    for c in ("sample_id", "clip_id"):
        if c in df.columns:
            return c
    raise SystemExit(f"STOP: no id column in {list(df.columns)}")


def load_prob_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cid = _idcol(df)
    df = df.rename(columns={cid: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.upper()
    return df


def rule_as_frame() -> pd.DataFrame:
    dev = pd.read_csv(RULE_DEV)
    tes = pd.read_csv(RULE_TEST)
    dev["split"] = "DEV"
    tes["split"] = "TEST"
    both = pd.concat([dev, tes], ignore_index=True)
    both["sample_id"] = both["sample_id"].astype(str)
    return both


def scale_dev(dev_scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    lo = float(np.min(dev_scores))
    hi = float(np.max(dev_scores))
    if hi <= lo:
        return np.full_like(all_scores, 0.5, dtype=float)
    x = (all_scores - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--rgb-preds", type=Path, default=RGB_PREDS)
    parser.add_argument(
        "--pose-preds",
        type=Path,
        default=None,
        help="optional CSV with sample_id, prob, split (DEV+TEST). "
             "Default: frozen shake rule scores.",
    )
    parser.add_argument("--gold-csv", type=Path, default=SHAKE_GOLD)
    parser.add_argument("--label-col", default="shake_label")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import check_split_leakage

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out = out.resolve()
    check_split_leakage.assert_unlocked_out_dir(out)
    if (out / "metrics.json").exists() and not args.force:
        raise SystemExit(
            f"STOP: {out / 'metrics.json'} exists — fusion TEST scored once."
        )

    gold_csv = args.gold_csv if args.gold_csv.is_absolute() else ROOT / args.gold_csv
    gold = pd.read_csv(gold_csv)
    gold["split"] = gold["split"].astype(str).str.upper()
    gold["sample_id"] = gold["sample_id"].astype(str)
    ymap = dict(
        zip(
            gold["sample_id"],
            pd.to_numeric(gold[args.label_col], errors="coerce"),
        )
    )

    rgb_path = args.rgb_preds if args.rgb_preds.is_absolute() else ROOT / args.rgb_preds
    rgb = load_prob_frame(rgb_path)
    if "prob" not in rgb.columns or "split" not in rgb.columns:
        raise SystemExit(
            f"STOP: {rgb_path} needs prob + split (DEV and TEST). "
            "Locked frozen-head predictions.csv is TEST-only and cannot "
            "tune fusion."
        )

    pose_source = "shake_amplitude_rule_score_minmax_on_DEV"
    pose_path_used = f"{RULE_DEV.name}+{RULE_TEST.name}"
    if args.pose_preds is not None:
        pp = args.pose_preds if args.pose_preds.is_absolute() else ROOT / args.pose_preds
        pose = load_prob_frame(pp)
        if "split" not in pose.columns or not (pose.split == "DEV").any():
            raise SystemExit(
                f"STOP: {pp} has no DEV rows. Locked shake CNN "
                "predictions.csv is TEST-only; checkpoint was never saved. "
                "Omit --pose-preds to fuse the frozen amplitude rule instead."
            )
        if "prob" not in pose.columns:
            raise SystemExit(f"STOP: {pp} has no prob column")
        pose["pose_prob"] = pose["prob"].astype(float)
        pose_source = str(pp)
        pose_path_used = str(pp)
    else:
        pose = rule_as_frame()
        if "score" not in pose.columns:
            raise SystemExit("STOP: rule predictions need a score column")
        dev_sc = pose.loc[pose.split == "DEV", "score"].to_numpy(float)
        pose["pose_prob"] = scale_dev(dev_sc, pose["score"].to_numpy(float))

    rgb = rgb.rename(columns={"prob": "rgb_prob"})
    merged = gold[["sample_id", "split"]].merge(
        rgb[["sample_id", "split", "rgb_prob"]],
        on=["sample_id", "split"],
        how="inner",
    ).merge(
        pose[["sample_id", "split", "pose_prob"]],
        on=["sample_id", "split"],
        how="inner",
    )
    merged["label"] = merged["sample_id"].map(ymap)
    if merged["label"].isna().any():
        raise SystemExit("STOP: fusion ids missing gold labels")
    merged["label"] = merged["label"].astype(int)

    dev = merged[merged.split == "DEV"]
    tes = merged[merged.split == "TEST"]
    if len(dev) < 3 or len(tes) < 3:
        raise SystemExit(
            f"STOP: fusion needs DEV+TEST (got {len(dev)}/{len(tes)}). "
            f"RGB={rgb_path}"
        )

    best = None
    rows = []
    for w in np.linspace(0.0, 1.0, 11):
        p_dv = w * dev["rgb_prob"].to_numpy(float) + (
            1.0 - w
        ) * dev["pose_prob"].to_numpy(float)
        for t in np.linspace(0.2, 0.8, 13):
            m = clip_binary_metrics(dev["label"], (p_dv >= t).astype(int))
            rec = {
                "w_rgb": float(w),
                "threshold": float(t),
                **m,
            }
            rows.append(rec)
            if best is None or m["f1"] > best["f1"] or (
                m["f1"] == best["f1"]
                and m["balanced_accuracy"] > best["balanced_accuracy"]
            ):
                best = rec
    assert best is not None
    w, thr = best["w_rgb"], best["threshold"]
    p_te = w * tes["rgb_prob"].to_numpy(float) + (
        1.0 - w
    ) * tes["pose_prob"].to_numpy(float)
    p_dv = w * dev["rgb_prob"].to_numpy(float) + (
        1.0 - w
    ) * dev["pose_prob"].to_numpy(float)
    tes_m = clip_binary_metrics(tes["label"], (p_te >= thr).astype(int))
    dev_m = clip_binary_metrics(dev["label"], (p_dv >= thr).astype(int))

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dev_weight_search.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": list(dev["sample_id"]) + list(tes["sample_id"]),
            "split": ["DEV"] * len(dev) + ["TEST"] * len(tes),
            "label": list(dev["label"]) + list(tes["label"]),
            "pose_prob": list(dev["pose_prob"]) + list(tes["pose_prob"]),
            "rgb_prob": list(dev["rgb_prob"]) + list(tes["rgb_prob"]),
            "fusion_prob": list(p_dv) + list(p_te),
            "pred": list((p_dv >= thr).astype(int))
            + list((p_te >= thr).astype(int)),
        }
    ).to_csv(out / "predictions.csv", index=False)

    cnn_note = (
        "Locked shake CNN predictions.csv is TEST-only and no best_model.pt "
        "was saved, so CNN probabilities could not be used on DEV. Pose "
        "stream = frozen amplitude-rule scores (axis z, τ≈11.15°), min-max "
        "scaled on DEV. RGB = locked fine-tuned VideoMAE probabilities "
        "(read-only). Neither locked metrics.json was overwritten."
    )
    if CNN_PREDS.exists():
        cnn_df = pd.read_csv(CNN_PREDS)
        if "split" not in cnn_df.columns:
            cnn_note += f" {CNN_PREDS.name} has {len(cnn_df)} rows and no split column."

    metrics = {
        "task": "head_shake",
        "model": "late fusion pose + VideoMAE (linear, DEV-tuned)",
        "script": Path(__file__).name,
        "out_dir": str(out),
        "pose_source": pose_source,
        "pose_path": pose_path_used,
        "rgb_preds": str(rgb_path),
        "gold_csv": str(gold_csv),
        "label_col": args.label_col,
        "w_rgb": float(w),
        "w_pose": float(1.0 - w),
        "dev_probability_threshold": float(thr),
        "dev_n": int(len(dev)),
        "test_n": int(len(tes)),
        "dev_metrics": dev_m,
        "test_metrics": tes_m,
        "selection_rule": (
            "w and threshold by DEV F1 (ties: balanced accuracy); "
            "TEST scored once"
        ),
        "note": cnn_note,
    }
    dump_json(out / "metrics.json", metrics)
    print(
        f"DEV-chosen w_rgb={w:.2f}  threshold={thr:.2f}  DEV F1={dev_m['f1']:.3f}"
    )
    print(
        f"TEST (once): P {tes_m['precision']:.2f}  R {tes_m['recall']:.2f}  "
        f"F1 {tes_m['f1']:.2f}  "
        f"(TP{tes_m['tp']} FP{tes_m['fp']} TN{tes_m['tn']} FN{tes_m['fn']})"
    )
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
