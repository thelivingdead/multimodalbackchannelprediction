#!/usr/bin/env python3
"""Paper-style RGB frame strips from RealTalk listener face crops.

Looks like the 16-frame qualitative figures in video papers, but uses the
**same windows VideoMAE saw**: ``features/rgb16/<id>.npz`` key ``rgb``
uint8 (16, 224, 224, 3). No optical flow, no 7-class, no invented frames.

Default TEST cases (locked error analysis):
  gold_016 TP nod, gold_017 FP unclear, gold_018 FN nod, gold_024 TN unclear.

RGB npz live on otter95 (``/scratch/db01550/rgb16`` or ``features/rgb16``).
They are not on the Mac. Run this script **on otter95**.

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    /scratch/db01550/venv/bin/python scripts/plot_rgb_frame_strips.py

Then copy the PNG to the Mac::

    scp otter95:~/multimodalbackchannelprediction/dissertation-behaviour-recognition/figures/paper/rgb_frame_strips.png \\
        "/Users/divyabisht/Downloads/Msc Dissertation Divya/dissertation-behaviour-recognition/figures/paper/"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
CASES = [
    ("gold_016", "TP", "gold nod"),
    ("gold_017", "FP", "gold unclear"),
    ("gold_018", "FN", "gold nod"),
    ("gold_024", "TN", "gold unclear"),
]


def find_rgb_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for p in (
        ROOT / "features" / "rgb16",
        Path("/scratch/db01550/rgb16"),
    ):
        if (p / "gold_016.npz").exists():
            return p
    return ROOT / "features" / "rgb16"


def load_pred_map() -> dict[str, dict]:
    gold = pd.read_csv(ROOT / "results" / "gold_dataset_summary.csv")
    gold = gold[gold["split"].astype(str).str.upper() == "TEST"]
    rule = pd.read_csv(ROOT / "results" / "rule_test_predictions.csv")
    cnn = pd.read_csv(ROOT / "results" / "classifier_test_predictions.csv")
    ft = pd.read_csv(ROOT / "results" / "videomae_finetuned" / "predictions_test.csv")
    ft = ft.rename(columns={"clip_id": "sample_id"})
    out = {}
    for r in gold.itertuples():
        sid = str(r.sample_id)
        out[sid] = {
            "video_id": str(r.video_id),
            "gold": int(r.label),
        }
    for df, key in ((rule, "rule"), (cnn, "cnn")):
        for r in df.itertuples():
            sid = str(r.sample_id)
            if sid in out:
                out[sid][key] = int(r.pred)
    for r in ft.itertuples():
        sid = str(r.sample_id)
        if sid in out:
            out[sid]["ft80"] = int(r.pred)
    return out


def load_rgb(path: Path) -> tuple[np.ndarray, str, str]:
    z = np.load(path, allow_pickle=True)
    rgb = np.asarray(z["rgb"], dtype=np.uint8)
    if rgb.ndim != 4 or rgb.shape[0] < 1:
        raise ValueError(f"{path} rgb shape {rgb.shape}")
    mode = str(z["crop_mode"]) if "crop_mode" in z.files else "?"
    vid = str(z["video_id"]) if "video_id" in z.files else path.stem
    return rgb, mode, vid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dir", type=Path, default=None)
    p.add_argument(
        "--ids",
        default="gold_016,gold_017,gold_018,gold_024",
        help="comma-separated sample_ids (must have rgb16 npz)",
    )
    args = p.parse_args()
    rgb_dir = find_rgb_dir(args.rgb_dir)
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    case_note = {c[0]: (c[1], c[2]) for c in CASES}
    preds = load_pred_map()

    missing = [i for i in ids if not (rgb_dir / f"{i}.npz").exists()]
    if missing:
        raise SystemExit(
            f"STOP: no RGB npz in {rgb_dir} for {missing}. "
            "Run this on otter95 (features/rgb16 or /scratch/db01550/rgb16)."
        )

    n = len(ids)
    t = 16
    fig = plt.figure(figsize=(13.2, 1.35 * n + 1.1))
    gs = fig.add_gridspec(n, t, wspace=0.03, hspace=0.55, left=0.16, right=0.99,
                          top=0.88, bottom=0.06)
    fig.suptitle(
        "RealTalk listener face crops — the 16 frames VideoMAE classified\n"
        "(Haar crop if a face was found; centre crop otherwise). Locked TEST gold.",
        fontsize=11,
    )

    for row, sid in enumerate(ids):
        rgb, mode, vid = load_rgb(rgb_dir / f"{sid}.npz")
        info = preds.get(sid, {})
        tag, _note = case_note.get(sid, ("", ""))
        gold = "nod" if int(info.get("gold", -1)) == 1 else "unclear"
        rule = info.get("rule", "?")
        cnn = info.get("cnn", "?")
        ft80 = info.get("ft80", "?")
        title = f"{sid}  {vid}   gold={gold}"
        if tag:
            title += f"  [{tag}]"
        title += f"   crop={mode}   rule={rule}  CNN={cnn}  FT80={ft80}"
        for col in range(t):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(rgb[min(col, len(rgb) - 1)])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
                sp.set_color("#d1d5db")
            if col == 0:
                ax.set_ylabel(title, fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=10)
            if row == n - 1:
                ax.set_xlabel(str(col + 1), fontsize=7, labelpad=1)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "rgb_frame_strips.png"
    jpg = OUT / "rgb_frame_strips.jpg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(jpg, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
