#!/usr/bin/env python3
"""Audit whether the 60 s RGB clip crops show the annotated person.

The 60 s fetcher takes one box per clip on the middle frame, choosing the
largest Haar detection with no side filter, and the 3 s fetcher imports that
same function. The 3 s crops were found to show the excluded person in 51 % of
windows, so the 60 s crops need the same check before their results stand.

A crop whose centre falls on the half of the frame the annotator was told to
ignore is a crop of the wrong person. DEV and TEST are both audited; this reads
saved crops only and scores no model.

Otter::

    /scratch/db01550/venv/bin/python scripts/check_clip_rgb_crop_identity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RGB_DIR = ROOT / "features" / "rgb16"
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"
OUT_DIR = ROOT / "results" / "windowed_nod" / "crop_audit"
GOLD_IDS = [f"gold_{i:03d}" for i in range(1, 31)]


def watch_sides() -> dict[str, str]:
    if not WATCH_LIST.exists():
        raise SystemExit(f"STOP: missing {WATCH_LIST}")
    frame = pd.read_csv(WATCH_LIST)
    side = frame["who_to_watch"].astype(str).str.extract(
        r"^(LEFT|RIGHT)", expand=False
    )
    if side.isna().any():
        raise SystemExit("STOP: watch_list.csv has a row without LEFT/RIGHT")
    return dict(zip(frame["video_id"].astype(str), side))


def read_clip(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        box = np.asarray(payload["crop_box"], dtype=int)
        return {
            "sample_id": str(payload["sample_id"]),
            "video_id": str(payload["video_id"]),
            "person": str(payload["person"]),
            "crop_x0": int(box[0]),
            "crop_side": int(box[2]),
            "crop_mode": str(payload["crop_mode"]),
            "n_faces": int(payload["n_faces"]),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", type=Path, default=RGB_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--frame-width",
        type=int,
        default=0,
        help="frame width in px; 0 infers it from the widest crop box",
    )
    args = ap.parse_args()
    rgb_dir = args.rgb_dir.resolve()

    rows = [
        read_clip(rgb_dir / f"{sid}.npz")
        for sid in GOLD_IDS
        if (rgb_dir / f"{sid}.npz").exists()
    ]
    if not rows:
        raise SystemExit(f"STOP: no gold clip crops in {rgb_dir}")
    frame = pd.DataFrame(rows)

    width = args.frame_width or int((frame["crop_x0"] + frame["crop_side"]).max())
    if not args.frame_width:
        print(f"NOTE: inferring frame width {width} px from the widest crop box")
    sides = watch_sides()
    missing = sorted(set(frame["video_id"]) - set(sides))
    if missing:
        raise SystemExit(f"STOP: no watch side for {missing}")
    frame["watch_side"] = frame["video_id"].map(sides)
    frame["crop_centre_x"] = frame["crop_x0"] + frame["crop_side"] / 2.0
    frame["split"] = np.where(
        frame["sample_id"].str.extract(r"(\d+)", expand=False).astype(int) <= 15,
        "DEV",
        "TEST",
    )
    frame["on_wrong_half"] = np.where(
        frame["watch_side"] == "LEFT",
        frame["crop_centre_x"] >= width / 2.0,
        frame["crop_centre_x"] < width / 2.0,
    ).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "clip_crop_identity_gold.csv"
    frame.to_csv(out_path, index=False)

    print("=====================================")
    print(f"60 s clip crop identity audit — {len(frame)} gold clips")
    print(f"crops read from: {rgb_dir}")
    for split in ("DEV", "TEST"):
        part = frame[frame["split"] == split]
        if part.empty:
            continue
        wrong = int(part["on_wrong_half"].sum())
        multi = int((part["n_faces"] > 1).sum())
        print(
            f"{split}: {wrong}/{len(part)} clips cropped the excluded person; "
            f"{multi}/{len(part)} had more than one face detected"
        )
    print()
    print(
        frame[
            [
                "sample_id",
                "split",
                "person",
                "watch_side",
                "crop_centre_x",
                "crop_side",
                "n_faces",
                "crop_mode",
                "on_wrong_half",
            ]
        ].to_string(index=False)
    )
    total = int(frame["on_wrong_half"].sum())
    print()
    if total:
        print(
            f"VERDICT: {total} of {len(frame)} clips show the wrong person. The RGB "
            "rows of the 60 s protocol cannot be quoted as they stand."
        )
    else:
        print(
            "VERDICT: every clip cropped the annotated half. The 60 s RGB results "
            "stand on identity grounds."
        )
    print(f"table: {out_path}")


if __name__ == "__main__":
    main()
