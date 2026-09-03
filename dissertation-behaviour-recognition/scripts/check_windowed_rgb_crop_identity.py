#!/usr/bin/env python3
"""Audit whether the 3 s RGB crops track one person per clip.

crop_window picks the largest Haar face independently per window and ignores
the annotated person, so the crop can switch speakers mid-clip. This reads the
stored crop_box of every DEV window, flags windows whose box sits away from
the clip's median box, and renders a contact sheet for visual confirmation.

DEV only. Read-only: writes figures and CSVs, touches no features.

Otter::

    PYTHONUNBUFFERED=1 /scratch/db01550/venv/bin/python \\
        scripts/check_windowed_rgb_crop_identity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_baselines import load_windows  # noqa: E402

WINDOWS = {
    "nod": ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv",
    "shake": ROOT / "data" / "windowed_annotations" / "shake_windows_dev.csv",
}
RGB_DIR = ROOT / "features" / "rgb16_windowed"
OUT_DIR = ROOT / "results" / "windowed_nod" / "crop_audit"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
THUMB = 56
SWITCH_TOLERANCE = 0.5


def read_meta(window_id: str) -> dict | None:
    path = RGB_DIR / f"{window_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        box = np.asarray(z["crop_box"], dtype=int)
        return {
            "crop_x0": int(box[0]),
            "crop_y0": int(box[1]),
            "crop_side": int(box[2]),
            "crop_mode": str(z["crop_mode"]),
            "n_faces": int(z["n_faces"]),
            "npz_person": str(z["person"]),
        }


def thumbnail(window_id: str) -> np.ndarray:
    path = RGB_DIR / f"{window_id}.npz"
    with np.load(path, allow_pickle=True) as z:
        rgb = z["rgb"]
    step = max(1, rgb.shape[1] // THUMB)
    mid = rgb[rgb.shape[0] // 2][::step, ::step][:THUMB, :THUMB]
    out = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    out[: mid.shape[0], : mid.shape[1]] = mid
    return out


def contact_sheet(frame: pd.DataFrame, out_path: Path, task: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    clips = sorted(frame["sample_id"].unique())
    per_clip = int(frame.groupby("sample_id").size().max())
    sheet = np.full((len(clips) * THUMB, per_clip * THUMB, 3), 30, dtype=np.uint8)
    marks: list[tuple[int, int, int, int]] = []
    for r, clip in enumerate(clips):
        rows = frame[frame["sample_id"] == clip].sort_values("start_frame_relative")
        for c, row in enumerate(rows.itertuples(index=False)):
            sheet[
                r * THUMB : (r + 1) * THUMB, c * THUMB : (c + 1) * THUMB
            ] = thumbnail(str(row.window_id))
            marks.append((r, c, int(row.label), int(row.box_is_outlier)))

    fig_w = max(12.0, per_clip * 0.42)
    fig_h = max(6.0, len(clips) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#faf8f4")
    ax.imshow(sheet, interpolation="nearest")
    for r, c, label, outlier in marks:
        if label:
            ax.add_patch(
                Rectangle(
                    (c * THUMB - 0.5, r * THUMB - 0.5),
                    THUMB,
                    THUMB,
                    fill=False,
                    edgecolor="#1b7f4b",
                    lw=1.8,
                )
            )
        if outlier:
            ax.add_patch(
                Rectangle(
                    (c * THUMB - 0.5, r * THUMB - 0.5),
                    THUMB,
                    THUMB,
                    fill=False,
                    edgecolor="#c0392b",
                    lw=2.2,
                    linestyle=":",
                )
            )
    ax.set_yticks([i * THUMB + THUMB / 2 for i in range(len(clips))])
    ax.set_yticklabels(clips, fontsize=8)
    ax.set_xticks([])
    ax.set_title(
        f"3 s RGB crops, DEV, one row per clip, time left to right ({task} labels)\n"
        "green box = labelled positive    red dotted = crop box away from the "
        "clip median (likely a different face)",
        fontsize=10,
        loc="left",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("nod", "shake"), default="nod")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    frame = load_windows(WINDOWS[args.task], "DEV", DEV_IDS)
    meta = [read_meta(str(w)) for w in frame["window_id"]]
    have = [m is not None for m in meta]
    if not any(have):
        raise SystemExit(f"STOP: no crops found in {RGB_DIR}")
    frame = frame[have].reset_index(drop=True)
    frame = pd.concat(
        [frame, pd.DataFrame([m for m in meta if m is not None])], axis=1
    )
    frame["crop_centre_x"] = frame["crop_x0"] + frame["crop_side"] / 2.0

    frame["box_is_outlier"] = 0
    rows = []
    for clip, group in frame.groupby("sample_id", sort=True):
        median_x = float(group["crop_centre_x"].median())
        scale = float(group["crop_side"].median())
        offset = (group["crop_centre_x"] - median_x).abs()
        outlier = offset > SWITCH_TOLERANCE * scale
        frame.loc[group.index, "box_is_outlier"] = outlier.astype(int).to_numpy()
        rows.append(
            {
                "sample_id": clip,
                "annotated_person": str(group["person"].iloc[0]),
                "npz_person": str(group["npz_person"].iloc[0]),
                "n_windows": int(len(group)),
                "n_positive": int(group["label"].sum()),
                "median_centre_x": round(median_x, 1),
                "centre_x_range_px": round(
                    float(group["crop_centre_x"].max() - group["crop_centre_x"].min()), 1
                ),
                "median_side_px": round(scale, 1),
                "n_box_outliers": int(outlier.sum()),
                "n_centre_fallback": int((group["crop_mode"] == "centre").sum()),
                "max_faces_detected": int(group["n_faces"].max()),
            }
        )
    summary = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / f"crop_boxes_dev_{args.task}.csv", index=False)
    summary.to_csv(OUT_DIR / f"crop_summary_dev_{args.task}.csv", index=False)

    n_multi = int((summary["max_faces_detected"] > 1).sum())
    n_switch = int((summary["n_box_outliers"] > 0).sum())
    print("=====================================")
    print(f"crop identity audit — DEV, {args.task} labels")
    print(f"windows with crops: {len(frame)} / {len(meta)}")
    print(f"clips where more than one face was ever detected: {n_multi}/15")
    print(f"clips with at least one off-median crop box: {n_switch}/15")
    print(f"windows using the centre fallback (no face): "
          f"{int((frame['crop_mode'] == 'centre').sum())}")
    print(f"windows with an off-median box: {int(frame['box_is_outlier'].sum())}")
    print()
    print(summary.to_string(index=False))
    if not args.no_figure:
        sheet = OUT_DIR / f"crop_contact_sheet_dev_{args.task}.png"
        contact_sheet(frame, sheet, args.task)
        print(f"\ncontact sheet: {sheet}")
    print(f"tables: {OUT_DIR}")


if __name__ == "__main__":
    main()
