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
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"
RGB_DIR = ROOT / "features" / "rgb16_windowed"
OUT_DIR = ROOT / "results" / "windowed_nod" / "crop_audit"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
THUMB = 56
SWITCH_TOLERANCE = 0.5


def read_meta(window_id: str, rgb_dir: Path) -> dict | None:
    path = rgb_dir / f"{window_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        box = np.asarray(z["crop_box"], dtype=int)
        meta = {
            "crop_x0": int(box[0]),
            "crop_y0": int(box[1]),
            "crop_side": int(box[2]),
            "crop_mode": str(z["crop_mode"]),
            "n_faces": int(z["n_faces"]),
            "npz_person": str(z["person"]),
            "frame_width": (
                int(np.asarray(z["frame_size"])[0]) if "frame_size" in z else 0
            ),
        }
    return meta


def watch_sides() -> dict[str, str]:
    if not WATCH_LIST.exists():
        raise SystemExit(f"STOP: missing {WATCH_LIST}")
    df = pd.read_csv(WATCH_LIST)
    side = df["who_to_watch"].astype(str).str.extract(r"^(LEFT|RIGHT)", expand=False)
    if side.isna().any():
        raise SystemExit("STOP: watch_list.csv has a row without LEFT/RIGHT")
    return dict(zip(df["video_id"].astype(str), side))


def thumbnail(window_id: str, rgb_dir: Path) -> np.ndarray:
    path = rgb_dir / f"{window_id}.npz"
    with np.load(path, allow_pickle=True) as z:
        rgb = z["rgb"]
    step = max(1, rgb.shape[1] // THUMB)
    mid = rgb[rgb.shape[0] // 2][::step, ::step][:THUMB, :THUMB]
    out = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    out[: mid.shape[0], : mid.shape[1]] = mid
    return out


def contact_sheet(
    frame: pd.DataFrame, out_path: Path, task: str, rgb_dir: Path
) -> None:
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
    ap.add_argument("--rgb-dir", type=Path, default=RGB_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tag", type=str, default="", help="suffix for output names")
    args = ap.parse_args()

    rgb_dir = args.rgb_dir.resolve()
    out_dir = args.out_dir.resolve()
    tag = f"_{args.tag}" if args.tag else ""

    frame = load_windows(WINDOWS[args.task], "DEV", DEV_IDS)
    meta = [read_meta(str(w), rgb_dir) for w in frame["window_id"]]
    have = [m is not None for m in meta]
    if not any(have):
        raise SystemExit(f"STOP: no crops found in {rgb_dir}")
    frame = frame[have].reset_index(drop=True)
    frame = pd.concat(
        [frame, pd.DataFrame([m for m in meta if m is not None])], axis=1
    )
    frame["crop_centre_x"] = frame["crop_x0"] + frame["crop_side"] / 2.0

    sides = watch_sides()
    missing_side = sorted(set(frame["video_id"].astype(str)) - set(sides))
    if missing_side:
        raise SystemExit(f"STOP: no watch side for {missing_side}")
    frame["watch_side"] = frame["video_id"].astype(str).map(sides)
    recorded = frame["frame_width"].to_numpy()
    fallback = int((frame["crop_x0"] + frame["crop_side"]).max())
    frame["frame_width"] = np.where(recorded > 0, recorded, fallback)
    if (recorded <= 0).any():
        print(
            f"NOTE: {int((recorded <= 0).sum())} windows have no stored frame size; "
            f"assuming width {fallback} from the widest box"
        )
    midline = frame["frame_width"] / 2.0
    frame["crop_on_wrong_half"] = np.where(
        frame["watch_side"] == "LEFT",
        frame["crop_centre_x"] >= midline,
        frame["crop_centre_x"] < midline,
    ).astype(int)

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
                "watch_side": str(group["watch_side"].iloc[0]),
                "npz_person": str(group["npz_person"].iloc[0]),
                "n_windows": int(len(group)),
                "n_positive": int(group["label"].sum()),
                "n_wrong_half": int(group["crop_on_wrong_half"].sum()),
                "n_positive_wrong_half": int(
                    group.loc[group["label"] == 1, "crop_on_wrong_half"].sum()
                ),
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

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"crop_boxes_dev_{args.task}{tag}.csv", index=False)
    summary.to_csv(out_dir / f"crop_summary_dev_{args.task}{tag}.csv", index=False)

    n_multi = int((summary["max_faces_detected"] > 1).sum())
    n_switch = int((summary["n_box_outliers"] > 0).sum())
    print("=====================================")
    print(f"crop identity audit — DEV, {args.task} labels")
    print(f"crops read from: {rgb_dir}")
    print(f"windows with crops: {len(frame)} / {len(meta)}")
    n_clips = int(len(summary))
    n_wrong = int(frame["crop_on_wrong_half"].sum())
    n_wrong_clips = int((summary["n_wrong_half"] > 0).sum())
    n_pos = int((frame["label"] == 1).sum())
    n_pos_wrong = int(frame.loc[frame["label"] == 1, "crop_on_wrong_half"].sum())
    print(f"clips where more than one face was ever detected: {n_multi}/{n_clips}")
    print(f"clips with at least one off-median crop box: {n_switch}/{n_clips}")
    print(
        f"WRONG PERSON: {n_wrong}/{len(frame)} windows cropped the half of the "
        f"frame the annotator was told to ignore ({100 * n_wrong / len(frame):.1f}%)"
    )
    print(f"  affected clips: {n_wrong_clips}/{n_clips}")
    print(f"  labelled-positive windows on the wrong half: {n_pos_wrong}/{n_pos}")
    print(f"windows using the centre fallback (no face): "
          f"{int((frame['crop_mode'] == 'centre').sum())}")
    print(f"windows with an off-median box: {int(frame['box_is_outlier'].sum())}")
    print()
    print(summary.to_string(index=False))
    if not args.no_figure:
        sheet = out_dir / f"crop_contact_sheet_dev_{args.task}{tag}.png"
        contact_sheet(frame, sheet, args.task, rgb_dir)
        print(f"\ncontact sheet: {sheet}")
    print(f"tables: {out_dir}")


if __name__ == "__main__":
    main()
