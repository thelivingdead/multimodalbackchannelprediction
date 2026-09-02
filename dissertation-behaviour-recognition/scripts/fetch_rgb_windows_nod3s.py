#!/usr/bin/env python3
"""Fetch 16-frame RGB face crops for each 3 s nod window.

Gold clips only. Writes features/rgb16_windowed/<window_id>.npz.
Does not write features/rgb16/ or locked VideoMAE folders.

Fetches each 60 s member once, then decodes 16 frames inside every
3 s window. Existing window files are skipped (resume).

Otter95::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    /scratch/db01550/venv/bin/python scripts/fetch_rgb_windows_nod3s.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_rgb_windows as fr  # noqa: E402

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
WINDOWS_TEST = ROOT / "data" / "windowed_annotations" / "nod_windows_test.csv"
GOLD_DIR = ROOT / "features" / "gold"
OUT_DIR = ROOT / "features" / "rgb16_windowed"
SUMMARY_JSON = ROOT / "results" / "windowed_nod" / "rgb16_windowed_fetch_summary.json"
N_FRAMES = 16


def _load_windows() -> pd.DataFrame:
    parts = []
    for path, split, lo, hi in (
        (WINDOWS_DEV, "DEV", 1, 15),
        (WINDOWS_TEST, "TEST", 16, 30),
    ):
        if not path.exists():
            raise SystemExit(f"STOP: missing {path}")
        df = pd.read_csv(path)
        df["sample_id"] = df["sample_id"].astype(str)
        df["split"] = df["split"].astype(str).str.upper()
        if (df["split"] != split).any():
            raise SystemExit(f"STOP: {path.name} has a non-{split} row")
        nums = df["sample_id"].str.extract(r"(\d+)", expand=False).astype(int)
        if ((nums < lo) | (nums > hi)).any():
            raise SystemExit(f"STOP: {path.name} has an id outside {split}")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default=None, help="gold_001,gold_016")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="abort if ~ has less free space than this (default 3.0)",
    )
    args = ap.parse_args()

    def check_disk(where: str = "") -> None:
        free = fr.free_gb()
        if free < args.min_free_gb:
            raise SystemExit(
                f"STOP: free disk on ~ is {free:.2f} GB < {args.min_free_gb} GB"
                f"{' at ' + where if where else ''}."
            )

    if not fr.INDEX_JSON.exists():
        raise SystemExit(f"STOP: missing {fr.INDEX_JSON}")
    index = json.loads(fr.INDEX_JSON.read_text())
    win = _load_windows()
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",") if s.strip()}
        win = win[win["sample_id"].isin(keep)].copy()
    clip_ids = list(dict.fromkeys(win["sample_id"].tolist()))
    if args.limit_clips is not None:
        clip_ids = clip_ids[: args.limit_clips]
        win = win[win["sample_id"].isin(clip_ids)].copy()
    if win.empty:
        raise SystemExit("STOP: no windows selected")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    check_disk("start")
    records: dict[str, dict] = {}
    if SUMMARY_JSON.exists():
        records = json.loads(SUMMARY_JSON.read_text()).get("windows", {})

    n_ok = n_skip = n_fail = 0
    for ci, sid in enumerate(clip_ids, 1):
        pose = GOLD_DIR / f"{sid}.npz"
        if not pose.exists():
            raise SystemExit(f"STOP: missing {pose}")
        with np.load(pose, allow_pickle=True) as pz:
            video_id = str(pz["video_id"].item() if hasattr(pz["video_id"], "item") else pz["video_id"])
            person = str(pz["person"].item() if hasattr(pz["person"], "item") else pz["person"])
            clip_start = int(np.asarray(pz["frames"]).min())
        if video_id not in index:
            raise SystemExit(f"STOP: {sid} video {video_id} not in shard index")
        entry = index[video_id]
        url = fr.SHARD_URL.format(entry["shard"])
        rows = win.loc[win["sample_id"] == sid]
        needed = []
        for r in rows.itertuples(index=False):
            out_path = OUT_DIR / f"{r.window_id}.npz"
            if out_path.exists():
                n_skip += 1
                continue
            needed.append(r)
        print(
            f"[{ci}/{len(clip_ids)}] {sid}: {len(rows)} windows, "
            f"{len(needed)} to fetch, {fr.free_gb():.2f} GB free"
        )
        if not needed:
            continue
        check_disk(sid)
        blob = fr.fetch_member(url, int(entry["offset"]), int(entry["size"]))
        sample = {"sample_id": sid, "video_id": video_id, "person": person}
        for r in needed:
            i0 = clip_start + int(r.start_frame_relative)
            i1 = clip_start + int(r.end_frame_relative) - 1
            frame_indices = fr.uniform_indices(i0, i1, N_FRAMES)
            out_path = OUT_DIR / f"{r.window_id}.npz"
            t0 = time.time()
            try:
                frames = fr.decode_frames(blob, frame_indices)
                crops, box, mode, n_faces = fr.crop_window(frames)
                fr.save_clip(out_path, crops, sample, frame_indices, box, mode, n_faces)
            except RuntimeError as exc:
                n_fail += 1
                records[str(r.window_id)] = {
                    "window_id": str(r.window_id),
                    "sample_id": sid,
                    "status": "failed",
                    "reason": str(exc)[:300],
                }
                print(f"  {r.window_id}: FAILED — {exc}")
                continue
            n_ok += 1
            records[str(r.window_id)] = {
                "window_id": str(r.window_id),
                "sample_id": sid,
                "split": str(r.split),
                "status": "ok",
                "crop_mode": mode,
                "n_faces": int(n_faces),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  {r.window_id}: ok crop={mode} {records[str(r.window_id)]['elapsed_s']}s")

    summary = {
        "script": Path(__file__).name,
        "n_windows": int(len(win)),
        "n_ok_this_run": n_ok,
        "n_skipped": n_skip,
        "n_failed_this_run": n_fail,
        "http_requests": fr._STATS["requests"],
        "bytes_ranged": fr._STATS["bytes"],
        "free_gb_end": round(fr.free_gb(), 2),
        "windows": records,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {SUMMARY_JSON.relative_to(ROOT)}: "
        f"{n_ok} new / {n_skip} skipped / {n_fail} failed"
    )
    if n_fail:
        raise SystemExit(f"INCOMPLETE: {n_fail} windows failed")


if __name__ == "__main__":
    main()
