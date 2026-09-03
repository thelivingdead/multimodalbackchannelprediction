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
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"
SIDE_TOLERANCE = 0.5
_CASCADE = None


def watch_sides() -> dict[str, str]:
    """video_id -> LEFT or RIGHT, from the annotator's instruction sheet."""
    if not WATCH_LIST.exists():
        raise SystemExit(f"STOP: missing {WATCH_LIST}")
    df = pd.read_csv(WATCH_LIST)
    side = df["who_to_watch"].astype(str).str.extract(r"^(LEFT|RIGHT)", expand=False)
    if side.isna().any():
        raise SystemExit("STOP: watch_list.csv has a row without LEFT/RIGHT")
    return dict(zip(df["video_id"].astype(str), side))


def detect_candidate(frame: np.ndarray, side: str):
    """Largest face on the annotated half. Returns (cx, cy, face_side) or None."""
    global _CASCADE
    import cv2

    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    height, width = frame.shape[:2]
    faces = _CASCADE.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    midline = width / 2.0
    keep = []
    for x, y, w, h in faces:
        cx = x + w / 2.0
        if (side == "LEFT" and cx < midline) or (side == "RIGHT" and cx >= midline):
            keep.append((int(w) * int(h), cx, y + h / 2.0, float(max(w, h))))
    if not keep:
        return None
    _, cx, cy, face_side = max(keep, key=lambda t: t[0])
    return cx, cy, face_side


def square_box(cx: float, cy: float, face_side: float, width: int, height: int):
    box_side = int(round(face_side * fr.CROP_SCALE))
    box_side = min(box_side, width, height)
    x0 = int(min(max(round(cx) - box_side // 2, 0), width - box_side))
    y0 = int(min(max(round(cy) - box_side // 2, 0), height - box_side))
    return (x0, y0, box_side, box_side)


def half_frame_box(side: str, width: int, height: int):
    box_side = min(width // 2, height)
    cx = width / 4.0 if side == "LEFT" else 3.0 * width / 4.0
    return square_box(cx, height / 2.0, box_side / fr.CROP_SCALE, width, height)


def crop_with_box(frames: np.ndarray, box) -> np.ndarray:
    import cv2

    x0, y0, box_side, _ = box
    return np.stack(
        [
            cv2.resize(
                f[y0 : y0 + box_side, x0 : x0 + box_side],
                (fr.CROP_SIZE, fr.CROP_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            for f in frames
        ]
    )


def save_window(out_path: Path, crops: np.ndarray, sample: dict,
                frame_indices: np.ndarray, box, side: str,
                box_source: str, n_faces_half: int) -> None:
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        rgb=crops.astype(np.uint8),
        frame_indices=frame_indices.astype(np.int32),
        sample_id=sample["sample_id"],
        video_id=sample["video_id"],
        person=sample["person"],
        crop_box=np.asarray(box, dtype=np.int32),
        crop_mode=box_source,
        n_faces=np.int64(n_faces_half),
        watch_side=side,
    )
    tmp_path.rename(out_path)


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
    ap.add_argument("--split", choices=("DEV", "TEST", "BOTH"), default="BOTH")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument(
        "--side-aware",
        action="store_true",
        help="keep only faces on the annotated half and hold one box per clip",
    )
    args = ap.parse_args()
    out_dir = (args.out_dir or OUT_DIR).resolve()
    summary_json = args.summary_json or SUMMARY_JSON
    if args.side_aware and out_dir == OUT_DIR.resolve():
        raise SystemExit(
            "STOP: --side-aware needs a fresh --out-dir; it would otherwise mix "
            "side-aware crops into features/rgb16_windowed/"
        )
    sides = watch_sides() if args.side_aware else {}

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
    if args.split != "BOTH":
        win = win[win["split"] == args.split].copy()
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",") if s.strip()}
        win = win[win["sample_id"].isin(keep)].copy()
    clip_ids = list(dict.fromkeys(win["sample_id"].tolist()))
    if args.limit_clips is not None:
        clip_ids = clip_ids[: args.limit_clips]
        win = win[win["sample_id"].isin(clip_ids)].copy()
    if win.empty:
        raise SystemExit("STOP: no windows selected")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    check_disk("start")
    records: dict[str, dict] = {}
    if summary_json.exists():
        records = json.loads(summary_json.read_text()).get("windows", {})

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
            out_path = out_dir / f"{r.window_id}.npz"
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

        side = sides.get(video_id, "") if args.side_aware else ""
        candidates: dict[str, tuple] = {}
        reference = None
        if args.side_aware:
            if not side:
                raise SystemExit(f"STOP: {sid} video {video_id} not in watch_list")
            probe_index = np.asarray(
                [
                    fr.uniform_indices(
                        clip_start + int(r.start_frame_relative),
                        clip_start + int(r.end_frame_relative) - 1,
                        N_FRAMES,
                    )[N_FRAMES // 2]
                    for r in needed
                ]
            )
            probes = fr.decode_frames(blob, probe_index)
            for r, probe in zip(needed, probes):
                found = detect_candidate(probe, side)
                if found is not None:
                    candidates[str(r.window_id)] = found
            if candidates:
                stack = np.asarray(list(candidates.values()), dtype=float)
                reference = (
                    float(np.median(stack[:, 0])),
                    float(np.median(stack[:, 1])),
                    float(np.median(stack[:, 2])),
                )
            print(
                f"  side={side}  face found on that half in "
                f"{len(candidates)}/{len(needed)} windows"
            )

        for r in needed:
            i0 = clip_start + int(r.start_frame_relative)
            i1 = clip_start + int(r.end_frame_relative) - 1
            frame_indices = fr.uniform_indices(i0, i1, N_FRAMES)
            out_path = out_dir / f"{r.window_id}.npz"
            t0 = time.time()
            try:
                frames = fr.decode_frames(blob, frame_indices)
                if args.side_aware:
                    height, width = frames.shape[1:3]
                    own = candidates.get(str(r.window_id))
                    if reference is None:
                        box = half_frame_box(side, width, height)
                        mode = "half_frame_no_face"
                    elif own is None:
                        box = square_box(*reference, width, height)
                        mode = "clip_reference_no_face"
                    elif abs(own[0] - reference[0]) > SIDE_TOLERANCE * reference[2]:
                        box = square_box(*reference, width, height)
                        mode = "clip_reference_snapped"
                    else:
                        box = square_box(*own, width, height)
                        mode = "own_detection"
                    crops = crop_with_box(frames, box)
                    n_faces = 1 if own is not None else 0
                    save_window(
                        out_path, crops, sample, frame_indices, box, side, mode, n_faces
                    )
                else:
                    crops, box, mode, n_faces = fr.crop_window(frames)
                    fr.save_clip(
                        out_path, crops, sample, frame_indices, box, mode, n_faces
                    )
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

    modes: dict[str, int] = {}
    for rec in records.values():
        if rec.get("status") == "ok":
            modes[rec.get("crop_mode", "?")] = modes.get(rec.get("crop_mode", "?"), 0) + 1
    summary = {
        "script": Path(__file__).name,
        "side_aware": bool(args.side_aware),
        "split": args.split,
        "out_dir": str(out_dir.relative_to(ROOT)),
        "n_windows": int(len(win)),
        "n_ok_this_run": n_ok,
        "n_skipped": n_skip,
        "n_failed_this_run": n_fail,
        "crop_modes": modes,
        "http_requests": fr._STATS["requests"],
        "bytes_ranged": fr._STATS["bytes"],
        "free_gb_end": round(fr.free_gb(), 2),
        "windows": records,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {summary_json.relative_to(ROOT)}: "
        f"{n_ok} new / {n_skip} skipped / {n_fail} failed"
    )
    if modes:
        print("crop modes: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))
    if n_fail:
        raise SystemExit(f"INCOMPLETE: {n_fail} windows failed")


if __name__ == "__main__":
    main()
