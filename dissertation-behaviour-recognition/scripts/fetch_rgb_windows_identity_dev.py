#!/usr/bin/env python3
"""Fetch identity-fixed 16-frame RGB crops for DEV 3 s nod windows only.

Never reads TEST window labels. Never writes TEST crops. Writes a new
directory, not features/rgb16_windowed/.

Each window is cropped by src.crop_target_person.crop_target_person_rgb.
Unresolved windows are recorded and get no rgb npz.

Otter::

    /scratch/db01550/venv/bin/python scripts/fetch_rgb_windows_identity_dev.py \\
        --out-dir /scratch/db01550/rgb16_windowed_identity_dev
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_rgb_windows as fr  # noqa: E402
from src.crop_target_person import crop_target_person_rgb, on_wrong_half  # noqa: E402

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"
GOLD_DIR = ROOT / "features" / "gold"
DEFAULT_OUT = ROOT / "features" / "rgb16_windowed_identity_dev"
DEFAULT_SUMMARY = (
    ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "fetch_summary.json"
)
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
N_FRAMES = 16


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


def gold_people() -> dict[str, str]:
    frame = pd.read_csv(GOLD_CSV)
    return dict(zip(frame["sample_id"].astype(str), frame["person"].astype(str)))


def load_dev_windows() -> pd.DataFrame:
    if not WINDOWS_DEV.exists():
        raise SystemExit(f"STOP: missing {WINDOWS_DEV}")
    frame = pd.read_csv(WINDOWS_DEV)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["split"] = frame["split"].astype(str).str.upper()
    if (frame["split"] != "DEV").any():
        raise SystemExit("STOP: nod_windows_dev.csv has a non-DEV row")
    if set(frame["sample_id"]) != DEV_IDS:
        raise SystemExit("STOP: DEV window file is not gold_001 to gold_015")
    leaked = set(frame["sample_id"]) & TEST_IDS
    if leaked:
        raise SystemExit(f"STOP: TEST id in a DEV-only fetch: {sorted(leaked)}")
    return frame.reset_index(drop=True)


def assert_person_side(person: str, side: str, sample_id: str) -> None:
    expected = "LEFT" if person == "p0" else "RIGHT" if person == "p1" else ""
    if expected != side:
        raise SystemExit(
            f"STOP: {sample_id} person={person} does not match watch_side={side}. "
            "See results/windowed_dev/videomae_identity_fixed/person_mapping.md"
        )


def save_resolved(path: Path, payload: dict, sample: dict,
                  frame_indices: np.ndarray, preview: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    box = payload["crop_box"]
    np.savez(
        tmp,
        rgb=payload["rgb"].astype(np.uint8),
        preview=preview.astype(np.uint8),
        frame_indices=frame_indices.astype(np.int32),
        sample_id=sample["sample_id"],
        video_id=sample["video_id"],
        person=sample["person"],
        watch_side=payload["watch_side"],
        crop_box=np.asarray(box, dtype=np.int32),
        crop_centre_x=np.float64(payload["crop_centre_x"]),
        crop_status=payload["crop_status"],
        crop_mode="target_person",
        n_target_detections=np.int64(payload["n_target_detections"]),
        n_other_detections=np.int64(payload["n_other_detections"]),
        frame_size=np.asarray(
            [payload["frame_width"], payload["frame_height"]], dtype=np.int32
        ),
    )
    tmp.rename(path)


def downscale_preview(frame: np.ndarray, box: tuple | None, max_width: int = 480):
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width)
    preview = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    if box is not None:
        x0, y0, side, _ = box
        p1 = (int(round(x0 * scale)), int(round(y0 * scale)))
        p2 = (int(round((x0 + side) * scale)), int(round((y0 + side) * scale)))
        cv2.rectangle(preview, p1, p2, (0, 255, 0), 2)
        mid_x = int(round(preview.shape[1] / 2))
        cv2.line(preview, (mid_x, 0), (mid_x, preview.shape[0] - 1), (255, 255, 0), 1)
    return preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if "rgb16_windowed" == out_dir.name and "identity" not in str(out_dir):
        raise SystemExit(
            "STOP: refusing to write identity-fixed crops into the old "
            "features/rgb16_windowed/ directory"
        )
    sides = watch_sides()
    people = gold_people()
    win = load_dev_windows()
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",") if s.strip()}
        win = win[win["sample_id"].isin(keep)].copy()
    clip_ids = list(dict.fromkeys(win["sample_id"].tolist()))
    if args.limit_clips is not None:
        clip_ids = clip_ids[: args.limit_clips]
        win = win[win["sample_id"].isin(clip_ids)].copy()
    if win.empty:
        raise SystemExit("STOP: no DEV windows selected")

    if not fr.INDEX_JSON.exists():
        raise SystemExit(f"STOP: missing shard index {fr.INDEX_JSON}")
    index = json.loads(fr.INDEX_JSON.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_ok = n_unresolved = n_fail = n_skip = 0

    for ci, sid in enumerate(clip_ids, 1):
        pose = GOLD_DIR / f"{sid}.npz"
        if not pose.exists():
            raise SystemExit(f"STOP: missing {pose}")
        with np.load(pose, allow_pickle=True) as payload:
            video_id = str(np.asarray(payload["video_id"]).item())
            person = str(people[sid])
            clip_start = int(np.asarray(payload["frames"]).min())
        if video_id not in index:
            raise SystemExit(f"STOP: {sid} video {video_id} not in shard index")
        if video_id not in sides:
            raise SystemExit(f"STOP: {sid} has no watch_list side")
        side = sides[video_id]
        assert_person_side(person, side, sid)
        free = fr.free_gb()
        if free < args.min_free_gb:
            raise SystemExit(f"STOP: free disk on home is {free:.2f} GB")
        needed = []
        for rec in win.loc[win["sample_id"] == sid].itertuples(index=False):
            path = out_dir / f"{rec.window_id}.npz"
            if path.exists():
                n_skip += 1
                continue
            needed.append(rec)
        print(
            f"[{ci}/{len(clip_ids)}] {sid} person={person} side={side} "
            f"{len(needed)} to fetch"
        )
        if not needed:
            continue
        entry = index[video_id]
        blob = fr.fetch_member(
            fr.SHARD_URL.format(entry["shard"]),
            int(entry["offset"]),
            int(entry["size"]),
        )
        sample = {"sample_id": sid, "video_id": video_id, "person": person}
        for rec in needed:
            i0 = clip_start + int(rec.start_frame_relative)
            i1 = clip_start + int(rec.end_frame_relative) - 1
            frame_indices = fr.uniform_indices(i0, i1, N_FRAMES)
            t0 = time.time()
            try:
                frames = fr.decode_frames(blob, frame_indices)
                cropped = crop_target_person_rgb(frames, side)
            except RuntimeError as exc:
                n_fail += 1
                rows.append(
                    {
                        "window_id": str(rec.window_id),
                        "sample_id": sid,
                        "split": "DEV",
                        "crop_status": "failed",
                        "reason": str(exc)[:300],
                    }
                )
                print(f"  {rec.window_id}: FAILED {exc}")
                continue
            mid = frames[len(frames) // 2]
            preview = downscale_preview(mid, cropped.get("crop_box"))
            record = {
                "window_id": str(rec.window_id),
                "sample_id": sid,
                "video_id": video_id,
                "split": "DEV",
                "start_frame_relative": int(rec.start_frame_relative),
                "end_frame_relative": int(rec.end_frame_relative),
                "label": int(rec.label),
                "person": person,
                "watch_side": side,
                "crop_status": cropped["crop_status"],
                "n_target_detections": cropped["n_target_detections"],
                "n_other_detections": cropped["n_other_detections"],
                "crop_centre_x": cropped["crop_centre_x"],
                "frame_width": cropped["frame_width"],
                "reason": cropped["reason"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            if cropped["crop_status"] == "resolved":
                if on_wrong_half(
                    float(cropped["crop_centre_x"]),
                    int(cropped["frame_width"]),
                    side,
                ):
                    raise SystemExit(
                        f"STOP: resolved crop for {rec.window_id} is on the "
                        "excluded half. The new cropper is not allowed to do that."
                    )
                save_resolved(
                    out_dir / f"{rec.window_id}.npz",
                    cropped,
                    sample,
                    frame_indices,
                    preview,
                )
                n_ok += 1
            else:
                n_unresolved += 1
            rows.append(record)
            print(
                f"  {rec.window_id}: {cropped['crop_status']} "
                f"{cropped['reason'] or 'ok'} {record['elapsed_s']}s"
            )

    manifest = pd.DataFrame(rows)
    manifest_path = args.summary_json.parent / "fetch_manifest.csv"
    if manifest_path.exists() and not manifest.empty:
        old = pd.read_csv(manifest_path)
        keep = old[~old["window_id"].isin(manifest["window_id"])]
        manifest = pd.concat([keep, manifest], ignore_index=True)
    if not manifest.empty:
        manifest.to_csv(manifest_path, index=False)
    summary = {
        "script": Path(__file__).name,
        "split": "DEV",
        "test_touched": False,
        "out_dir": str(out_dir),
        "out_dir_note": "may live on /scratch, outside the repo",
        "n_windows_selected": int(len(win)),
        "n_resolved_this_run": n_ok,
        "n_unresolved_this_run": n_unresolved,
        "n_failed_this_run": n_fail,
        "n_skipped": n_skip,
        "http_requests": fr._STATS["requests"],
        "bytes_ranged": fr._STATS["bytes"],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if n_fail:
        raise SystemExit(f"INCOMPLETE: {n_fail} windows failed to decode")


if __name__ == "__main__":
    main()
