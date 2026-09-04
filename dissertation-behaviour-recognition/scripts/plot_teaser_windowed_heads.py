#!/usr/bin/env python3
"""3 s windowed paper teaser: listener heads across a nod and a shake.

This is the 3 s windowed face teaser. It is not the 60 s nod teaser
(plot_teaser_figure.py → teaser_backchannel) and not the pose-only yaw
chart (plot_teaser_shake_windowed.py → teaser_shake_windowed).

Figures only. Does not train, does not write a new TEST score, and refuses
to write locked 60 s directories or the old 60 s teaser files.

Crops: official RealTalk gold-person boxes (p0 LEFT / p1 RIGHT), one box
held for the window, then a uniform square with pad. Side-aware
Haar on the annotator half is the last resort. Withdrawn largest-face
Haar is never used.

    python3 scripts/plot_teaser_windowed_heads.py
    python3 scripts/plot_teaser_windowed_heads.py --check

Frames are decoded from avis already in cache/teaser_windowed/. This script
does not download new RealTalk members, does not run fetch_rgb_windows_nod3s.py,
and does not write features/rgb16_windowed/.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.paper_figure_style import (  # noqa: E402
    GREY,
    INK,
    MUTED,
    PAPER,
    save,
)

RED = "#c0392b"
OOF_CNN = (
    ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "predictions_oof_dev.csv",
    ROOT / "results" / "windowed_shake" / "pose_cnn_loco_dev" / "predictions_oof_dev.csv",
)
from src.crop_target_person import crop_target_person_rgb  # noqa: E402
import check_split_leakage as gate  # noqa: E402

CACHE = ROOT / "cache" / "teaser_windowed"
OUT_DEFAULT = ROOT / "figures" / "paper" / "teaser_windowed_heads"
GOLD_DIR = ROOT / "features" / "gold"
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"
INDEX_JSON = ROOT / "results" / "video_shard_index.json"
SHAKE_EVENTS = ROOT / "data" / "windowed_annotations" / "shake_events_windowed_test.csv"
NOD_EVENTS = ROOT / "data" / "windowed_annotations" / "nod_events_windowed_test.csv"
SHAKE_METRICS = ROOT / "results" / "windowed_shake" / "baselines_bacc" / "metrics.json"

FPS = 25.0
WINDOW_SEC = 3.0
N_FRAMES = 5
CROP_SIZE = 224
CROP_PAD = 0.28
SAVGOL_WINDOW = 11
SAVGOL_POLY = 2
NOD_AMP_THRESHOLD = 1.4921350723657896
NOD_RR_THRESHOLD = 0.2129783741925497
SHARD_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/videos/{}"
)
ANN_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/"
    "annotations.tar.gz"
)
UA = {"User-Agent": "Mozilla/5.0 (dissertation-windowed-teaser)"}
LINE = "#d8d8dc"
BAND = "#ececee"

BLOCKED_STEMS = (
    "teaser_backchannel",
    "teaser_shake_windowed",
)
BLOCKED_FEATURE_DIRS = (
    ROOT / "features" / "rgb16",
    ROOT / "features" / "rgb16_windowed",
)

# Shake: gold_023 15-18 s (local avi cached). Nod: gold_030 21-24 s (local avi cached).
# Not gold_019 / gold_025. Face ids stay off the figure.
PANELS = (
    {
        "kind": "shake",
        "sample_id": "gold_023",
        "window_id": "gold_023_w00375",
        "start_sec": 15.0,
        "headline": "Head shake, 3 s",
        "pose_name": "Yaw, EMOCA y (°)",
        "axis": 1,
        "watch_side": "RIGHT",
    },
    {
        "kind": "nod",
        "sample_id": "gold_030",
        "window_id": "gold_030_w00525",
        "start_sec": 21.0,
        "headline": "Head nod, 3 s",
        "pose_name": "Pitch, EMOCA x (°)",
        "axis": 0,
        "watch_side": "RIGHT",
    },
)


def refuse_output(path: Path) -> None:
    """Refuse locked 60 s dirs, withdrawn RGB dirs, and the old teasers."""
    resolved = path.expanduser().resolve()
    stem = resolved.name.split(".")[0] if resolved.suffix else resolved.name
    if stem == "teaser_backchannel" or "teaser_backchannel" in resolved.as_posix():
        raise SystemExit(
            "STOP: refusing to write the 60 s teaser path "
            "(teaser_backchannel). This script writes teaser_windowed_heads."
        )
    if stem == "teaser_shake_windowed" or "teaser_shake_windowed" in resolved.as_posix():
        raise SystemExit(
            "STOP: refusing to overwrite the pose-only 3 s teaser "
            "(teaser_shake_windowed). This script writes teaser_windowed_heads."
        )
    if any(stem == blocked or stem.startswith(blocked + ".") for blocked in BLOCKED_STEMS):
        raise SystemExit(
            f"STOP: refusing to overwrite {resolved.name}. "
            "This script writes figures/paper/teaser_windowed_heads only."
        )
    for blocked in BLOCKED_FEATURE_DIRS:
        try:
            resolved.relative_to(blocked.resolve())
        except ValueError:
            continue
        raise SystemExit(
            f"STOP: refusing to write {resolved} under withdrawn RGB dir {blocked}"
        )
    for blocked in gate.LOCKED_OUT_DIRS:
        try:
            resolved.relative_to(Path(blocked).resolve())
        except ValueError:
            continue
        raise SystemExit(f"STOP: refusing to write locked directory {blocked}")


def watch_sides() -> dict[str, str]:
    frame = pd.read_csv(WATCH_LIST)
    side = frame["who_to_watch"].astype(str).str.extract(
        r"^(LEFT|RIGHT)", expand=False
    )
    if side.isna().any():
        raise SystemExit("STOP: watch_list.csv has a row without LEFT/RIGHT")
    return dict(zip(frame["video_id"].astype(str), side))


def load_gold_row(sample_id: str) -> pd.Series:
    gold = pd.read_csv(GOLD_CSV)
    hit = gold[gold["sample_id"].astype(str) == sample_id]
    if hit.empty:
        raise SystemExit(f"STOP: {sample_id} missing from gold_annotations.csv")
    return hit.iloc[0]


def frozen_shake_tau() -> float:
    payload = json.loads(SHAKE_METRICS.read_text())
    if int(payload["axis"]) != 1 or str(payload["axis_name"]) != "y":
        raise SystemExit("STOP: frozen shake TEST rule is not yaw (axis y)")
    return float(payload["dev_selected_window_threshold"])


def savgol_smooth(x: np.ndarray) -> np.ndarray:
    """Savitzky-Golay window 11, poly 2 (same as rule_score)."""
    x = np.asarray(x, dtype=float)
    fill = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
    x = np.where(np.isfinite(x), x, fill)
    win = min(SAVGOL_WINDOW, x.size if x.size % 2 == 1 else x.size - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    half = win // 2
    t = np.arange(-half, half + 1, dtype=float)
    a = np.vander(t, SAVGOL_POLY + 1, increasing=True)
    kernel = np.linalg.pinv(a)[0]
    pad = np.pad(x, half, mode="edge")
    return np.convolve(pad, kernel[::-1], mode="valid")


def turning_pair(sm: np.ndarray) -> tuple[float, int, int]:
    """Peak-to-peak pair used by rule_score (5-50 frames)."""
    sm = np.asarray(sm, dtype=float)
    if sm.size < 3:
        return 0.0, 0, max(0, sm.size - 1)
    d = np.diff(sm)
    turns = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    best = 0.0
    a_best, b_best = 0, int(sm.size - 1)
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            span = int(b - a)
            if span < 5 or span > 50:
                continue
            amp = float(abs(sm[int(b)] - sm[int(a)]))
            if amp > best:
                best = amp
                a_best, b_best = int(a), int(b)
    if best == 0.0:
        a_best = int(np.argmin(sm))
        b_best = int(np.argmax(sm))
        if a_best > b_best:
            a_best, b_best = b_best, a_best
        best = float(abs(sm[b_best] - sm[a_best]))
    return best, a_best, b_best


def return_ratio(sm: np.ndarray) -> float:
    sm = np.asarray(sm, dtype=float)
    if sm.size < 2:
        return 0.0
    net = float(abs(sm[-1] - sm[0]))
    path = float(np.sum(np.abs(np.diff(sm))))
    return net / (path + 1e-6)


def gesture_sample_times(lo: float, hi: float, n: int = N_FRAMES) -> np.ndarray:
    """n times from start to end of the annotated interval, inclusive."""
    if hi <= lo:
        raise SystemExit("STOP: empty annotated gesture interval")
    n = max(2, int(n))
    return np.linspace(float(lo), float(hi), n)


def frame_sample_times(n: int) -> np.ndarray:
    """Deprecated equal-tile helper kept for tests that still call it."""
    return gesture_sample_times(0.0, WINDOW_SEC, n)


def window_indices(clip_start: int, start_sec: float, rel_times: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel_times, dtype=float)
    return (int(clip_start) + np.round((float(start_sec) + rel) * FPS)).astype(np.int64)


def resize_sq(rgb: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    return np.asarray(Image.fromarray(rgb).resize((size, size), resample))


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("STOP: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _drain(stream, sink: bytearray) -> None:
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        sink += chunk


def decode_frames_file(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    exe = ffmpeg_exe()
    expr = "+".join(f"eq(n,{int(i)})" for i in frame_indices)
    cmd = [
        exe, "-hide_banner", "-loglevel", "info",
        "-an", "-sn", "-dn", "-i", str(video_path),
        "-vf", f"select='{expr}'", "-vsync", "0",
        "-frames:v", str(len(frame_indices)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, err_buf = bytearray(), bytearray()
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_buf))
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_buf))
    t_out.start()
    t_err.start()
    proc.wait(timeout=600)
    t_out.join()
    t_err.join()
    err = bytes(err_buf).decode("utf-8", "replace")
    match = re.search(r"Video: .{0,200}?(\d{2,6})x(\d{2,6})", err)
    if not match:
        raise RuntimeError("ffmpeg found no video size: " + err[-400:])
    width, height = int(match.group(1)), int(match.group(2))
    nbytes = width * height * 3
    n_out = len(out_buf) // nbytes
    if n_out < len(frame_indices):
        raise RuntimeError(f"short_decode wanted {len(frame_indices)} got {n_out}")
    frames = np.frombuffer(bytes(out_buf[: n_out * nbytes]), np.uint8)
    return frames.reshape(n_out, height, width, 3).copy()


def download_range(url: str, offset: int, size: int, dest: Path) -> None:
    """Write exactly ``size`` bytes from a Range request into ``dest``."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl, "-L", "--fail", "--retry", "4", "--retry-delay", "3",
            "--connect-timeout", "30",
            "-A", UA["User-Agent"],
            "-H", f"Range: bytes={offset}-{offset + size - 1}",
            "-o", str(tmp), url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"STOP: curl Range read failed for {url} "
                f"(exit {proc.returncode}): {proc.stderr[-400:]}"
            )
    else:
        try:
            import requests
        except ImportError as exc:
            raise SystemExit("STOP: pip install requests") from exc
        headers = dict(UA)
        headers["Range"] = f"bytes={offset}-{offset + size - 1}"
        with requests.get(url, headers=headers, stream=True, timeout=(60, 600)) as resp:
            if resp.status_code != 206:
                raise SystemExit(
                    f"STOP: HTTP {resp.status_code} for {url} (need 206)"
                )
            n = 0
            with open(tmp, "wb") as handle:
                for chunk in resp.iter_content(1 << 20):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    n += len(chunk)
                    if n % (80 << 20) < (1 << 20):
                        print(f"  {n / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
    if not tmp.exists() or tmp.stat().st_size != size:
        got = tmp.stat().st_size if tmp.exists() else 0
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"STOP: incomplete video download ({got} bytes, index says {size})"
        )
    tmp.rename(dest)


def fetch_scene_frames(video_id: str, frame_indices: np.ndarray) -> np.ndarray:
    """Load cached scene npy only. Never decode avi or download."""
    tag = "_".join(str(int(i)) for i in frame_indices)
    npy = CACHE / f"{video_id}_scene_{tag}.npy"
    if npy.exists():
        print("cache hit", npy.name)
        return np.load(npy)
    want = len(frame_indices)
    fallback = []
    for path in sorted(CACHE.glob(f"{video_id}_scene_*.npy")):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 4 and int(arr.shape[0]) == want:
            fallback.append(path)
    if fallback:
        # Prefer the 5-frame grey-band caches written 16:42 (shortest name).
        path = min(fallback, key=lambda p: (p.stat().st_size, p.name))
        print("cache reuse", path.name)
        return np.load(path)
    raise SystemExit(
        f"STOP: no cached npy for {video_id} ({want} frames). "
        "Will not decode video or download RealTalk members."
    )


def ensure_annotation_json(video_id: str) -> Path:
    dest = CACHE / "metadata" / f"{video_id}.json"
    if dest.exists():
        return dest
    raise SystemExit(
        f"STOP: cached metadata missing for {video_id}; will not download"
    )


def cnn_oof_verdict(window_id: str) -> str:
    """DEV OOF only. TEST windows and missing ids print n/a."""
    wid = str(window_id)
    for path in OOF_CNN:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "window_id" not in frame.columns:
            continue
        hit = frame[frame["window_id"].astype(str) == wid]
        if hit.empty:
            continue
        rec = hit.iloc[0]
        split = str(rec["split"]).upper() if "split" in hit.columns else "DEV"
        if split == "TEST":
            return "n/a"
        for col in ("prediction", "pred", "y_hat"):
            if col in hit.columns:
                return "yes" if int(rec[col]) == 1 else "no"
    return "n/a"


def rule_verdict(spec: dict, amp: float, rr: float) -> str:
    if spec["kind"] == "shake":
        return "yes" if float(amp) >= frozen_shake_tau() else "no"
    fired = float(amp) >= NOD_AMP_THRESHOLD and float(rr) <= NOD_RR_THRESHOLD
    return "yes" if fired else "no"


def listener_boxes(
    video_id: str, person: str, frame_indices: np.ndarray,
) -> np.ndarray:
    raw = json.loads(ensure_annotation_json(video_id).read_text())
    key = "p0" if person == "p0" else "p1"
    boxes = np.full((len(frame_indices), 4), np.nan)
    for i, fnum in enumerate(frame_indices):
        rec = raw.get(str(int(fnum)), {}) or {}
        people = rec.get("people") or {}
        if key not in people:
            continue
        box = people[key].get("bbox")
        if box is not None:
            boxes[i] = np.asarray(box, dtype=float)[:4]
    return boxes


def scale_box(xyxy, frame_shape, native=(720, 1280)):
    h, w = frame_shape[:2]
    sx, sy = w / native[1], h / native[0]
    x1, y1, x2, y2 = xyxy
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=float)


def square_crop(
    frame: np.ndarray,
    xyxy,
    size: int = CROP_SIZE,
    pad: float = CROP_PAD,
):
    """Centered square crop: official box, then equal pad on all sides."""
    if xyxy is None or not np.all(np.isfinite(xyxy)):
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = float(max(bw, bh) * (1.0 + 2.0 * pad))
    side = min(
        side,
        2.0 * cx,
        2.0 * (w - cx),
        2.0 * cy,
        2.0 * (h - cy),
        float(w),
        float(h),
    )
    if side < max(bw, bh) * 0.95:
        return None
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    side_i = int(round(side))
    x0 = int(np.clip(x0, 0, max(0, w - side_i)))
    y0 = int(np.clip(y0, 0, max(0, h - side_i)))
    patch = frame[y0 : y0 + side_i, x0 : x0 + side_i]
    if patch.size == 0:
        return None
    return resize_sq(patch, size), (x0, y0, side_i, side_i)


def held_realtalk_crops(
    frames: np.ndarray, boxes: np.ndarray, watch_side: str,
) -> tuple[np.ndarray | None, tuple | None]:
    finite = boxes[np.all(np.isfinite(boxes), axis=1)]
    if len(finite) < 2:
        return None, None
    native = np.median(finite, axis=0)
    scaled = scale_box(native, frames[0].shape)
    cx = 0.5 * (scaled[0] + scaled[2])
    mid = frames[0].shape[1] / 2.0
    on_left = cx < mid
    if (watch_side == "LEFT" and not on_left) or (watch_side == "RIGHT" and on_left):
        return None, None
    crops = []
    held = None
    for frame in frames:
        out = square_crop(frame, scaled)
        if out is None:
            return None, None
        crop, box = out
        crops.append(crop)
        held = box
    return np.stack(crops), held


def rgb_gap_message() -> str:
    return (
        "STOP: no decoded RealTalk frames for this 3 s figure. "
        "Need local avis in cache/teaser_windowed/ for gold_023 (shake) and "
        "gold_030 (nod). Will not download new RealTalk members. "
        "Do not use features/rgb16_windowed/ (withdrawn largest-face Haar)."
    )


def crop_listener(
    frames: np.ndarray,
    watch_side: str,
    video_id: str,
    person: str,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, tuple | None, str]:
    """Official RealTalk gold-person box first; side-aware Haar if needed."""
    boxes = listener_boxes(video_id, person, frame_indices)
    crops, held = held_realtalk_crops(frames, boxes, watch_side)
    if crops is not None:
        return crops, held, "RealTalk listener box (gold person)"
    try:
        payload = crop_target_person_rgb(frames, watch_side)
    except SystemExit as exc:
        payload = {
            "crop_status": "unresolved",
            "reason": str(exc),
            "rgb": None,
            "crop_box": None,
        }
    if payload.get("crop_status") == "resolved" and payload.get("rgb") is not None:
        rgb = np.asarray(payload["rgb"])
        if rgb.shape[0] != len(frames):
            idx = np.linspace(0, len(rgb) - 1, len(frames)).round().astype(int)
            rgb = rgb[idx]
        return rgb, tuple(payload["crop_box"]), "identity-fixed (annotator half)"
    raise SystemExit(
        "STOP: RealTalk listener boxes are missing for "
        f"{video_id} and identity-fixed crop unresolved "
        f"({payload.get('reason', 'no target-half face')}). "
        + rgb_gap_message()
    )


def events_in_window(path: Path, sample_id: str, t0: float, t1: float) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    frame = pd.read_csv(path)
    part = frame[frame["sample_id"].astype(str) == sample_id]
    return part[(part["end_sec"] > t0) & (part["start_sec"] < t1)].copy()


def load_panel_data(spec: dict, fetch: bool) -> dict:
    gold = load_gold_row(spec["sample_id"])
    video_id = str(gold["video_id"])
    person = str(gold["person"])
    sides = watch_sides()
    watch_side = sides[video_id]
    if watch_side != spec["watch_side"]:
        raise SystemExit(
            f"STOP: {spec['sample_id']} watch_side {watch_side} != {spec['watch_side']}"
        )
    expected = "LEFT" if person == "p0" else "RIGHT"
    if expected != watch_side:
        raise SystemExit(
            f"STOP: {spec['sample_id']} person={person} does not match {watch_side}"
        )
    pose_path = GOLD_DIR / f"{spec['sample_id']}.npz"
    if not pose_path.exists():
        raise SystemExit(f"STOP: missing {pose_path}")
    with np.load(pose_path, allow_pickle=True) as payload:
        rot = np.asarray(payload["rotation_xyz"], dtype=float)
        frames_abs = np.asarray(payload["frames"], dtype=int)
    clip_start = int(frames_abs.min())
    t0 = float(spec["start_sec"])
    t1 = t0 + WINDOW_SEC
    i0 = int(round(t0 * FPS))
    i1 = i0 + int(round(WINDOW_SEC * FPS))
    rot_win = rot[i0:i1]
    pose_win = rot_win[:, int(spec["axis"])]

    events_path = SHAKE_EVENTS if spec["kind"] == "shake" else NOD_EVENTS
    events = events_in_window(events_path, spec["sample_id"], t0, t1)
    if events.empty:
        raise SystemExit(
            f"STOP: {spec['sample_id']} {spec['kind']} window "
            f"{t0:.1f}-{t1:.1f}s has no annotated gesture"
        )
    spans = []
    for rec in events.itertuples(index=False):
        lo = max(float(rec.start_sec), t0) - t0
        hi = min(float(rec.end_sec), t1) - t0
        if hi > lo:
            spans.append((lo, hi))
    if not spans:
        raise SystemExit(f"STOP: {spec['sample_id']} has no in-window gesture span")
    lo, hi = max(spans, key=lambda p: p[1] - p[0])
    sample_rel = gesture_sample_times(lo, hi, N_FRAMES)
    frame_idx = window_indices(clip_start, t0, sample_rel)

    if not fetch:
        raise SystemExit(rgb_gap_message())
    scenes = fetch_scene_frames(video_id, frame_idx)
    crops, box, crop_source = crop_listener(
        scenes, watch_side, video_id, person, frame_idx
    )
    print(f"{spec['window_id']}: {crop_source}")

    sm = savgol_smooth(pose_win)
    amp, ia, ib = turning_pair(sm)
    rr = return_ratio(sm)
    gold_hit = "yes"
    rule_hit = rule_verdict(spec, amp, rr)
    cnn_hit = cnn_oof_verdict(spec["window_id"])

    return {
        **spec,
        "video_id": video_id,
        "person": person,
        "watch_side": watch_side,
        "pose": pose_win,
        "smoothed": sm,
        "rot": rot_win,
        "sample_rel": sample_rel,
        "t0": t0,
        "t1": t1,
        "scenes": scenes,
        "crops": crops,
        "box": box,
        "crop_source": crop_source,
        "events": events,
        "rule_amp": amp,
        "turn_a": ia,
        "turn_b": ib,
        "return_ratio": rr,
        "gold_verdict": gold_hit,
        "rule_verdict": rule_hit,
        "cnn_verdict": cnn_hit,
        "used_identity_npz": False,
    }


def _thin_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _event_spans(data: dict) -> list[tuple[float, float]]:
    t0, t1 = data["t0"], data["t1"]
    spans = []
    for rec in data["events"].itertuples(index=False):
        lo = max(float(rec.start_sec), t0) - t0
        hi = min(float(rec.end_sec), t1) - t0
        if hi > lo:
            spans.append((lo, hi))
    return spans


def draw_panel(fig, gs, data: dict) -> None:
    pose = data["pose"]
    sm = np.asarray(data["smoothed"], dtype=float)
    t_rel = np.arange(len(pose)) / FPS
    sample_rel = np.asarray(data["sample_rel"], dtype=float)
    n = len(sample_rel)
    sample_idx = np.clip(np.round(sample_rel * FPS).astype(int), 0, len(pose) - 1)
    spans = _event_spans(data)
    if n == 1:
        width = 0.36
    else:
        spacing = float(np.min(np.diff(np.sort(sample_rel))))
        width = min(0.36, 0.88 * spacing)

    axh = fig.add_subplot(gs[0, 0])
    axh.set_axis_off()
    axh.set_facecolor(PAPER)
    axh.text(
        0.0, 0.52, data["headline"],
        fontsize=10.5, color=INK, va="center", ha="left", transform=axh.transAxes,
    )

    axf = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[2, 0], sharex=axf)
    axv = fig.add_subplot(gs[3, 0])
    axm = fig.add_subplot(gs[4, 0])
    axf.set_xlim(0.0, WINDOW_SEC)
    axr.set_xlim(0.0, WINDOW_SEC)
    axf.set_ylim(0.0, 1.0)
    axf.set_facecolor(PAPER)
    axf.set_axis_off()
    scenes = data.get("scenes")
    if scenes is not None and len(scenes) > 0:
        mid = scenes[len(scenes) // 2]
        axf.imshow(
            mid,
            extent=(0.04, 0.88, 0.16, 1.0),
            aspect="auto",
            interpolation="bilinear",
            zorder=2,
        )
        axf.text(
            0.46, 0.07, "both people",
            ha="center", va="center", fontsize=6.5, color=MUTED,
        )
    for lo, hi in spans:
        axf.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
    for i in range(n):
        left = float(sample_rel[i]) - 0.5 * width
        right = float(sample_rel[i]) + 0.5 * width
        axf.imshow(
            data["crops"][i],
            extent=(left, right, 0.16, 1.0),
            aspect="auto",
            interpolation="bilinear",
            zorder=3,
        )
        for x in (left, right):
            axf.plot(
                [x, x], [0.16, 1.0],
                color=LINE, lw=0.4, zorder=4, solid_capstyle="butt",
            )
        axf.plot(
            [left, right], [0.16, 0.16],
            color=LINE, lw=0.4, zorder=4, solid_capstyle="butt",
        )
        axf.plot(
            [left, right], [1.0, 1.0],
            color=LINE, lw=0.4, zorder=4, solid_capstyle="butt",
        )
        axf.text(
            0.5 * (left + right), 0.07,
            f"{sample_rel[i]:.1f} s",
            ha="center", va="center", fontsize=7, color=MUTED,
        )
    axf.set_xlim(0.0, WINDOW_SEC)
    axf.set_ylim(0.0, 1.0)

    axr.set_facecolor(PAPER)
    finite = pose[np.isfinite(pose)]
    pad = max(2.0, 0.22 * float(np.nanmax(finite) - np.nanmin(finite))) if finite.size else 2.0
    ymin = float(np.nanmin(pose)) - pad
    ymax = float(np.nanmax(pose)) + pad
    axr.set_ylim(ymin, ymax)
    for lo, hi in spans:
        axr.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        axr.axvline(lo, color=INK, ls=(0, (3, 2.2)), lw=0.7, zorder=2)
    axr.plot(t_rel, pose, color=INK, lw=1.15, solid_capstyle="round", zorder=3)
    axr.scatter(
        sample_rel, pose[sample_idx],
        s=9, color=INK, zorder=4, linewidths=0,
    )
    _draw_rule(axr, data, t_rel, ymin, ymax)
    axr.set_xlim(0.0, WINDOW_SEC)
    axr.set_xticks([0.0, 1.0, 2.0, 3.0])
    axr.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    axr.tick_params(axis="both", length=2.5, width=0.6, labelsize=7, pad=2)
    axr.set_xlabel("Time (s)", fontsize=8, labelpad=3)
    axr.set_ylabel(data["pose_name"], fontsize=8, labelpad=4)
    _thin_spines(axr)

    axv.set_axis_off()
    axv.set_facecolor(PAPER)
    axv.text(
        0.0, 0.5,
        f"Gold {data['gold_verdict']}    "
        f"Rule {data['rule_verdict']}    "
        f"CNN {data['cnn_verdict']}",
        fontsize=8.5, color=INK, va="center", ha="left", transform=axv.transAxes,
    )
    axm.set_axis_off()
    axm.set_facecolor(PAPER)
    if data["kind"] == "shake":
        meta = (
            f"TEST  {data['t0']:.1f}-{data['t1']:.1f}s  watch {data['watch_side']}  "
            f"yaw tau {frozen_shake_tau():.2f} deg  amp {data['rule_amp']:.2f}"
        )
    else:
        meta = (
            f"TEST  {data['t0']:.1f}-{data['t1']:.1f}s  watch {data['watch_side']}  "
            f"amp {data['rule_amp']:.2f}  RR {data['return_ratio']:.3f}  "
            f"thr {NOD_AMP_THRESHOLD:.2f} / {NOD_RR_THRESHOLD:.2f}"
        )
    axm.text(
        0.0, 0.45, meta,
        fontsize=6.5, color=MUTED, va="center", ha="left", transform=axm.transAxes,
    )


def _draw_rule(ax, data: dict, t_rel: np.ndarray, ymin: float, ymax: float) -> None:
    sm = np.asarray(data["smoothed"], dtype=float)
    ia, ib = int(data["turn_a"]), int(data["turn_b"])
    y0, y1 = float(sm[ia]), float(sm[ib])
    lo, hi = _event_spans(data)[0]
    x_br = min(WINDOW_SEC - 0.12, hi + 0.10)
    ax.plot([x_br, x_br], [y0, y1], color=RED, ls=(0, (3, 2.2)), lw=0.85, zorder=5)
    ax.plot([x_br - 0.05, x_br + 0.05], [y0, y0], color=RED, lw=0.85, zorder=5)
    ax.plot([x_br - 0.05, x_br + 0.05], [y1, y1], color=RED, lw=0.85, zorder=5)
    if data["kind"] == "shake":
        tau = frozen_shake_tau()
        mid = 0.5 * (y0 + y1)
        ax.axhline(mid + 0.5 * tau, color=RED, ls=(0, (3, 2.2)), lw=0.85, zorder=2)
        ax.axhline(mid - 0.5 * tau, color=RED, ls=(0, (3, 2.2)), lw=0.85, zorder=2)
        ax.text(
            0.04, 0.96, "yaw rule threshold",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=RED,
        )
    else:
        ax.text(
            lo + 0.04, ymax - 0.12 * (ymax - ymin),
            "return ratio <= 0.21",
            ha="left", va="top", fontsize=6.5, color=MUTED, zorder=5,
        )


def plot_panels(rows: list[dict], out: Path) -> None:
    refuse_output(out)
    n_panel = len(rows)
    fig_w = 7.16
    left, right = 0.10, 0.97
    ax_w = fig_w * (right - left)
    tile_sec = 0.36
    frame_h = ax_w * (tile_sec / WINDOW_SEC)
    title_h = 0.28
    plot_h = 1.00
    verdict_h = 0.22
    meta_h = 0.16
    panel_gap = 0.38
    height = (
        n_panel * (title_h + frame_h + plot_h + verdict_h + meta_h)
        + max(0, n_panel - 1) * panel_gap
        + 0.55
    )
    fig = plt.figure(figsize=(fig_w, height), facecolor=PAPER)
    outer = fig.add_gridspec(
        n_panel, 1,
        hspace=0.30,
        left=left, right=right, top=0.96, bottom=0.06,
    )
    for i, data in enumerate(rows):
        inner = outer[i].subgridspec(
            5, 1,
            height_ratios=[title_h, frame_h, plot_h, verdict_h, meta_h],
            hspace=0.08,
        )
        draw_panel(fig, inner, data)

    save(fig, out)
    from PIL import Image

    jpg = out.with_suffix(".jpg")
    Image.open(out.with_suffix(".png")).convert("RGB").save(jpg, quality=92)
    print(f"wrote {jpg}")


def check_only() -> None:
    print("3 s windowed face teaser (not the 60 s teaser, not pose-only yaw)")
    print(f"output: {OUT_DEFAULT.with_suffix('.png')}")
    for spec in PANELS:
        gold = load_gold_row(spec["sample_id"])
        print(
            f"{spec['kind']:5} {spec['window_id']:18} "
            f"{gold['video_id']} watch {spec['watch_side']} "
            f"{spec['start_sec']:.1f}-{spec['start_sec'] + 3:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="print sources; do not download")
    p.add_argument("--nod-only", action="store_true")
    p.add_argument("--shake-only", action="store_true")
    p.add_argument("--no-fetch", action="store_true",
                   help="refuse to run without decoded frames")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    refuse_output(args.out)
    if args.check:
        check_only()
        return
    specs = list(PANELS)
    if args.nod_only and args.shake_only:
        raise SystemExit("STOP: choose one of --nod-only / --shake-only")
    if args.nod_only:
        specs = [s for s in specs if s["kind"] == "nod"]
    if args.shake_only:
        specs = [s for s in specs if s["kind"] == "shake"]
    rows = []
    errors = []
    for spec in specs:
        try:
            rows.append(load_panel_data(spec, fetch=not args.no_fetch))
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            errors.append(str(exc))
            if args.no_fetch or "pip install" in str(exc):
                raise
            print(f"skipping {spec['window_id']}", file=sys.stderr)
    if not rows:
        raise SystemExit(errors[0] if errors else rgb_gap_message())
    CACHE.mkdir(parents=True, exist_ok=True)
    plot_panels(rows, args.out)
    kinds = [row["kind"] for row in rows]
    print("panels:", ", ".join(kinds))
    print("clip ids:", ", ".join(f"{row['sample_id']} {row['t0']:.1f}-{row['t1']:.1f}s" for row in rows))
    print("crop sources:", "; ".join(row["crop_source"] for row in rows))
    for row in rows:
        print(
            f"{row['kind']} frames at "
            + ", ".join(f"{t:.2f}s" for t in row["sample_rel"])
            + f"; rule amp={row['rule_amp']:.2f} RR={row['return_ratio']:.3f}"
        )
    print(
        "nod alignment: event times match start_frame_relative = start_sec * 25; "
        "the annotated band covers the trough and return (not a 1 s index bug)"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
