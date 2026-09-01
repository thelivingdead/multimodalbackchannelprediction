#!/usr/bin/env python3
"""
Visualise RealTalk video frames against reported FLAME / EMOCA values.

What it does
------------
1. Reads a dyadic (or single-face) video.
2. Loads per-frame pose parameters (pitch / yaw / roll / translation / expression).
3. Aligns frames ↔ parameter rows (by fps / frame index).
4. Builds:
   - time-series plots of reported values with event markers
   - contact-sheet of sampled frames with overlaid numeric values
   - event gallery: frames at detected nod/shake peaks
5. Writes a CSV of detected events for annotation QA.

Run on VS Code Remote / lab GPU (GPU not required for this script):

  pip install -r scripts/requirements.txt

  # Demo mode (no RealTalk mount needed):
  python scripts/visualise_flame_vs_frames.py --demo --out outputs/viz_demo

  # Real data:
  python scripts/visualise_flame_vs_frames.py \\
      --video /path/to/clip.mp4 \\
      --flame /path/to/params.npz \\
      --fps 25 \\
      --out outputs/viz_clip01

NPZ expected keys (flexible — script auto-maps common aliases):
  pitch, yaw, roll, trans_z, brow   OR
  pose (N,3) / rotation (N,3) / expression (N,D)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "numpy is required. On the lab machine run:\n"
        "  python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -U pip && pip install -r scripts/requirements.txt"
    ) from e

# ---------------------------------------------------------------------------
# Optional heavy deps imported lazily so --help works without OpenCV installed
# ---------------------------------------------------------------------------


def _require_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _require_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "OpenCV is required for real --video runs.\n"
            "  pip install 'opencv-python-headless==4.10.0.84'\n"
            "Demo mode (--demo) does not need OpenCV."
        ) from e
    return cv2


# ---------------------------------------------------------------------------
# FLAME / parameter loading
# ---------------------------------------------------------------------------

ALIAS = {
    "pitch": ["pitch", "rx", "rot_x", "pose_0", "neck_pitch"],
    "yaw": ["yaw", "ry", "rot_y", "pose_1", "neck_yaw"],
    "roll": ["roll", "rz", "rot_z", "pose_2", "neck_roll"],
    "trans_z": ["trans_z", "tz", "translation_z", "cam_t_z", "depth"],
    "brow": ["brow", "brow_raise", "exp_brow", "au01", "AU01"],
}


def _first_key(d: dict, names: list[str]) -> Optional[str]:
    lower = {k.lower(): k for k in d}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def load_flame(path: Optional[Path], n_frames: Optional[int] = None, fps: float = 25.0) -> dict[str, Any]:
    if path is None:
        raise FileNotFoundError("No FLAME path provided")

    path = Path(path)
    if path.suffix.lower() == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
    elif path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 3:
            raw = {"pose": arr[:, :3], "expression": arr[:, 3:] if arr.shape[1] > 3 else None}
        else:
            raise ValueError(f"Unsupported npy shape: {getattr(arr, 'shape', None)}")
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        raw = {k: np.asarray(v) for k, v in raw.items()}
    else:
        raise ValueError(f"Unsupported flame file type: {path.suffix}")

    # Flatten nested dicts of arrays
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            flat[k] = v
        else:
            try:
                flat[k] = np.asarray(v)
            except Exception:
                pass

    out: dict[str, Any] = {"fps": fps}

    # Direct aliases
    for canon, names in ALIAS.items():
        key = _first_key(flat, names)
        if key is not None:
            out[canon] = np.asarray(flat[key], dtype=float).reshape(-1)

    # pose / rotation (N,3) → pitch,yaw,roll (axis-angle or Euler — treat as ordered rx,ry,rz)
    pose_key = _first_key(flat, ["pose", "rotation", "rot", "neck_pose", "global_pose"])
    if pose_key is not None:
        pose = np.asarray(flat[pose_key], dtype=float)
        if pose.ndim == 2 and pose.shape[1] >= 3:
            if "pitch" not in out:
                out["pitch"] = pose[:, 0]
            if "yaw" not in out:
                out["yaw"] = pose[:, 1]
            if "roll" not in out:
                out["roll"] = pose[:, 2]

    # translation (N,3)
    t_key = _first_key(flat, ["translation", "trans", "cam_t", "t"])
    if t_key is not None and "trans_z" not in out:
        t = np.asarray(flat[t_key], dtype=float)
        if t.ndim == 2 and t.shape[1] >= 3:
            out["trans_z"] = t[:, 2]
        elif t.ndim == 1:
            out["trans_z"] = t

    # expression → brow proxy = L2 of first few coeffs or mean abs
    e_key = _first_key(flat, ["expression", "exp", "shape", "exp_code"])
    if e_key is not None and "brow" not in out:
        exp = np.asarray(flat[e_key], dtype=float)
        if exp.ndim == 2:
            out["brow"] = np.mean(np.abs(exp[:, : min(5, exp.shape[1])]), axis=1)
        else:
            out["brow"] = np.abs(exp)

    # Fill missing with zeros once we know length
    length = None
    for k in ("pitch", "yaw", "roll", "trans_z", "brow"):
        if k in out:
            length = len(out[k])
            break
    if length is None:
        length = n_frames or 100
    for k in ("pitch", "yaw", "roll", "trans_z", "brow"):
        if k not in out:
            out[k] = np.zeros(length, dtype=float)
        out[k] = np.asarray(out[k], dtype=float).reshape(-1)

    # Truncate / pad to common length
    length = min(len(out[k]) for k in ("pitch", "yaw", "roll", "trans_z", "brow"))
    if n_frames is not None:
        length = min(length, n_frames)
    for k in ("pitch", "yaw", "roll", "trans_z", "brow"):
        out[k] = out[k][:length]

    out["n_frames"] = length
    return out


def make_demo_flame(n_frames: int = 200, fps: float = 25.0) -> dict[str, Any]:
    t = np.arange(n_frames) / fps
    pitch = 0.01 * np.sin(2 * np.pi * 0.3 * t)
    # inject a clear nod around 2.0–3.2 s
    nod_mask = (t >= 2.0) & (t <= 3.2)
    pitch = pitch + nod_mask * 0.08 * np.sin(2 * np.pi * 2.0 * t)
    yaw = 0.01 * np.sin(2 * np.pi * 0.2 * t)
    shake_mask = (t >= 5.0) & (t <= 6.0)
    yaw = yaw + shake_mask * 0.07 * np.sin(2 * np.pi * 1.8 * t)
    roll = 0.02 * np.sin(2 * np.pi * 0.15 * t)
    trans_z = 0.002 * np.cumsum(np.random.default_rng(0).normal(0, 1, n_frames))
    brow = np.clip(0.05 + 0.4 * ((t >= 7.0) & (t <= 7.6)), 0, None)
    return {
        "fps": fps,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "trans_z": trans_z,
        "brow": brow,
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# Detectors (same spirit as api/label_rules.py)
# ---------------------------------------------------------------------------


def bandpass(x: np.ndarray, fps: float, low: float, high: float, order: int = 3) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    if x.size < 16:
        return np.zeros_like(x)
    nyq = 0.5 * fps
    high = min(high, nyq * 0.99)
    low = max(low, 1e-3)
    if low >= high:
        return np.zeros_like(x)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def find_event_peaks(signal: np.ndarray, fps: float, low: float, high: float, min_height_deg: float = 1.0):
    """Return peak frame indices for oscillatory events. min_height_deg in degrees if signal is radians*180/pi or degrees."""
    from scipy.signal import find_peaks

    filt = bandpass(signal, fps, low, high)
    height = max(np.deg2rad(min_height_deg) if np.max(np.abs(signal)) < 1.0 else min_height_deg, float(np.std(filt) * 0.5))
    peaks, props = find_peaks(np.abs(filt), height=height)
    return peaks, filt, props


def detect_events(flame: dict[str, Any], min_height_deg: float = 1.0) -> list[dict[str, Any]]:
    fps = float(flame["fps"])
    events: list[dict[str, Any]] = []

    nod_peaks, nod_f, _ = find_event_peaks(flame["pitch"], fps, 1.0, 3.0, min_height_deg)
    for p in nod_peaks:
        events.append({"frame": int(p), "time_s": p / fps, "type": "nod", "value": float(flame["pitch"][p]), "filtered": float(nod_f[p])})

    shake_peaks, shake_f, _ = find_event_peaks(flame["yaw"], fps, 0.8, 2.5, min_height_deg)
    for p in shake_peaks:
        events.append({"frame": int(p), "time_s": p / fps, "type": "shake", "value": float(flame["yaw"][p]), "filtered": float(shake_f[p])})

    # brow: simple threshold
    brow = flame["brow"]
    thr = max(0.35, float(np.mean(brow) + 2 * np.std(brow)))
    above = np.where(brow > thr)[0]
    if above.size:
        # take segment starts
        starts = [int(above[0])]
        for i in range(1, len(above)):
            if above[i] != above[i - 1] + 1:
                starts.append(int(above[i]))
        for s in starts:
            events.append({"frame": s, "time_s": s / fps, "type": "eyebrow_raise", "value": float(brow[s]), "filtered": float(brow[s])})

    events.sort(key=lambda e: e["frame"])
    return events


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def read_video_meta(path: Path) -> tuple[float, int, int, int]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return float(fps), n, w, h


def sample_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    want = set(indices)
    got: dict[int, np.ndarray] = {}
    i = 0
    while want and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            want.remove(i)
        i += 1
    cap.release()
    return got


def make_demo_frames(n_frames: int, w: int = 320, h: int = 240) -> dict[int, np.ndarray]:
    frames = {}
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = 30
        img[:, :, 1] = 40
        img[:, :, 2] = 55
        # moving "face" circle so motion is visible
        cx = int(w * (0.35 + 0.15 * np.sin(i / 8)))
        cy = int(h * (0.45 + 0.08 * np.sin(i / 5)))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 35**2
        img[mask] = (220, 190, 160)
        frames[i] = img
    return frames


def overlay_values(frame: np.ndarray, flame_row: dict[str, float], frame_idx: int, event_type: str = "") -> np.ndarray:
    cv2 = _require_cv2()
    img = frame.copy()
    # draw on BGR for putText then convert back — work in RGB with simple rectangle
    lines = [
        f"f={frame_idx}",
        f"pitch={flame_row['pitch']:+.4f}",
        f"yaw  ={flame_row['yaw']:+.4f}",
        f"roll ={flame_row['roll']:+.4f}",
        f"tz   ={flame_row['trans_z']:+.4f}",
        f"brow ={flame_row['brow']:+.4f}",
    ]
    if event_type:
        lines.append(f"EVENT: {event_type}")
    # convert to BGR for OpenCV text
    bgr = img[:, :, ::-1].copy()
    y0 = 18
    for line in lines:
        cv2.putText(bgr, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y0 += 16
    return bgr[:, :, ::-1]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_timeseries(flame: dict[str, Any], events: list[dict[str, Any]], out_path: Path) -> None:
    plt = _require_plotting()
    fps = float(flame["fps"])
    n = int(flame["n_frames"])
    t = np.arange(n) / fps

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    series = [
        ("pitch", "Pitch (nod axis)", "nod"),
        ("yaw", "Yaw (shake axis)", "shake"),
        ("roll", "Roll (tilt axis)", None),
        ("brow", "Brow proxy", "eyebrow_raise"),
    ]
    for ax, (key, title, etype) in zip(axes, series):
        ax.plot(t, flame[key], color="#1f4e5f", lw=1.2, label="reported value")
        if key in ("pitch", "yaw"):
            low, high = (1.0, 3.0) if key == "pitch" else (0.8, 2.5)
            filt = bandpass(flame[key], fps, low, high)
            ax.plot(t, filt, color="#b4532a", lw=1.0, alpha=0.85, label="band-pass")
        for e in events:
            if etype and e["type"] == etype:
                ax.axvline(e["time_s"], color="#0f6a5c", alpha=0.35, lw=1)
                ax.scatter([e["time_s"]], [e["value"]], color="#0f6a5c", s=28, zorder=3)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Reported FLAME values vs detected oscillatory events", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_contact_sheet(
    frames: dict[int, np.ndarray],
    flame: dict[str, Any],
    indices: list[int],
    events_by_frame: dict[int, str],
    out_path: Path,
    cols: int = 4,
) -> None:
    plt = _require_plotting()
    indices = [i for i in indices if i in frames]
    if not indices:
        return
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, indices):
        row = {
            "pitch": float(flame["pitch"][idx]),
            "yaw": float(flame["yaw"][idx]),
            "roll": float(flame["roll"][idx]),
            "trans_z": float(flame["trans_z"][idx]),
            "brow": float(flame["brow"][idx]),
        }
        img = overlay_values(frames[idx], row, idx, events_by_frame.get(idx, ""))
        ax.imshow(img)
        ax.set_title(f"t={idx / flame['fps']:.2f}s", fontsize=9)
        ax.axis("off")
    fig.suptitle("Frame samples with overlaid reported FLAME values", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_value_vs_frame_strip(flame: dict[str, Any], frames: dict[int, np.ndarray], center: int, out_path: Path, window: int = 8) -> None:
    """Compare a local burst of frames against the pitch curve (good for nod QA)."""
    plt = _require_plotting()
    fps = float(flame["fps"])
    n = int(flame["n_frames"])
    lo = max(0, center - window)
    hi = min(n - 1, center + window)
    idxs = list(range(lo, hi + 1, max(1, (hi - lo) // 6)))
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, len(idxs), height_ratios=[1.2, 1])

    ax_ts = fig.add_subplot(gs[0, :])
    t = np.arange(n) / fps
    ax_ts.plot(t, flame["pitch"], label="pitch")
    ax_ts.axvspan(lo / fps, hi / fps, color="#0f6a5c", alpha=0.15)
    ax_ts.axvline(center / fps, color="#b4532a", ls="--", label="focus frame")
    ax_ts.set_title(f"Pitch around frame {center} ({center / fps:.2f}s)")
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.25)

    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(gs[1, i])
        if idx in frames:
            row = {
                "pitch": float(flame["pitch"][idx]),
                "yaw": float(flame["yaw"][idx]),
                "roll": float(flame["roll"][idx]),
                "trans_z": float(flame["trans_z"][idx]),
                "brow": float(flame["brow"][idx]),
            }
            ax.imshow(overlay_values(frames[idx], row, idx))
        ax.set_title(f"{idx}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def compare_reported_vs_finite_diff(flame: dict[str, Any], out_path: Path) -> None:
    """Sanity check: reported pitch vs finite-difference 'motion energy' proxy."""
    plt = _require_plotting()
    pitch = flame["pitch"]
    motion = np.abs(np.gradient(pitch))
    fps = float(flame["fps"])
    t = np.arange(len(pitch)) / fps
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, pitch, label="reported pitch")
    ax.plot(t, motion, label="|Δ pitch|", alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.legend()
    ax.set_title("Reported pitch vs frame-to-frame change (sanity check)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise video frames vs reported FLAME values")
    p.add_argument("--video", type=Path, default=None, help="Path to mp4/avi")
    p.add_argument("--flame", type=Path, default=None, help="Path to npz/npy/json FLAME params")
    p.add_argument("--fps", type=float, default=None, help="Override fps (else from video / 25)")
    p.add_argument("--out", type=Path, default=Path("outputs/flame_viz"))
    p.add_argument("--demo", action="store_true", help="Synthetic demo (no files needed)")
    p.add_argument("--max-frames", type=int, default=300, help="Cap frames processed for speed")
    p.add_argument("--min-height-deg", type=float, default=1.0, help="Peak amplitude threshold (degrees)")
    p.add_argument("--every", type=int, default=15, help="Sample every Nth frame for contact sheet")
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.demo or (args.video is None and args.flame is None):
        print("Running DEMO mode (synthetic FLAME + frames)")
        fps = args.fps or 25.0
        flame = make_demo_flame(n_frames=min(args.max_frames, 200), fps=fps)
        frames = make_demo_frames(flame["n_frames"])
        video_path = None
    else:
        if args.video is None or args.flame is None:
            raise SystemExit("Provide both --video and --flame, or use --demo")
        v_fps, v_n, _, _ = read_video_meta(args.video)
        fps = float(args.fps or v_fps or 25.0)
        flame = load_flame(args.flame, n_frames=min(args.max_frames, v_n or args.max_frames), fps=fps)
        flame["fps"] = fps
        # sample indices first, then read only those frames
        n = flame["n_frames"]
        sample_idx = list(range(0, n, max(1, args.every)))
        events_tmp = detect_events(flame, min_height_deg=args.min_height_deg)
        sample_idx += [e["frame"] for e in events_tmp]
        # local windows around first few events
        for e in events_tmp[:5]:
            sample_idx += list(range(max(0, e["frame"] - 6), min(n, e["frame"] + 7)))
        sample_idx = sorted(set(i for i in sample_idx if 0 <= i < n))
        print(f"Reading {len(sample_idx)} frames from {args.video}")
        frames = sample_frames(args.video, sample_idx)
        video_path = args.video

    events = detect_events(flame, min_height_deg=args.min_height_deg)
    events_by_frame = {e["frame"]: e["type"] for e in events}

    # Save events CSV
    csv_path = out / "detected_events.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "type", "value", "filtered"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    # Summary JSON (for dissertation appendix)
    summary = {
        "video": str(video_path) if video_path else "DEMO",
        "n_frames": flame["n_frames"],
        "fps": flame["fps"],
        "n_events": len(events),
        "events_by_type": {k: sum(1 for e in events if e["type"] == k) for k in ("nod", "shake", "eyebrow_raise")},
        "pitch_mean": float(np.mean(flame["pitch"])),
        "pitch_std": float(np.std(flame["pitch"])),
        "yaw_std": float(np.std(flame["yaw"])),
        "note": "Detector is precision-leaning; subtle events below threshold may be missed.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_timeseries(flame, events, out / "timeseries_events.png")
    compare_reported_vs_finite_diff(flame, out / "pitch_vs_delta.png")

    n = flame["n_frames"]
    contact_idx = list(range(0, n, max(1, args.every)))
    contact_idx += [e["frame"] for e in events[:12]]
    contact_idx = sorted(set(i for i in contact_idx if i in frames))[:24]
    plot_contact_sheet(frames, flame, contact_idx, events_by_frame, out / "contact_sheet_overlay.png")

    # Event focus strips
    for i, e in enumerate(events[:5]):
        # ensure local frames exist in demo (all frames present) or fetched set
        local = list(range(max(0, e["frame"] - 8), min(n, e["frame"] + 9)))
        missing = [j for j in local if j not in frames]
        if missing and video_path is not None:
            frames.update(sample_frames(video_path, missing))
        elif missing and video_path is None:
            # demo already has all
            pass
        plot_value_vs_frame_strip(flame, frames, e["frame"], out / f"event_{i:02d}_{e['type']}_strip.png")

    print("Wrote:")
    for fp in sorted(out.glob("*")):
        print(" ", fp)
    print("\nSummary:", json.dumps(summary, indent=2))
    print(
        "\nQA tip: open contact_sheet_overlay.png and event_*_strip.png. "
        "If numbers move but the face does not (or vice versa), fix axis mapping / fps alignment."
    )


if __name__ == "__main__":
    main()
