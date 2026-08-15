"""Shared EMOCA/FLAME pose helpers for the tiny-subset nod pipeline.

EMOCA typically stores a 6D pose vector per person:
  pose[0:3]  global / neck rotation as axis-angle (radians)
  pose[3:6]  jaw rotation

Nodding is mainly **pitch** (up/down). We convert axis-angle → Euler XYZ
and take rx as pitch, ry as yaw, rz as roll. If your plots look like
shakes instead of nods, swap axes with --pitch-axis on extract.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

FPS_DEFAULT = 25.0
CLIP_SECONDS = 60.0

POSE_KEYS = (
    "pose",
    "global_pose",
    "poseparams",
    "pose_params",
    "rot",
    "rotation",
    "neck_pose",
    "global_rot",
)


def axis_angle_to_euler_xyz(aa: np.ndarray) -> tuple[float, float, float]:
    """Rodrigues axis-angle (3,) → Euler XYZ in radians (pitch, yaw, roll)."""
    aa = np.asarray(aa, dtype=float).reshape(-1)[:3]
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return 0.0, 0.0, 0.0
    k = aa / theta
    kx, ky, kz = k
    c, s = math.cos(theta), math.sin(theta)
    v = 1.0 - c
    r00 = c + kx * kx * v
    r01 = kx * ky * v - kz * s
    r02 = kx * kz * v + ky * s
    r10 = ky * kx * v + kz * s
    r11 = c + ky * ky * v
    r12 = ky * kz * v - kx * s
    r20 = kz * kx * v - ky * s
    r21 = kz * ky * v + kx * s
    r22 = c + kz * kz * v
    # XYZ intrinsic: pitch=x, yaw=y, roll=z
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        pitch = math.atan2(r21, r22)
        yaw = math.atan2(-r20, sy)
        roll = math.atan2(r10, r00)
    else:
        pitch = math.atan2(-r12, r11)
        yaw = math.atan2(-r20, sy)
        roll = 0.0
    return pitch, yaw, roll


def extract_axis_angle(emb: Any) -> Optional[np.ndarray]:
    """Pull a 3-vector rotation from a nested EMOCA embedding dict."""
    if emb is None:
        return None
    if isinstance(emb, dict):
        for key in POSE_KEYS:
            if key in emb:
                return extract_axis_angle(emb[key])
        # common nested 'flame' / 'params'
        for key in ("flame", "params", "code", "expcode"):
            if key in emb:
                found = extract_axis_angle(emb[key])
                if found is not None:
                    return found
        return None
    arr = np.asarray(emb, dtype=float).reshape(-1)
    if arr.size >= 6:
        return arr[:3]
    if arr.size >= 3:
        return arr[:3]
    return None


def euler_degrees(aa: np.ndarray, pitch_axis: int = 0) -> tuple[float, float, float]:
    pitch, yaw, roll = axis_angle_to_euler_xyz(aa)
    eulers = [pitch, yaw, roll]
    # Re-order if user says pitch is not rx
    if pitch_axis == 0:
        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
    if pitch_axis == 1:
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
    return math.degrees(roll), math.degrees(yaw), math.degrees(pitch)


def load_emoca_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def frame_to_timestamp(frame: int, fps: float = FPS_DEFAULT, origin_frame: int = 0) -> float:
    return (int(frame) - int(origin_frame)) / float(fps)


def format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def list_clip_dirs(subset_root: Path) -> list[Path]:
    if not subset_root.exists():
        return []
    return sorted(p for p in subset_root.iterdir() if p.is_dir() and (p / "meta.json").exists())


def read_meta(clip_dir: Path) -> dict:
    return json.loads((clip_dir / "meta.json").read_text())


def bandpass(x: np.ndarray, fps: float, low: float = 1.0, high: float = 3.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    if x.size < 16:
        return np.zeros_like(x)
    nyq = 0.5 * fps
    high = min(high, nyq * 0.99)
    low = max(low, 1e-3)
    if low >= high:
        return np.zeros_like(x)
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def synthetic_pitch(n_frames: int, fps: float = FPS_DEFAULT, n_nods: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    """Make a 60s-ish pitch series with a few clear nod cycles."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    pitch = 2.0 * np.sin(2 * np.pi * 0.15 * t) + rng.normal(0, 0.15, n_frames)
    yaw = rng.normal(0, 0.4, n_frames)
    roll = rng.normal(0, 0.25, n_frames)
    # inject nods: 2 Hz, ~0.7 s, amplitude ~8 deg
    for i in range(n_nods):
        t0 = 4.0 + i * (n_frames / fps - 8.0) / max(n_nods, 1)
        mask = (t >= t0) & (t <= t0 + 0.7)
        pitch = pitch + mask * (-8.0 * np.sin(2 * np.pi * 2.0 * (t - t0)))
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t}
