"""Frame-level head-pose time series + derivatives."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .emoca_loader import find_pose_array, load_pickle, rotvec_to_euler_deg

HEADPOSE_COLUMNS = [
    "time_s",
    "frame_idx",
    "person",
    "pitch",
    "yaw",
    "roll",
    "pitch_velocity",
    "yaw_velocity",
    "roll_velocity",
    "pitch_acceleration",
    "yaw_acceleration",
    "roll_acceleration",
    "tracking_confidence",
    "valid",
]


def interpolate_short_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(float).copy()
    valid = valid.astype(bool).copy()
    n = len(x)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap:
            x[i:j] = np.linspace(x[i - 1], x[j], gap + 2)[1:-1]
            valid[i:j] = True
        i = j
    return x, valid


def extract_person_pose(pkl_path: Path, person: str, fps: float) -> pd.DataFrame:
    obj = load_pickle(pkl_path)
    if not isinstance(obj, dict):
        raise ValueError(f"EMOCA pickle is {type(obj)}, expected frame dict")
    frames: list[int] = []
    for k in obj:
        try:
            frames.append(int(k))
        except (TypeError, ValueError):
            continue
    frames.sort()
    pitch = np.zeros(len(frames))
    yaw = np.zeros(len(frames))
    roll = np.zeros(len(frames))
    valid = np.zeros(len(frames), dtype=bool)
    for i, fi in enumerate(frames):
        rec = obj.get(fi, obj.get(str(fi)))
        if not isinstance(rec, dict):
            continue
        emb = rec.get(person, rec)
        aa = find_pose_array(emb)
        if aa is None or aa.size < 3:
            continue
        p, y, r = rotvec_to_euler_deg(aa[:3])
        pitch[i], yaw[i], roll[i] = p, y, r
        valid[i] = True
    pitch, valid = interpolate_short_gaps(pitch, valid)
    yaw, _ = interpolate_short_gaps(yaw, valid)
    roll, _ = interpolate_short_gaps(roll, valid)
    dt = 1.0 / float(fps)
    pv, pa = np.gradient(pitch, dt), None
    yv = np.gradient(yaw, dt)
    rv = np.gradient(roll, dt)
    pa = np.gradient(pv, dt)
    ya = np.gradient(yv, dt)
    ra = np.gradient(rv, dt)
    t = np.array(frames, dtype=float) / float(fps)
    # if frames were reindexed to 0..N-1, time is still frame/fps
    if frames and min(frames) != 0:
        t = (np.array(frames, dtype=float) - min(frames)) / float(fps)
    df = pd.DataFrame(
        {
            "time_s": t,
            "frame_idx": frames,
            "person": person,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "pitch_velocity": pv,
            "yaw_velocity": yv,
            "roll_velocity": rv,
            "pitch_acceleration": pa,
            "yaw_acceleration": ya,
            "roll_acceleration": ra,
            "tracking_confidence": valid.astype(float),
            "valid": valid.astype(np.uint8),
        }
    )
    return df
