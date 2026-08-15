"""Load RealTalk-style EMOCA pickles without assuming unverified extra keys."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

POSE_KEY_CANDIDATES = (
    "pose",
    "global_pose",
    "poseparams",
    "pose_params",
    "rot",
    "rotation",
    "neck_pose",
    "global_rot",
)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def find_pose_array(emb: Any) -> np.ndarray | None:
    if emb is None:
        return None
    if hasattr(emb, "detach"):
        try:
            emb = emb.detach().cpu().numpy()
        except Exception:
            return None
    if isinstance(emb, dict):
        for k in POSE_KEY_CANDIDATES:
            if k in emb:
                found = find_pose_array(emb[k])
                if found is not None:
                    return found
        for k, v in emb.items():
            if str(k).lower() in ("flame", "params", "code", "emoca"):
                found = find_pose_array(v)
                if found is not None:
                    return found
        return None
    arr = np.asarray(emb, dtype=float).reshape(-1)
    if arr.size >= 3:
        return arr
    return None


def rotvec_to_euler_deg(aa: np.ndarray) -> tuple[float, float, float]:
    """Axis-angle radians → Euler XYZ degrees. Convention: pitch, yaw, roll.

    Status: LIKELY (standard EMOCA 6D pose[0:3] global rotvec). Confirm with 03_inspect_emoca.py.
    """
    aa = np.asarray(aa, dtype=float).reshape(-1)[:3]
    eul = Rotation.from_rotvec(aa).as_euler("xyz", degrees=True)
    return float(eul[0]), float(eul[1]), float(eul[2])


def summarize_structure(obj: Any, max_depth: int = 4) -> dict[str, Any]:
    """Compact nested-type summary for schema reports (no huge arrays)."""

    def rec(x: Any, depth: int) -> Any:
        if depth > max_depth:
            return "..."
        if isinstance(x, dict):
            keys = list(x.keys())[:24]
            sample = keys[0] if keys else None
            out: dict[str, Any] = {"type": "dict", "n": len(x), "keys_sample": [str(k) for k in keys]}
            if sample is not None:
                out["child"] = rec(x[sample], depth + 1)
            return out
        if isinstance(x, (list, tuple)):
            return {"type": type(x).__name__, "n": len(x), "child": rec(x[0], depth + 1) if x else None}
        if isinstance(x, np.ndarray):
            return {"type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
        shape = getattr(x, "shape", None)
        if shape is not None:
            return {"type": type(x).__name__, "shape": list(shape)}
        return {"type": type(x).__name__, "repr": repr(x)[:80]}

    return rec(obj, 0)
