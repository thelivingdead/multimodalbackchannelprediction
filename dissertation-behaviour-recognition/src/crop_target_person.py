"""Target-person RGB crop for 3 s windows.

The withdrawn VideoMAE crops used scripts/fetch_rgb_windows.py:crop_window,
which takes the largest Haar face on the middle frame and ignores the gold
person field. This module does the opposite: it is given the annotated side
explicitly, keeps only detections on that half, holds one box for the whole
window, and returns crop_status=unresolved instead of guessing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

CROP_SIZE = 224
CROP_SCALE = 1.6
MIN_DETECTIONS = 2
MAX_CENTRE_JUMP = 0.12


def _cascade():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "STOP: opencv-python-headless is not installed."
        ) from exc
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise SystemExit(f"STOP: Haar cascade failed to load from {path}")
    return cv2, cascade


def split_faces(faces: np.ndarray, width: int, side: str) -> tuple[list, list]:
    """Split Haar boxes into the annotated half and the other half."""
    if side not in {"LEFT", "RIGHT"}:
        raise SystemExit(f"STOP: watch_side must be LEFT or RIGHT, got {side!r}")
    midline = width / 2.0
    target: list[tuple[float, float, float, int]] = []
    other: list[tuple[float, float, float, int]] = []
    for x, y, w, h in faces:
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        face_side = float(max(w, h))
        area = int(w) * int(h)
        row = (cx, cy, face_side, area)
        on_left = cx < midline
        if (side == "LEFT" and on_left) or (side == "RIGHT" and not on_left):
            target.append(row)
        else:
            other.append(row)
    return target, other


def square_box(
    cx: float, cy: float, face_side: float, width: int, height: int,
    scale: float = CROP_SCALE,
) -> tuple[int, int, int, int]:
    box_side = int(round(face_side * scale))
    box_side = min(box_side, width, height)
    x0 = int(min(max(round(cx) - box_side // 2, 0), width - box_side))
    y0 = int(min(max(round(cy) - box_side // 2, 0), height - box_side))
    return (x0, y0, box_side, box_side)


def on_wrong_half(cx: float, width: int, side: str) -> bool:
    midline = width / 2.0
    if side == "LEFT":
        return cx >= midline
    return cx < midline


def crop_target_person_rgb(
    frames: np.ndarray,
    watch_side: str,
    *,
    crop_scale: float = CROP_SCALE,
    crop_size: int = CROP_SIZE,
) -> dict[str, Any]:
    """Crop one 3 s window to the annotated participant.

    ``frames`` is (n, H, W, 3) uint8 RGB. One box is chosen from detections
    on the annotated half across the sampled frames, then applied to every
    frame so the crop cannot switch person mid-window.

    Returns a dict with crop_status 'resolved' or 'unresolved'. Resolved
    windows include rgb (n, 224, 224, 3). Unresolved windows have rgb=None
    and never silently take the largest face in the full frame.
    """
    cv2, cascade = _cascade()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise SystemExit(f"STOP: expected (n, H, W, 3) RGB, got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    centres: list[tuple[float, float, float]] = []
    n_other = 0
    for frame in frames:
        faces = cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        target, other = split_faces(np.asarray(faces), width, watch_side)
        n_other += len(other)
        if target:
            cx, cy, face_side, _ = max(target, key=lambda t: t[3])
            centres.append((cx, cy, face_side))

    reason = ""
    status = "resolved"
    if len(centres) < MIN_DETECTIONS:
        status = "unresolved"
        reason = (
            f"only {len(centres)} target-half detections "
            f"(need {MIN_DETECTIONS})"
        )
    else:
        stack = np.asarray(centres, dtype=float)
        cx = float(np.median(stack[:, 0]))
        cy = float(np.median(stack[:, 1]))
        face_side = float(np.median(stack[:, 2]))
        jump = float(np.max(np.abs(stack[:, 0] - cx)) / max(width, 1))
        if jump > MAX_CENTRE_JUMP:
            status = "unresolved"
            reason = f"target-half centre jumped {jump:.3f} of frame width"
        elif on_wrong_half(cx, width, watch_side):
            status = "unresolved"
            reason = "median centre landed on the excluded half"

    result: dict[str, Any] = {
        "crop_status": status,
        "watch_side": watch_side,
        "n_target_detections": int(len(centres)),
        "n_other_detections": int(n_other),
        "frame_width": width,
        "frame_height": height,
        "reason": reason,
        "rgb": None,
        "crop_box": None,
        "crop_centre_x": None,
        "selected_side": None,
    }
    if status != "resolved":
        return result

    box = square_box(cx, cy, face_side, width, height, scale=crop_scale)
    x0, y0, box_side, _ = box
    crops = np.stack(
        [
            cv2.resize(
                frame[y0 : y0 + box_side, x0 : x0 + box_side],
                (crop_size, crop_size),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
    )
    result.update(
        {
            "rgb": crops.astype(np.uint8),
            "crop_box": box,
            "crop_centre_x": float(x0 + box_side / 2.0),
            "selected_side": watch_side,
        }
    )
    return result
