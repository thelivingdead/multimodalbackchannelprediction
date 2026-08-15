"""Heuristic / rule-based 7-class backchannel labelling utilities.

These rules mirror the dissertation label-generation protocol and power the
demo API when no trained checkpoint is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CLASSES = [
    "nod",
    "shake",
    "tilt",
    "lean_forward",
    "lean_back",
    "eyebrow_raise",
    "neutral",
]

PRIORITY = [
    "nod",
    "shake",
    "eyebrow_raise",
    "tilt",
    "lean_forward",
    "lean_back",
    "neutral",
]


@dataclass
class CueHit:
    name: str
    score: float
    detail: str


def _bandpass(x: np.ndarray, fps: float, low: float, high: float, order: int = 3) -> np.ndarray:
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


def _peak_count(x: np.ndarray, height_scale: float = 0.5) -> int:
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0
    height = float(np.std(x) * height_scale)
    peaks, _ = find_peaks(x, height=height if height > 0 else None)
    return int(len(peaks))


def synthesise_flame_from_text(text: str, n_frames: int = 64, fps: float = 25.0) -> dict[str, Any]:
    """Build a toy FLAME-like signal from lexical cues so the demo works without video."""
    rng = np.random.default_rng(abs(hash(text.lower())) % (2**32))
    t = np.arange(n_frames) / fps
    lower = text.lower()

    pitch = rng.normal(0, 0.01, size=n_frames)
    yaw = rng.normal(0, 0.01, size=n_frames)
    roll = rng.normal(0, 0.01, size=n_frames)
    trans_z = np.cumsum(rng.normal(0, 0.002, size=n_frames))
    brow = np.clip(rng.normal(0.05, 0.05, size=n_frames), 0, None)

    if any(w in lower for w in ("yes", "yeah", "right", "agree", "mm", "uh-huh", "okay", "ok")):
        pitch = pitch + 0.08 * np.sin(2 * np.pi * 2.0 * t)
    if any(w in lower for w in ("no", "nope", "disagree", "nah")):
        yaw = yaw + 0.07 * np.sin(2 * np.pi * 1.8 * t)
    if any(w in lower for w in ("maybe", "unsure", "confused", "hmm")):
        roll = roll + 0.06 * np.sin(2 * np.pi * 1.0 * t)
    if any(w in lower for w in ("wow", "really", "surprised", "interesting")):
        brow = brow + 0.5
    if any(w in lower for w in ("lean in", "tell me more", "go on")):
        trans_z = trans_z + np.linspace(0, 0.12, n_frames)
    if any(w in lower for w in ("back off", "whoa", "retreat")):
        trans_z = trans_z + np.linspace(0, -0.12, n_frames)

    return {
        "fps": fps,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "trans_z": trans_z,
        "brow": brow,
    }


def detect_cues(flame: dict[str, Any]) -> list[CueHit]:
    fps = float(flame.get("fps", 25.0))
    pitch = np.asarray(flame["pitch"], dtype=float)
    yaw = np.asarray(flame["yaw"], dtype=float)
    roll = np.asarray(flame["roll"], dtype=float)
    trans_z = np.asarray(flame["trans_z"], dtype=float)
    brow = np.asarray(flame["brow"], dtype=float)

    hits: list[CueHit] = []

    pitch_f = _bandpass(pitch, fps, 1.0, 3.0)
    nod_peaks = _peak_count(pitch_f)
    nod_score = min(1.0, nod_peaks / 4.0) * (float(np.max(np.abs(pitch_f))) / 0.08 if np.max(np.abs(pitch_f)) else 0.0)
    if nod_peaks >= 2:
        hits.append(CueHit("nod", float(np.clip(nod_score, 0.15, 1.0)), f"{nod_peaks} peaks in 1–3 Hz pitch band"))

    yaw_f = _bandpass(yaw, fps, 0.8, 2.5)
    shake_peaks = _peak_count(np.abs(yaw_f))
    if shake_peaks >= 2 and float(np.max(np.abs(yaw_f))) > 0.02:
        hits.append(
            CueHit(
                "shake",
                float(np.clip(shake_peaks / 4.0, 0.15, 1.0)),
                f"{shake_peaks} oscillatory yaw peaks",
            )
        )

    brow_mean = float(np.mean(brow))
    if brow_mean > 0.35:
        hits.append(CueHit("eyebrow_raise", float(np.clip(brow_mean, 0.15, 1.0)), f"mean brow proxy={brow_mean:.2f}"))

    roll_mag = float(np.max(np.abs(roll)))
    if roll_mag > 0.05:
        hits.append(CueHit("tilt", float(np.clip(roll_mag / 0.1, 0.15, 1.0)), f"max |roll|={roll_mag:.3f}"))

    dz = float(trans_z[-1] - trans_z[0]) if trans_z.size else 0.0
    if dz > 0.05:
        hits.append(CueHit("lean_forward", float(np.clip(dz / 0.15, 0.15, 1.0)), f"Δz={dz:.3f}"))
    if dz < -0.05:
        hits.append(CueHit("lean_back", float(np.clip((-dz) / 0.15, 0.15, 1.0)), f"Δz={dz:.3f}"))

    return hits


def cues_to_probabilities(hits: list[CueHit], text: str = "") -> dict[str, float]:
    scores = {c: 0.02 for c in CLASSES}
    for hit in hits:
        scores[hit.name] = max(scores[hit.name], hit.score)

    # Light lexical prior when no strong visual cue (demo-friendly).
    lower = text.lower()
    lexical = {
        "nod": ("yes", "yeah", "right", "agree", "mm", "okay", "ok", "uh-huh"),
        "shake": ("no", "nope", "disagree", "nah"),
        "tilt": ("maybe", "unsure", "confused", "hmm"),
        "eyebrow_raise": ("wow", "really", "surprised", "interesting"),
        "lean_forward": ("lean in", "tell me more", "go on"),
        "lean_back": ("back off", "whoa", "retreat"),
    }
    if not hits:
        for cls, words in lexical.items():
            if any(w in lower for w in words):
                scores[cls] = max(scores[cls], 0.55)

    # Softmax-like normalisation with temperature.
    logits = np.array([scores[c] for c in CLASSES], dtype=float)
    # Boost neutral if nothing strong.
    if float(np.max(logits)) < 0.2:
        scores["neutral"] = 0.8
        logits = np.array([scores[c] for c in CLASSES], dtype=float)

    logits = logits / 0.35
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return {c: float(p) for c, p in zip(CLASSES, probs)}


def predict_from_flame(flame: dict[str, Any], text: str = "") -> dict[str, Any]:
    hits = detect_cues(flame)
    probs = cues_to_probabilities(hits, text=text)
    top = max(probs, key=probs.get)
    return {
        "probabilities": probs,
        "prediction": top,
        "confidence": probs[top],
        "cues": [{"name": h.name, "score": h.score, "detail": h.detail} for h in hits],
        "priority": PRIORITY,
        "mode": "heuristic_flame_rules",
    }


def predict_from_text(text: str) -> dict[str, Any]:
    flame = synthesise_flame_from_text(text or "neutral listener")
    result = predict_from_flame(flame, text=text or "")
    result["mode"] = "heuristic_text_to_flame"
    return result
