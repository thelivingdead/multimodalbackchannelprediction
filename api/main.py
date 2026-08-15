"""BackchannelAI demo API — heuristic 7-class predictor with checkpoint hook."""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from label_rules import CLASSES, predict_from_flame, predict_from_text, synthesise_flame_from_text

app = FastAPI(
    title="BackchannelAI API",
    description="Demo API for multimodal fine-grained backchannel prediction (7-class).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional path for a future PyTorch checkpoint (swap-in point for real model).
MODEL_CHECKPOINT: Optional[str] = None


class PredictRequest(BaseModel):
    text: str = Field("", description="ASR transcript or listener context text")
    modalities: list[str] = Field(default_factory=lambda: ["text"], description="Subset of text|audio|video")
    use_checkpoint: bool = Field(False, description="If true and checkpoint loaded, run neural model")


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    cues: list[dict[str, Any]]
    mode: str
    classes: list[str]
    modalities_used: list[str]
    explanation: str


def _explanation(result: dict[str, Any], modalities: list[str]) -> str:
    cues = result.get("cues") or []
    if cues:
        parts = [f"{c['name']} ({c['detail']})" for c in cues]
        return (
            f"Top class `{result['prediction']}` from heuristic FLAME rules. "
            f"Fired cues: " + "; ".join(parts) + f". Modalities requested: {', '.join(modalities)}."
        )
    return (
        f"Top class `{result['prediction']}` (confidence {result['confidence']:.2f}). "
        "No strong pose cue fired; lexical priors and neutral bias applied. "
        f"Modalities requested: {', '.join(modalities)}."
    )


def _run_checkpoint_stub(text: str, modalities: list[str]) -> Optional[dict[str, Any]]:
    """Placeholder for loading a real multimodal checkpoint later."""
    if not MODEL_CHECKPOINT or not MODEL_CHECKPOINT.endswith((".pt", ".pth", ".ckpt")):
        return None
    # Intentionally not loading heavy deps in the demo path.
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "classes": CLASSES,
        "checkpoint_loaded": bool(MODEL_CHECKPOINT),
    }


@app.get("/classes")
def classes() -> dict[str, Any]:
    return {
        "classes": CLASSES,
        "descriptions": {
            "nod": "Vertical head oscillation (~1–3 Hz) signalling affirmation/attention",
            "shake": "Horizontal yaw oscillation signalling negation/disagreement",
            "tilt": "Roll / lateral head tilt signalling uncertainty or curiosity",
            "lean_forward": "Forward posture translation — engagement",
            "lean_back": "Backward posture translation — distancing",
            "eyebrow_raise": "Brow raise — surprise / emphasis / social accent",
            "neutral": "No backchannel event detected in the window",
        },
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    modalities = [m.lower() for m in (body.modalities or ["text"])]
    if body.use_checkpoint:
        ckpt = _run_checkpoint_stub(body.text, modalities)
        if ckpt is not None:
            result = ckpt
        else:
            result = predict_from_text(body.text)
            result["mode"] = "heuristic_fallback_no_checkpoint"
    else:
        result = predict_from_text(body.text)

    return PredictResponse(
        prediction=result["prediction"],
        confidence=float(result["confidence"]),
        probabilities=result["probabilities"],
        cues=result.get("cues", []),
        mode=result.get("mode", "heuristic"),
        classes=CLASSES,
        modalities_used=modalities,
        explanation=_explanation(result, modalities),
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    text: str = Form(""),
    modalities: str = Form("text,audio,video"),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
) -> PredictResponse:
    """Accept optional audio/video uploads; demo uses size/energy heuristics + text rules."""
    mod_list = [m.strip().lower() for m in modalities.split(",") if m.strip()]
    flame = synthesise_flame_from_text(text or "listener window")

    # If audio uploaded, nudge pitch variance from byte energy (demo signal only).
    if audio is not None and "audio" in mod_list:
        raw = await audio.read()
        if raw:
            arr = np.frombuffer(raw[: min(len(raw), 50_000)], dtype=np.uint8).astype(float)
            energy = float(np.std(arr) / 64.0)
            t = np.arange(len(flame["pitch"])) / flame["fps"]
            flame["pitch"] = flame["pitch"] + energy * 0.05 * np.sin(2 * np.pi * 2.0 * t)

    if video is not None and "video" in mod_list:
        raw = await video.read()
        # Without a vision stack, use payload length as a weak proxy for motion presence.
        if raw and len(raw) > 10_000:
            flame["brow"] = flame["brow"] + 0.15

    result = predict_from_flame(flame, text=text)
    result["mode"] = "heuristic_upload_fused"
    return PredictResponse(
        prediction=result["prediction"],
        confidence=float(result["confidence"]),
        probabilities=result["probabilities"],
        cues=result.get("cues", []),
        mode=result["mode"],
        classes=CLASSES,
        modalities_used=mod_list,
        explanation=_explanation(result, mod_list),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "BackchannelAI",
        "docs": "/docs",
        "predict": "POST /predict",
        "upload": "POST /predict/upload",
    }
