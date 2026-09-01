#!/usr/bin/env python3
"""Frozen HuBERT DEV-only experiment. GOLD TEST is refused.

Frozen HuBERT representations extracted from approximately sixty second
mixed conversation audio windows, trained using the existing pose-derived
pseudo labelled training set and evaluated exploratorily on fifteen gold
development clips.

Outputs only under ``results/hubert_dev/``. Does not fine-tune HuBERT.
Does not regenerate pseudo-labels. Does not score gold_016–gold_030.

    OMP_NUM_THREADS=1 python scripts/run_hubert_dev.py --smoke
    OMP_NUM_THREADS=1 python scripts/run_hubert_dev.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio_io  # noqa: E402

if shutil.which("ffmpeg"):
    audio_io.ffmpeg_exe = lambda: shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

from src.audio_io import (  # noqa: E402
    DUR_TOL_S,
    TARGET_SR,
    WAV_DIR,
    extract_window_wav,
    inventory_clip,
    load_shard_index,
    load_wav_mono,
    nod_train_dev_ids,
    resample_mono,
    resolve_video_file,
    wav_stats,
)
from src.clip_metrics import always_predict, choose_dev_threshold, clip_binary_metrics  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402
from src.paths import sanitise_artefact  # noqa: E402

OUT_DIR = ROOT / "results" / "hubert_dev"
HUBERT_MODEL = "facebook/hubert-base-ls960"
HUBERT_DIM = 768
CHUNK_SECONDS = 10.0
SEED = 42
PCA_DIM = 16
SMOKE_ID = "gold_001"
EXPECTED_DEV = [f"gold_{i:03d}" for i in range(1, 16)]
TEST_ID_RE = re.compile(r"^gold_(0*(1[6-9]|2[0-9]|30))$")
RGB_CSV = OUT_DIR / "rgb_probabilities.csv"
RGB_SRC = ROOT / "results" / "audio_visual_fusion_dev" / "dev_predictions.csv"
SCRATCH_WAV = Path("/scratch/db01550/hubert_wav")
SCRATCH_TMP = Path("/scratch/db01550/hubert_tmp")
HF_HOME = Path("/scratch/db01550/hf_cache")
TEST_MSG = "REFUSING TO SCORE GOLD TEST FOR HUBERT DEVELOPMENT EXPERIMENT"


def refuse_gold_test_id(sample_id: str) -> None:
    sid = str(sample_id)
    if TEST_ID_RE.match(sid) or sid in {f"gold_{i:03d}" for i in range(16, 31)}:
        print(TEST_MSG)
        raise SystemExit(TEST_MSG)


def refuse_split(split: str | None) -> None:
    if split is None:
        return
    token = str(split).strip().lower().replace("-", "_")
    if token in {"test", "gold_test", "holdout", "hold_out"}:
        print(TEST_MSG)
        raise SystemExit(TEST_MSG)


def metrics_pack(y_true, y_pred) -> dict:
    m = clip_binary_metrics(y_true, y_pred)
    y_pred = np.asarray(y_pred).astype(int)
    m["positive_prediction_rate"] = float(y_pred.mean()) if len(y_pred) else 0.0
    return m


def load_rgb_probabilities() -> tuple[bool, dict[str, float], dict[str, int], str]:
    """Return (available, sid->prob, sid->label, reason). Fusion must not invent probs."""
    path = RGB_CSV if RGB_CSV.exists() else RGB_SRC
    if not path.exists():
        return False, {}, {}, f"missing {RGB_CSV} and {RGB_SRC}"
    rows = list(csv.DictReader(path.open()))
    probs: dict[str, float] = {}
    labels: dict[str, int] = {}
    key = "rgb_probability" if "rgb_probability" in (rows[0] if rows else {}) else "prob_rgb"
    lab_key = "gold_label" if "gold_label" in (rows[0] if rows else {}) else "y_true"
    ids = []
    for rec in rows:
        sid = str(rec["sample_id"])
        refuse_gold_test_id(sid)
        ids.append(sid)
        try:
            p = float(rec[key])
            y = int(rec[lab_key])
        except (KeyError, TypeError, ValueError) as exc:
            return False, {}, {}, f"{path}: cannot parse continuous rgb probability ({exc})"
        if p != p or p in (float("inf"), float("-inf")):
            return False, {}, {}, f"{path}: non-finite rgb_probability for {sid}"
        if p in (0.0, 1.0) and key == "pred":
            return False, {}, {}, f"{path}: appears to be binary predictions"
        probs[sid] = p
        labels[sid] = y
    if ids != EXPECTED_DEV:
        return False, {}, {}, f"{path}: ids {ids} are not exactly gold_001–gold_015"
    if all(p in (0.0, 1.0) for p in probs.values()):
        return False, {}, {}, f"{path}: rgb values are only 0 or 1; not continuous probabilities"
    return True, probs, labels, str(path)


def wav_path_for(sample_id: str, video_id: str) -> Path:
    named = WAV_DIR / f"{sample_id}_{video_id}.wav"
    if named.is_file() and named.stat().st_size > 64:
        return named
    scratch = SCRATCH_WAV / f"{sample_id}_{video_id}.wav"
    if scratch.is_file() and scratch.stat().st_size > 64:
        return scratch
    return scratch


def ensure_wav(sample_id: str, index: dict) -> tuple[Path, dict]:
    refuse_gold_test_id(sample_id)
    meta = inventory_clip(sample_id)
    if str(meta.get("split", "")).upper() == "TEST":
        print(TEST_MSG)
        raise SystemExit(TEST_MSG)
    dest = wav_path_for(sample_id, meta["video_id"])
    if dest.is_file() and dest.stat().st_size > 64:
        return dest, meta
    dest.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
    video_path, _prov, delete_video = resolve_video_file(
        meta["video_id"], index=index, tmp_dir=SCRATCH_TMP, keep=False
    )
    try:
        extract_window_wav(
            video_path,
            dest,
            t0_s=float(meta["t0_s"]),
            duration_s=float(meta["duration_s"]),
        )
    finally:
        if delete_video and Path(video_path).exists():
            try:
                Path(video_path).unlink()
            except OSError:
                pass
    return dest, meta


def load_window_16k(wav: Path, meta: dict) -> tuple[np.ndarray, dict]:
    y, sr = load_wav_mono(wav)
    stats = wav_stats(y, sr)
    y16 = resample_mono(y, sr, TARGET_SR)
    info = {
        "src_sr": int(sr),
        "n_src": int(y.size),
        "duration_src": float(stats["duration_s"]),
        "n_16k": int(y16.size),
        "sr_16k": int(TARGET_SR),
        "duration_16k": float(y16.size / TARGET_SR) if TARGET_SR else 0.0,
        "gold_start": float(meta["t0_s"]),
        "gold_end": float(meta["t1_s"]),
        "window_duration": float(meta["duration_s"]),
        "duration_ok": abs(float(stats["duration_s"]) - float(meta["duration_s"])) <= DUR_TOL_S,
        "wav_path": str(wav),
    }
    return y16.astype(np.float32), info


def load_hubert(device: str):
    try:
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise SystemExit(
            f"HUBERT_ENV_FAIL: transformers/torch import failed: {exc}. "
            "Do not switch encoder."
        ) from exc
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL)
    model = HubertModel.from_pretrained(HUBERT_MODEL)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return extractor, model, torch


def embed_waveform(y16: np.ndarray, extractor, model, torch, device: str) -> tuple[np.ndarray, dict]:
    chunk = int(CHUNK_SECONDS * TARGET_SR)
    if chunk < 1600:
        raise SystemExit("HUBERT_CONFIG_FAIL: CHUNK_SECONDS too small.")
    y16 = np.asarray(y16, dtype=np.float32).reshape(-1)
    if y16.size < 1600:
        raise RuntimeError("audio too short for HuBERT")
    pieces = []
    n_frames = []
    n_chunks = 0
    for start in range(0, y16.size, chunk):
        sl = y16[start : start + chunk]
        if sl.size < 400:
            continue
        n_chunks += 1
        inputs = extractor(
            sl,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            hidden = out.last_hidden_state  # (1, T, 768)
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
                t_frames = int(hidden.shape[1])
            else:
                m = mask.unsqueeze(-1).to(hidden.dtype)
                denom = m.sum(dim=1).clamp(min=1.0)
                pooled = (hidden * m).sum(dim=1) / denom
                t_frames = int(mask.sum().item())
        vec = pooled.squeeze(0).detach().float().cpu().numpy()
        pieces.append(vec)
        n_frames.append(t_frames)
    if not pieces:
        raise RuntimeError("no HuBERT chunks produced")
    weights = np.asarray(n_frames, dtype=np.float64)
    stacked = np.stack(pieces, axis=0)
    emb = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    if emb.shape != (HUBERT_DIM,):
        raise RuntimeError(f"embedding shape {emb.shape} != ({HUBERT_DIM},)")
    info = {
        "n_chunks": int(n_chunks),
        "n_frames_total": int(sum(n_frames)),
        "dim": int(emb.shape[0]),
        "has_nan": bool(np.isnan(emb).any()),
        "has_inf": bool(np.isinf(emb).any()),
        "l2": float(np.linalg.norm(emb)),
    }
    return emb, info


def embedding_path(sample_id: str) -> Path:
    return OUT_DIR / "embeddings" / f"{sample_id}.npz"


def save_embedding(sample_id: str, emb: np.ndarray, extra: dict) -> Path:
    path = embedding_path(sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=emb, sample_id=np.array(sample_id))
    return path


def load_embedding(sample_id: str) -> np.ndarray | None:
    path = embedding_path(sample_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        arr = np.asarray(z["embedding"], dtype=np.float32).reshape(-1)
    return arr


def require_device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"HUBERT_ENV_FAIL: torch missing: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "HUBERT_CUDA_FAIL: torch.cuda.is_available() is False. "
            "GPU is required for the smoke test. Encoder not changed."
        )
    return "cuda"


def run_smoke(index: dict) -> dict:
    t0 = time.time()
    device = require_device()
    wav, meta = ensure_wav(SMOKE_ID, index)
    y16, audio_info = load_window_16k(wav, meta)
    if not audio_info["duration_ok"]:
        payload = {
            "SMOKE_TEST": "FAIL",
            "reason": "audio duration does not match labelled window",
            "audio": audio_info,
        }
        dump_json(OUT_DIR / "smoke_test.json", payload)
        print("SMOKE_TEST = FAIL")
        return payload
    if int(audio_info["sr_16k"]) != TARGET_SR:
        payload = {
            "SMOKE_TEST": "FAIL",
            "reason": f"sample rate {audio_info['sr_16k']} != {TARGET_SR}",
        }
        dump_json(OUT_DIR / "smoke_test.json", payload)
        print("SMOKE_TEST = FAIL")
        return payload
    try:
        import torch

        extractor, model, torch_mod = load_hubert(device)
        mem_alloc = int(torch.cuda.memory_allocated())
        emb, emb_info = embed_waveform(y16, extractor, model, torch_mod, device)
        mem_after = int(torch.cuda.memory_allocated())
    except Exception as exc:  # noqa: BLE001
        payload = {
            "SMOKE_TEST": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "audio": audio_info,
        }
        dump_json(OUT_DIR / "smoke_test.json", payload)
        print("SMOKE_TEST = FAIL")
        print(payload["reason"])
        return payload
    ok = (
        emb.shape == (HUBERT_DIM,)
        and not emb_info["has_nan"]
        and not emb_info["has_inf"]
        and audio_info["duration_ok"]
        and int(audio_info["sr_16k"]) == TARGET_SR
    )
    path = save_embedding(f"smoke_{SMOKE_ID}", emb, emb_info) if ok else None
    payload = {
        "SMOKE_TEST": "PASS" if ok else "FAIL",
        "sample_id": SMOKE_ID,
        "split": meta.get("split"),
        "hubert_model": HUBERT_MODEL,
        "hubert_dim": HUBERT_DIM,
        "chunk_seconds": CHUNK_SECONDS,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_memory_allocated_before_forward": mem_alloc,
        "cuda_memory_allocated_after_forward": mem_after,
        "audio": audio_info,
        "embedding": emb_info,
        "wrote": str(path) if path else None,
        "seconds": time.time() - t0,
        "gold_test_scored": False,
    }
    if not ok:
        payload["reason"] = "embedding checks failed"
    dump_json(OUT_DIR / "smoke_test.json", payload)
    print(f"SMOKE_TEST = {payload['SMOKE_TEST']}")
    return payload


def collect_ids() -> tuple[list[str], list[str]]:
    train_ids, dev_ids = nod_train_dev_ids()
    for sid in list(train_ids) + list(dev_ids):
        refuse_gold_test_id(sid)
    if list(dev_ids) != EXPECTED_DEV:
        raise SystemExit(f"STOP: DEV ids {dev_ids} != {EXPECTED_DEV}")
    return list(train_ids), list(dev_ids)


def extract_split(ids: list[str], split_name: str, index: dict, extractor, model, torch, device: str) -> list[dict]:
    rows = []
    for i, sid in enumerate(ids, 1):
        refuse_gold_test_id(sid)
        cached = load_embedding(sid)
        meta = inventory_clip(sid)
        if str(meta.get("split", "")).upper() == "TEST":
            print(TEST_MSG)
            raise SystemExit(TEST_MSG)
        if cached is not None and cached.shape == (HUBERT_DIM,) and not np.isnan(cached).any():
            rows.append(
                {
                    "sample_id": sid,
                    "split": split_name,
                    "video_id": meta["video_id"],
                    "gold_start": float(meta["t0_s"]),
                    "gold_end": float(meta["t1_s"]),
                    "wav_path": str(wav_path_for(sid, meta["video_id"])),
                    "embedding_path": str(embedding_path(sid)),
                    "dim": HUBERT_DIM,
                    "status": "cached",
                }
            )
            print(f"[{split_name} {i}/{len(ids)}] {sid} cached")
            continue
        wav, meta = ensure_wav(sid, index)
        y16, audio_info = load_window_16k(wav, meta)
        emb, emb_info = embed_waveform(y16, extractor, model, torch, device)
        if emb_info["has_nan"] or emb_info["has_inf"]:
            raise SystemExit(f"HUBERT_EMBED_FAIL: {sid} has NaN/Inf. Encoder not changed.")
        path = save_embedding(sid, emb, emb_info)
        rows.append(
            {
                "sample_id": sid,
                "split": split_name,
                "video_id": meta["video_id"],
                "gold_start": audio_info["gold_start"],
                "gold_end": audio_info["gold_end"],
                "wav_path": audio_info["wav_path"],
                "wav_duration": audio_info["duration_src"],
                "sr_16k": audio_info["sr_16k"],
                "n_chunks": emb_info["n_chunks"],
                "embedding_path": str(path),
                "dim": emb_info["dim"],
                "has_nan": emb_info["has_nan"],
                "has_inf": emb_info["has_inf"],
                "status": "ok",
            }
        )
        print(f"[{split_name} {i}/{len(ids)}] {sid} ok dim={emb_info['dim']}")
    return rows


def stack_embeddings(ids: list[str]) -> np.ndarray:
    mats = []
    for sid in ids:
        refuse_gold_test_id(sid)
        arr = load_embedding(sid)
        if arr is None:
            raise SystemExit(f"HUBERT_EMBED_FAIL: missing embedding for {sid}")
        mats.append(arr)
    x = np.stack(mats, axis=0)
    if np.isnan(x).any() or np.isinf(x).any():
        raise SystemExit("HUBERT_EMBED_FAIL: NaN/Inf in stacked embeddings")
    return x


def labels_for(ids: list[str], kind: str) -> np.ndarray:
    if kind == "train":
        pseudo = {
            str(r["sample_id"]): int(r["pseudo_label"])
            for r in csv.DictReader((ROOT / "results" / "pseudo_labels.csv").open())
        }
        return np.asarray([pseudo[s] for s in ids], dtype=int)
    gold = {
        str(r["sample_id"]): int(r["label"])
        for r in csv.DictReader((ROOT / "data" / "gold_annotations.csv").open())
        if str(r["split"]).upper() == "DEV"
    }
    return np.asarray([gold[s] for s in ids], dtype=int)


def train_and_eval(train_ids: list[str], dev_ids: list[str]) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    set_seed(SEED)
    x_tr = stack_embeddings(train_ids)
    y_tr = labels_for(train_ids, "train")
    x_dv = stack_embeddings(dev_ids)
    y_dv = labels_for(dev_ids, "dev")
    n_pca = int(min(PCA_DIM, x_tr.shape[0] - 1, x_tr.shape[1]))
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=SEED)),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(x_tr, y_tr)
    prob = pipe.predict_proba(x_dv)[:, 1]
    pred_05 = (prob >= 0.5).astype(int)
    m05 = metrics_pack(y_dv, pred_05)
    thr, m_thr_raw = choose_dev_threshold(y_dv, prob, criterion="f1")
    pred_thr = (prob >= thr).astype(int)
    m_thr = metrics_pack(y_dv, pred_thr)
    always = metrics_pack(y_dv, np.ones(len(y_dv), dtype=int))
    pred_csv = OUT_DIR / "hubert_dev_predictions.csv"
    with pred_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "prob_hubert",
                "pred_thr0.5",
                "pred_exploratory_dev_threshold",
            ],
        )
        writer.writeheader()
        for sid, y, p, a, b in zip(dev_ids, y_dv, prob, pred_05, pred_thr):
            writer.writerow(
                {
                    "sample_id": sid,
                    "split": "DEV",
                    "y_true": int(y),
                    "prob_hubert": float(p),
                    "pred_thr0.5": int(a),
                    "pred_exploratory_dev_threshold": int(b),
                }
            )
    metrics = {
        "split": "DEV",
        "gold_test_scored": False,
        "test_n": 0,
        "train_n": int(len(train_ids)),
        "dev_n": int(len(dev_ids)),
        "dev_ids": list(dev_ids),
        "train_ids": list(train_ids),
        "hubert_model": HUBERT_MODEL,
        "hubert_dim": HUBERT_DIM,
        "chunk_seconds": CHUNK_SECONDS,
        "pooling": "mean over HuBERT frames per chunk; length-weighted mean over chunks",
        "sample_rate_hz": TARGET_SR,
        "pca_dim": n_pca,
        "classifier": "StandardScaler + PCA + LogisticRegression(class_weight=balanced, seed=42)",
        "scaler_pca_fitted_on": "TRAIN only",
        "pseudo_labels_file": str(ROOT / "results" / "pseudo_labels.csv"),
        "pseudo_labels_regenerated": False,
        "threshold": 0.5,
        "threshold_policy": "fixed 0.5",
        "metrics": m05,
        "always_positive": always,
        "EXPLORATORY_DEV_SELECTED_THRESHOLD": {
            "threshold": float(thr),
            "metrics": m_thr,
            "note": "stored separately; does not replace threshold 0.5",
        },
        "train_class_counts": {
            "n_pos": int((y_tr == 1).sum()),
            "n_neg": int((y_tr == 0).sum()),
        },
        "dev_class_counts": {
            "n_pos": int((y_dv == 1).sum()),
            "n_neg": int((y_dv == 0).sum()),
        },
    }
    dump_json(OUT_DIR / "hubert_dev_metrics.json", metrics)
    return metrics


def maybe_fusion(dev_ids: list[str]) -> None:
    available, rgb_probs, rgb_labels, src = load_rgb_probabilities()
    dump_json(
        OUT_DIR / "rgb_probability_check.json",
        {
            "RGB_PROBABILITIES_AVAILABLE": available,
            "source": src,
            "ids": list(rgb_probs.keys()),
            "n": len(rgb_probs),
        },
    )
    if not available:
        print("RGB_PROBABILITIES_AVAILABLE = NO")
        print(src)
        (OUT_DIR / "FUSION_SKIPPED.txt").write_text(
            "RGB_PROBABILITIES_AVAILABLE = NO\n"
            f"reason: {src}\n"
            "Fusion stage stopped. No alternative fusion method was used.\n"
        )
        return
    print("RGB_PROBABILITIES_AVAILABLE = YES")
    hubert_rows = list(csv.DictReader((OUT_DIR / "hubert_dev_predictions.csv").open()))
    hubert_ids = [r["sample_id"] for r in hubert_rows]
    if hubert_ids != EXPECTED_DEV or list(rgb_probs.keys()) != EXPECTED_DEV:
        print("RGB_PROBABILITIES_AVAILABLE = NO")
        reason = f"id mismatch hubert={hubert_ids} rgb={list(rgb_probs.keys())}"
        print(reason)
        (OUT_DIR / "FUSION_SKIPPED.txt").write_text(
            "RGB_PROBABILITIES_AVAILABLE = NO\n" + reason + "\n"
        )
        return
    y = []
    p_rgb = []
    p_h = []
    with (OUT_DIR / "hubert_rgb_fusion_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "rgb_probability",
                "hubert_probability",
                "prob_fusion_50_50",
                "pred_fusion_thr0.5",
            ],
        )
        writer.writeheader()
        for rec in hubert_rows:
            sid = rec["sample_id"]
            refuse_gold_test_id(sid)
            yt = int(rec["y_true"])
            if yt != rgb_labels[sid]:
                raise SystemExit(f"STOP: label mismatch {sid} hubert={yt} rgb={rgb_labels[sid]}")
            pr = float(rgb_probs[sid])
            ph = float(rec["prob_hubert"])
            pf = 0.5 * pr + 0.5 * ph
            pred = int(pf >= 0.5)
            y.append(yt)
            p_rgb.append(pr)
            p_h.append(ph)
            writer.writerow(
                {
                    "sample_id": sid,
                    "split": "DEV",
                    "y_true": yt,
                    "rgb_probability": pr,
                    "hubert_probability": ph,
                    "prob_fusion_50_50": pf,
                    "pred_fusion_thr0.5": pred,
                }
            )
    y_a = np.asarray(y)
    pred_f = (0.5 * np.asarray(p_rgb) + 0.5 * np.asarray(p_h) >= 0.5).astype(int)
    pred_r = (np.asarray(p_rgb) >= 0.5).astype(int)
    pred_h = (np.asarray(p_h) >= 0.5).astype(int)
    fusion_metrics = {
        "split": "DEV",
        "gold_test_scored": False,
        "test_n": 0,
        "formula": "p_fusion = 0.5 * p_rgb + 0.5 * p_hubert",
        "threshold": 0.5,
        "threshold_policy": "fixed 0.5 (parameter-free; not searched on DEV)",
        "weights_searched": False,
        "dev_ids": list(dev_ids),
        "rgb_source": src,
        "metrics": metrics_pack(y_a, pred_f),
        "rgb_thr0.5": metrics_pack(y_a, pred_r),
        "hubert_thr0.5": metrics_pack(y_a, pred_h),
        "always_positive": metrics_pack(y_a, np.ones(len(y_a), dtype=int)),
    }
    dump_json(OUT_DIR / "hubert_rgb_fusion_metrics.json", fusion_metrics)
    rows = [
        ("Always positive", "none (always 1)", fusion_metrics["always_positive"]),
        ("HuBERT only", "fixed 0.5", fusion_metrics["hubert_thr0.5"]),
        ("RGB only", "fixed 0.5", fusion_metrics["rgb_thr0.5"]),
        ("50/50 HuBERT+RGB probability fusion", "fixed 0.5", fusion_metrics["metrics"]),
    ]
    with (OUT_DIR / "multimodal_dev_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "split",
                "threshold_policy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "tp",
                "fp",
                "tn",
                "fn",
                "positive_prediction_rate",
            ],
        )
        writer.writeheader()
        for name, pol, m in rows:
            writer.writerow(
                {
                    "model": name,
                    "split": "DEV",
                    "threshold_policy": pol,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "tn": m["tn"],
                    "fn": m["fn"],
                    "positive_prediction_rate": m["positive_prediction_rate"],
                }
            )


def write_run_manifest(extra: dict) -> None:
    payload = {
        "experiment": (
            "Frozen HuBERT representations extracted from approximately "
            "sixty second mixed conversation audio windows, trained using "
            "the existing pose derived pseudo labelled training set and "
            "evaluated exploratorily on fifteen gold development clips."
        ),
        "audio_source": "mixed conversation audio (RealTalk container soundtrack)",
        "hubert_model": HUBERT_MODEL,
        "hubert_dim": HUBERT_DIM,
        "chunk_seconds": CHUNK_SECONDS,
        "sample_rate_hz": TARGET_SR,
        "fine_tuned_hubert": False,
        "pseudo_labels_regenerated": False,
        "gold_test_scored": False,
        "test_n": 0,
        "out_dir": str(OUT_DIR),
        "locked_results_modified": False,
        **extra,
    }
    dump_json(OUT_DIR / "run_manifest.json", payload)


def write_embedding_manifest(rows: list[dict]) -> None:
    path = OUT_DIR / "embedding_manifest.csv"
    keys = []
    for rec in rows:
        for k in rec:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sanitise_artefact(rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    refuse_split(args.split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    index = load_shard_index()
    if args.smoke:
        smoke = run_smoke(index)
        write_run_manifest(
            {
                "stage": "smoke",
                "SMOKE_TEST": smoke.get("SMOKE_TEST"),
                "train_n": None,
                "dev_n": None,
            }
        )
        if smoke.get("SMOKE_TEST") != "PASS":
            raise SystemExit("SMOKE_TEST = FAIL")
        return

    smoke_path = OUT_DIR / "smoke_test.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text()).get("SMOKE_TEST") != "PASS":
        raise SystemExit(
            "STOP: run scripts/run_hubert_dev.py --smoke first. "
            "Batch extraction is refused until SMOKE_TEST = PASS."
        )

    t0 = time.time()
    train_ids, dev_ids = collect_ids()
    device = require_device()
    extractor, model, torch = load_hubert(device)
    train_rows = extract_split(train_ids, "TRAIN", index, extractor, model, torch, device)
    dev_rows = extract_split(dev_ids, "DEV", index, extractor, model, torch, device)
    write_embedding_manifest(train_rows + dev_rows)
    metrics = train_and_eval(train_ids, dev_ids)
    maybe_fusion(dev_ids)
    write_run_manifest(
        {
            "stage": "full",
            "SMOKE_TEST": "PASS",
            "train_n": len(train_ids),
            "dev_n": len(dev_ids),
            "test_n": 0,
            "seconds": time.time() - t0,
            "device": device,
        }
    )
    print("TRAIN_N =", len(train_ids))
    print("DEV_N =", len(dev_ids))
    print("TEST_N = 0")
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
