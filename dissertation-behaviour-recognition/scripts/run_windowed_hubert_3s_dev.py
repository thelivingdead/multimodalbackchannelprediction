#!/usr/bin/env python3
"""DEV 3 s frozen HuBERT nod experiment. TEST is never read.

Same windows, labels, and leave-one-clip-out folds as the MFCC run.
HuBERT-base is frozen. Each 3 s mix is mean-pooled to 768-D. Only a
weighted logistic regression is trained. No fine-tune.

    /scratch/db01550/venv/bin/python scripts/run_windowed_hubert_3s_dev.py \\
        --extract --train --figures

Otter::

    bash scripts/run_windowed_hubert_3s_otter.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from run_windowed_audio_3s_dev import (  # noqa: E402
    existing_clip_wav,
    load_dev_windows,
    train_loco,
    write_audit,
)
from src.audio_io import (  # noqa: E402
    TARGET_SR,
    load_wav_mono,
    refuse_test_scoring,
    resample_mono,
)
from src.utils import dump_json, set_seed  # noqa: E402

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
OUT = ROOT / "results" / "windowed_dev" / "audio_3s_hubert"
MFCC_METRICS = ROOT / "results" / "windowed_dev" / "audio_3s" / "metrics.json"
_WAV_ENV = os.environ.get("AUDIO_WINDOWED_CACHE", "").strip()
WAV_CACHE = (
    Path(_WAV_ENV).expanduser()
    if _WAV_ENV
    else ROOT / "data" / "features" / "audio_windowed_dev"
)
_EMB_ENV = os.environ.get("HUBERT_WINDOWED_CACHE", "").strip()
EMB_CACHE = (
    Path(_EMB_ENV).expanduser()
    if _EMB_ENV
    else Path("/scratch/db01550/audio_windowed_hubert")
)
HUBERT_MODEL = "facebook/hubert-base-ls960"
HUBERT_DIM = 768
SEED = 42
FIXED_THRESHOLD = 0.5
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}


def _device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"STOP: torch missing ({exc}). Use /scratch/db01550/venv/bin/python"
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    print("NOTE: CUDA not available; frozen HuBERT will run on CPU.")
    return "cpu"


def load_hubert(device: str):
    try:
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise SystemExit(
            f"STOP: transformers/torch import failed: {exc}. Do not switch encoder."
        ) from exc
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL)
    model = HubertModel.from_pretrained(HUBERT_MODEL)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return extractor, model, torch


def embed_window(y16: np.ndarray, extractor, model, torch, device: str) -> np.ndarray:
    y16 = np.asarray(y16, dtype=np.float32).reshape(-1)
    if y16.size < TARGET_SR // 4:
        raise RuntimeError(f"short window {y16.size} samples")
    inputs = extractor(
        y16,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
        mask = inputs.get("attention_mask")
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    vec = pooled.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if vec.shape != (HUBERT_DIM,):
        raise RuntimeError(f"embedding {vec.shape} != ({HUBERT_DIM},)")
    if not np.isfinite(vec).all():
        raise RuntimeError("non-finite HuBERT embedding")
    return vec


def clip_wav_path(sample_id: str) -> Path | None:
    named = WAV_CACHE / f"{sample_id}_clip.wav"
    if named.is_file() and named.stat().st_size > 64:
        return named
    return existing_clip_wav(sample_id, "")


def extract_embeddings(frame: pd.DataFrame) -> list[str]:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    statuses = ["not_extracted"] * len(frame)
    device = _device()
    extractor, model, torch = load_hubert(device)
    print(
        f"frozen {HUBERT_MODEL} on {device}; mean-pool to {HUBERT_DIM}-D; no fine-tune",
        flush=True,
    )
    clip_audio: dict[str, tuple[np.ndarray, int]] = {}
    for i, rec in enumerate(frame.itertuples(index=False)):
        sid = str(rec.sample_id)
        if sid in TEST_IDS:
            raise SystemExit(f"STOP: TEST id {sid}")
        dest = EMB_CACHE / f"{rec.window_id}.npz"
        if dest.exists():
            statuses[i] = "ok"
            continue
        try:
            if sid not in clip_audio:
                wav = clip_wav_path(sid)
                if wav is None:
                    raise RuntimeError(
                        f"missing clip wav for {sid}. "
                        "Run bash scripts/run_windowed_audio_3s_otter.sh first."
                    )
                y, sr = load_wav_mono(wav)
                clip_audio[sid] = (resample_mono(y, sr, TARGET_SR), TARGET_SR)
            y16, sr = clip_audio[sid]
            s0 = int(round(float(rec.start_sec) * sr))
            s1 = int(round(float(rec.end_sec) * sr))
            chunk = y16[s0:s1]
            emb = embed_window(chunk, extractor, model, torch, device)
            np.savez(
                dest,
                embedding=emb,
                window_id=str(rec.window_id),
                sample_id=sid,
                model=np.asarray(HUBERT_MODEL),
            )
            statuses[i] = "ok"
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: {rec.window_id} HuBERT failed ({exc})", flush=True)
            statuses[i] = "failed"
        if (i + 1) % 29 == 0:
            print(f"embed {sid} done", flush=True)
    return statuses


def load_matrix(frame: pd.DataFrame, statuses: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    keep = []
    for rec, status in zip(frame.itertuples(index=False), statuses):
        if status != "ok":
            keep.append(False)
            rows.append(np.full(HUBERT_DIM, np.nan, dtype=np.float32))
            continue
        path = EMB_CACHE / f"{rec.window_id}.npz"
        with np.load(path, allow_pickle=True) as z:
            feat = np.asarray(z["embedding"], dtype=np.float32).reshape(-1)
        if feat.size != HUBERT_DIM:
            raise SystemExit(f"STOP: {rec.window_id} is {feat.size}-D, expected {HUBERT_DIM}")
        rows.append(feat)
        keep.append(True)
    return np.stack(rows), np.asarray(keep, dtype=bool)


def write_tables(pack: dict, out_dir: Path) -> None:
    m = pack["metrics"]
    b = pack["boot"]["balanced_accuracy"]
    row = {
        "model": "audio HuBERT frozen LR",
        "ba": m["balanced_accuracy"],
        "ci_lo": b["ci_lower_95"],
        "ci_hi": b["ci_upper_95"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "pr_auc": pack["pr_auc"],
        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"],
    }
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(tables / "table2_dev_hubert.csv", index=False)
    (tables / "table2_dev_hubert.tex").write_text(
        "model & BA & 95\\% CI & F1 & P & R & PR-AUC & TP & FP & TN & FN \\\\\n"
        f"audio HuBERT frozen LR & {row['ba']:.3f} & "
        f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] & "
        f"{row['f1']:.3f} & {row['precision']:.3f} & {row['recall']:.3f} & "
        f"{row['pr_auc']:.3f} & {row['tp']} & {row['fp']} & {row['tn']} & {row['fn']} \\\\\n"
    )


def compare_mfcc(hubert_ba: float, hubert_lo: float, hubert_hi: float) -> dict:
    mfcc = {
        "balanced_accuracy": 0.5339927696324563,
        "ci_lower_95": 0.4702353861629596,
        "ci_upper_95": 0.6012570264677177,
        "source": "locked MFCC DEV headline",
    }
    if MFCC_METRICS.exists():
        payload = json.loads(MFCC_METRICS.read_text())
        ba = payload["at_fixed_threshold_0.5"]["balanced_accuracy"]
        boot = payload["clip_bootstrap_at_0.5"]["balanced_accuracy"]
        mfcc = {
            "balanced_accuracy": float(ba),
            "ci_lower_95": float(boot["ci_lower_95"]),
            "ci_upper_95": float(boot["ci_upper_95"]),
            "source": str(MFCC_METRICS.relative_to(ROOT)),
        }
    delta = float(hubert_ba - mfcc["balanced_accuracy"])
    improves = bool(hubert_lo > mfcc["ci_upper_95"] or (delta >= 0.03 and hubert_lo > 0.5))
    print(
        f"MFCC   DEV BA {mfcc['balanced_accuracy']:.3f} "
        f"[{mfcc['ci_lower_95']:.3f}, {mfcc['ci_upper_95']:.3f}]"
    )
    print(f"HuBERT DEV BA {hubert_ba:.3f} [{hubert_lo:.3f}, {hubert_hi:.3f}]")
    print(f"delta HuBERT-MFCC {delta:+.3f}")
    if improves:
        print("Frozen HuBERT improves on MFCC on DEV. Still do not score TEST.")
    else:
        print("Frozen HuBERT does not clearly beat MFCC on DEV. Do not fine-tune yet. Do not score TEST.")
    return {
        "mfcc": mfcc,
        "hubert_minus_mfcc": delta,
        "clearly_improves_on_mfcc": improves,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    refuse_test_scoring(score_test=False, split="dev")
    if not (args.extract or args.train or args.figures):
        args.extract = True
        args.train = True
        args.figures = True
    if "test" in WINDOWS.name.lower():
        raise SystemExit("STOP: refused TEST window file")
    out = assert_unlocked_out_dir(OUT)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    frame = load_dev_windows()
    print("WINDOWED HUBERT 3s  DEV only  TEST not read")
    print(f"encoder: frozen {HUBERT_MODEL}; mean-pool {HUBERT_DIM}-D; LR only")
    print("wavs reused from the MFCC 3 s cache. HuBERT weights are not updated.")
    statuses = ["not_extracted"] * len(frame)
    if args.extract:
        statuses = extract_embeddings(frame)
    else:
        statuses = [
            "ok" if (EMB_CACHE / f"{wid}.npz").exists() else "not_extracted"
            for wid in frame["window_id"].astype(str)
        ]
    write_audit(frame, statuses, out)
    if args.train:
        x, keep = load_matrix(frame, statuses)
        pack = train_loco(frame, x, keep)
        pack["pred_df"].to_csv(out / "predictions.csv", index=False)
        write_tables(pack, out)
        folds = []
        for rec in pack["folds"]:
            slim = {k: v for k, v in rec.items() if k not in {"train_mean", "train_std"}}
            folds.append(slim)
        m = pack["metrics"]
        b = pack["boot"]["balanced_accuracy"]
        cmp_ = compare_mfcc(m["balanced_accuracy"], b["ci_lower_95"], b["ci_upper_95"])
        dump_json(
            out / "metrics.json",
            {
                "protocol": "windowed_nod_3s_audio_hubert_frozen_loco",
                "development_only": True,
                "test_scored": False,
                "test_read": False,
                "fine_tuned": False,
                "seed": args.seed,
                "threshold": FIXED_THRESHOLD,
                "threshold_selection": "fixed 0.5; not chosen on held-out clips",
                "encoder": HUBERT_MODEL,
                "encoder_trainable": False,
                "pooling": "mean over HuBERT time steps in the 3 s window",
                "feature_dim": HUBERT_DIM,
                "sample_rate_hz": TARGET_SR,
                "normalisation": "train-fold z-score over windows",
                "model": "weighted logistic regression, L2=1e-2, no sklearn required",
                "input_shape": f"(n_windows, {HUBERT_DIM})",
                "representation": (
                    f"frozen {HUBERT_MODEL} last_hidden_state, mean-pooled, "
                    "container soundtrack, 16 kHz mono"
                ),
                "n_windows_scored": pack["n_scored"],
                "at_fixed_threshold_0.5": pack["metrics"],
                "pr_auc_out_of_fold": pack["pr_auc"],
                "clip_bootstrap_at_0.5": pack["boot"],
                "always_no": pack["always_no"],
                "always_yes": pack["always_yes"],
                "folds": folds,
                "chance_balanced_accuracy": 0.5,
                "mfcc_comparison": cmp_,
            },
        )
        dump_json(
            out / "config.json",
            {
                "seed": args.seed,
                "encoder": HUBERT_MODEL,
                "frozen": True,
                "feature_dim": HUBERT_DIM,
                "classifier": "logreg_newton",
                "class_weight": "n_neg/n_pos on train clips",
                "threshold": FIXED_THRESHOLD,
                "windows": str(WINDOWS.relative_to(ROOT)),
                "test_read": False,
                "fine_tune": False,
            },
        )
        print(
            f"HUBERT-FROZEN DEV BA {m['balanced_accuracy']:.3f} "
            f"[{b['ci_lower_95']:.3f}, {b['ci_upper_95']:.3f}]  "
            f"F1 {m['f1']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  "
            f"PR {pack['pr_auc']:.3f}  "
            f"TP{m['tp']} FP{m['fp']} TN{m['tn']} FN{m['fn']}"
        )
        if b["ci_lower_95"] <= 0.5:
            print("Frozen HuBERT is at chance on DEV. Do not fuse. Do not score TEST.")
        print("TEST not loaded.")
    if args.figures:
        from plot_windowed_hubert_3s_dev import write_figures

        write_figures(out)
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
