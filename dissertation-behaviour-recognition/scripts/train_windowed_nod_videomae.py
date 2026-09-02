#!/usr/bin/env python3
"""Fine-tune VideoMAE on 3 s nod windows. New folder only.

Needs features/rgb16_windowed/<window_id>.npz from
fetch_rgb_windows_nod3s.py. Trains on DEV windows, selects on DEV,
scores TEST once. Does not write results/videomae_finetuned/.

Otter95 (after the fetch has finished)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_windowed_nod_videomae.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import choose_dev_threshold, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
WINDOWS_TEST = ROOT / "data" / "windowed_annotations" / "nod_windows_test.csv"
RGB_DIR = ROOT / "features" / "rgb16_windowed"
DEFAULT_OUT = ROOT / "results" / "windowed_nod" / "videomae_finetuned"
LOCKED = ROOT / "results" / "videomae_finetuned"
CHECKPOINT = "MCG-NJU/videomae-base"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
MAX_MISSING_FRAC = 0.10


def _load_windows(path: Path, split: str, allowed: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.upper()
    if (df["split"] != split).any():
        raise SystemExit(f"STOP: {path.name} has a non-{split} row")
    ids = set(df["sample_id"])
    if ids - allowed:
        raise SystemExit(f"STOP: {path.name} has ids outside {split}")
    return df.reset_index(drop=True)


def load_rgb(window_id: str) -> np.ndarray | None:
    path = RGB_DIR / f"{window_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        rgb = z["rgb"]
    if rgb.shape != (16, 224, 224, 3) or rgb.dtype != np.uint8:
        raise SystemExit(f"STOP: {path.name} has rgb {rgb.shape} {rgb.dtype}")
    return rgb


def pack_split(df: pd.DataFrame, name: str):
    clips, ys, keep_idx, missing = [], [], [], []
    for i, r in enumerate(df.itertuples(index=False)):
        rgb = load_rgb(str(r.window_id))
        if rgb is None:
            missing.append(str(r.window_id))
            continue
        clips.append(rgb)
        ys.append(int(r.label))
        keep_idx.append(i)
    if missing:
        print(f"NOTE: {name}: {len(missing)} windows missing rgb")
    if len(df) and len(missing) / len(df) > MAX_MISSING_FRAC:
        raise SystemExit(
            f"STOP: {name} missing {len(missing)}/{len(df)} rgb crops. "
            "Finish scripts/fetch_rgb_windows_nod3s.py first."
        )
    kept = df.iloc[keep_idx].reset_index(drop=True)
    return clips, np.asarray(ys, dtype=np.int64), kept


def clip_any(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_c, p_c = [], []
    for _, g in df.groupby("sample_id", sort=True):
        y_c.append(int(g["label"].max()))
        p_c.append(int(pred[g.index.to_numpy()].max()))
    return np.asarray(y_c, dtype=int), np.asarray(p_c, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir == LOCKED.resolve() or LOCKED.resolve() in out_dir.parents:
        raise SystemExit("STOP: will not write locked results/videomae_finetuned/")
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_split_leakage import assert_unlocked_out_dir

    assert_unlocked_out_dir(out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"STOP: {metrics_path} exists. TEST already scored.")

    dev = _load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    tes = _load_windows(WINDOWS_TEST, "TEST", TEST_IDS)
    if set(dev["sample_id"]) & set(tes["sample_id"]):
        raise SystemExit("STOP: DEV/TEST sample overlap")
    dv_clips, ydv, dev = pack_split(dev, "DEV")
    te_clips, yte, tes = pack_split(tes, "TEST")

    try:
        import torch
        import transformers
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            VideoMAEConfig,
            VideoMAEForVideoClassification,
            VideoMAEImageProcessor,
        )
        try:
            from transformers import VideoMAEImageProcessorPil
        except ImportError:
            VideoMAEImageProcessorPil = None
    except ImportError as exc:
        raise SystemExit(
            "STOP: torch/transformers missing. On otter use "
            "/scratch/db01550/venv/bin/python"
        ) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    try:
        processor = (
            VideoMAEImageProcessorPil.from_pretrained(CHECKPOINT)
            if VideoMAEImageProcessorPil is not None
            else VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
        )
    except Exception:
        processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    mean_t = torch.tensor(processor.image_mean, dtype=torch.float32)
    std_t = torch.tensor(processor.image_std, dtype=torch.float32)

    def preprocess(rgb_u8: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).to(torch.float32) / 255.0
        x = (x - mean_t) / std_t
        return x.permute(0, 3, 1, 2).contiguous()

    class WinDS(Dataset):
        def __init__(self, clips, labels, flip: bool):
            self.clips, self.labels, self.flip = clips, labels, flip

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            x = preprocess(self.clips[i])
            if self.flip and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            return x, float(self.labels[i])

    def predict_probs(model, clips) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(clips), args.batch_size):
                xb = torch.stack([preprocess(c) for c in clips[i : i + args.batch_size]]).to(device)
                logits = model(pixel_values=xb).logits.squeeze(-1)
                out.append(1 / (1 + np.exp(-logits.float().cpu().numpy())))
        return np.concatenate(out) if out else np.asarray([])

    config = VideoMAEConfig.from_pretrained(CHECKPOINT)
    config.num_labels = 1
    model = VideoMAEForVideoClassification.from_pretrained(
        CHECKPOINT, config=config, ignore_mismatched_sizes=True
    )
    model.to(device)
    layers = model.videomae.encoder.layer
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.classifier] + ([model.fc_norm] if model.fc_norm is not None else []):
        for p in module.parameters():
            p.requires_grad_(True)
    for layer in layers[-4:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [
            {"params": [p for layer in layers[-4:] for p in layer.parameters()], "lr": 1e-5},
            {
                "params": [p for m in [model.classifier] + ([model.fc_norm] if model.fc_norm is not None else []) for p in m.parameters()],
                "lr": 1e-4,
            },
        ]
    )
    pos = max(int((ydv == 1).sum()), 1)
    neg = max(int((ydv == 0).sum()), 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32).to(device))
    loader = DataLoader(
        WinDS(dv_clips, ydv, flip=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(
        f"DEV {len(ydv)} windows ({int((ydv==1).sum())} pos); "
        f"TEST {len(yte)}; transformers {transformers.__version__}"
    )

    hist, best, best_state, bad = [], None, None, 0
    ckpt = out_dir / "best_model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    loss = crit(model(pixel_values=xb).logits.squeeze(-1), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = crit(model(pixel_values=xb).logits.squeeze(-1), yb)
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        prob = predict_probs(model, dv_clips)
        thr, mm = choose_dev_threshold(ydv, prob)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1": float(mm["f1"]),
            "dev_probability_threshold": float(thr),
        }
        hist.append(row)
        print(f"epoch {epoch} loss={row['train_loss']:.4f} DEV window F1={row['dev_f1']:.3f}")
        if best is None or row["dev_f1"] > best["dev_f1"]:
            best = dict(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    thr = float(best["dev_probability_threshold"])
    pdv = predict_probs(model, dv_clips)
    pte = predict_probs(model, te_clips)
    pred_te = (pte >= thr).astype(int)
    test_w = clip_binary_metrics(yte, pred_te)
    y_clip, p_clip = clip_any(tes, pred_te)
    test_c = clip_binary_metrics(y_clip, p_clip)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "window_id": tes["window_id"].astype(str).tolist(),
            "sample_id": tes["sample_id"].astype(str).tolist(),
            "label": yte,
            "prob": pte,
            "pred": pred_te,
        }
    ).to_csv(out_dir / "predictions_test.csv", index=False)
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_nod_3s_videomae",
            "selection": "DEV window F1; trained on DEV windows; TEST once",
            "best_epoch": int(best["epoch"]),
            "dev_window_f1": float(best["dev_f1"]),
            "dev_probability_threshold": thr,
            "test_window": test_w,
            "test_clip_any_window": test_c,
            "n_dev": int(len(ydv)),
            "n_test": int(len(yte)),
        },
    )
    print("=====================================")
    print("windowed nod VideoMAE (3 s)")
    print(f"  best epoch (DEV): {best['epoch']}   DEV window F1: {best['dev_f1']:.3f}")
    print(
        f"  TEST window P {test_w['precision']:.3f}  R {test_w['recall']:.3f}  "
        f"F1 {test_w['f1']:.3f}"
    )
    print(f"  artifacts: {out_dir}")
    print("  locked 60 s VideoMAE folders were not written.")


if __name__ == "__main__":
    main()
