#!/usr/bin/env python3
"""3 s windowed nod pose CNN. New experiment. Locked 60 s results are not written.

Uses gold EMOCA tracks already in features/gold/*.npz. Each row of
nod_windows_dev.csv / nod_windows_test.csv is one 75-frame (3 s) slice.
Feature set C = rotation xyz + first differences. Threshold and epoch
are chosen on DEV windows only. TEST is scored once.

This is not the old 60 s clip trainer. Do not run train_pose_cnn.py for
this protocol. Do not write results/videomae_finetuned/ or
results/classifier_test_metrics.json.

Otter95 (copy the new window CSVs and this script first)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_windowed_nod_pose_cnn.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.clip_metrics import choose_dev_threshold, clip_binary_metrics  # noqa: E402
from src.pose_cnn import _build_cnn, load_npz, resample_seq  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402

WINDOW_FRAMES = 75
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
DEFAULT_OUT = ROOT / "results" / "windowed_nod" / "pose_cnn"
WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
WINDOWS_TEST = ROOT / "data" / "windowed_annotations" / "nod_windows_test.csv"
GOLD_DIR = ROOT / "features" / "gold"


def _load_windows(path: Path, split: str, allowed: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    df = pd.read_csv(path)
    need = {
        "window_id",
        "sample_id",
        "split",
        "start_frame_relative",
        "end_frame_relative",
        "label",
    }
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"STOP: {path.name} missing {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.upper()
    if (df["split"] != split).any():
        raise SystemExit(f"STOP: {path.name} has a non-{split} row")
    ids = set(df["sample_id"])
    if ids - allowed:
        raise SystemExit(f"STOP: {path.name} has ids outside {split}: {sorted(ids - allowed)}")
    if allowed - ids:
        raise SystemExit(f"STOP: {path.name} missing {sorted(allowed - ids)}")
    return df.reset_index(drop=True)


def slice_rotation(z: dict, start_frame_relative: int, end_frame_relative: int) -> np.ndarray:
    i0 = int(start_frame_relative)
    i1 = int(end_frame_relative)
    rot = np.asarray(z["rotation_xyz"], dtype=np.float32)
    if i0 < 0 or i1 > len(rot) or i1 <= i0:
        raise SystemExit(f"STOP: bad slice {i0}:{i1} on {z.get('sample_id')}")
    chunk = rot[i0:i1]
    if len(chunk) != WINDOW_FRAMES:
        chunk = resample_seq(chunk, t=WINDOW_FRAMES)
    return chunk.astype(np.float32)


def window_matrix(df: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    cache: dict[str, dict] = {}
    xs = []
    for r in df.itertuples(index=False):
        sid = str(r.sample_id)
        if sid not in cache:
            path = GOLD_DIR / f"{sid}.npz"
            if not path.exists():
                raise SystemExit(f"STOP: missing pose file {path}")
            cache[sid] = load_npz(path)
        rot = slice_rotation(cache[sid], int(r.start_frame_relative), int(r.end_frame_relative))
        drot = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(rot, axis=0)])
        xs.append(np.concatenate([rot, drot], axis=1))
    X = np.stack(xs).astype(np.float32)
    if mean is None:
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1)) + 1e-6
    X = (X - mean) / std
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y, mean, std


def clip_any(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    for sid, g in df.groupby("sample_id", sort=True):
        idx = g.index.to_numpy()
        rows.append((str(sid), int(g["label"].max()), int(pred[idx].max())))
    ids = [r[0] for r in rows]
    return np.array([r[1] for r in rows], dtype=int), np.array([r[2] for r in rows], dtype=int), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_split_leakage import assert_unlocked_out_dir  # noqa: WPS433

    assert_unlocked_out_dir(out_dir)
    if args.smoke_test:
        out_dir = ROOT / "results" / "windowed_nod" / "pose_cnn_smoke"
        epochs = 2
    else:
        epochs = args.epochs
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        raise SystemExit(
            f"STOP: {metrics_path} already exists. TEST was scored. "
            "Do not --force unless you intend to replace this run."
        )

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise SystemExit(
            "STOP: torch is not available. On otter use "
            "/scratch/db01550/venv/bin/python"
        ) from exc

    dev = _load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    tes = _load_windows(WINDOWS_TEST, "TEST", TEST_IDS)
    if set(dev["sample_id"]) & set(tes["sample_id"]):
        raise SystemExit("STOP: DEV and TEST sample_id overlap")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    Xdv, ydv, mean, std = window_matrix(dev)
    Xte, yte, _, _ = window_matrix(tes, mean, std)
    dump_json(out_dir / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "mode": "C"})

    pos = max(int((ydv == 1).sum()), 1)
    neg = max(int((ydv == 0).sum()), 1)
    model = _build_cnn(nn, int(Xdv.shape[-1]))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.transpose(Xdv, (0, 2, 1))),
            torch.from_numpy(ydv.astype(np.float32)),
        ),
        batch_size=16,
        shuffle=True,
    )

    hist = []
    best = None
    bad = 0
    ckpt = out_dir / "best_1dcnn.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(np.transpose(Xdv, (0, 2, 1)))).numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        thr, dev_m = choose_dev_threshold(ydv, prob)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1": float(dev_m["f1"]),
            "dev_probability_threshold": float(thr),
        }
        hist.append(row)
        print(f"epoch {epoch} loss={row['train_loss']:.4f} DEV window F1={row['dev_f1']:.3f}")
        if best is None or row["dev_f1"] > best["dev_f1"]:
            best = dict(row)
            torch.save(model.state_dict(), ckpt)
            bad = 0
        else:
            bad += 1
            if bad >= 4 and not args.smoke_test:
                break

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        dv = model(torch.from_numpy(np.transpose(Xdv, (0, 2, 1)))).numpy()
        te = model(torch.from_numpy(np.transpose(Xte, (0, 2, 1)))).numpy()
    pdv = 1.0 / (1.0 + np.exp(-dv))
    pte = 1.0 / (1.0 + np.exp(-te))
    thr = float(best["dev_probability_threshold"])
    pred_dv = (pdv >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)
    test_w = clip_binary_metrics(yte, pred_te)
    y_clip, p_clip, clip_ids = clip_any(tes, pred_te)
    test_c = clip_binary_metrics(y_clip, p_clip)
    always1 = clip_binary_metrics(yte, np.ones_like(yte))
    always0 = clip_binary_metrics(yte, np.zeros_like(yte))

    pred_path = out_dir / "predictions.csv"
    pd.concat(
        [
            dev.assign(split="DEV", prob=pdv, pred=pred_dv),
            tes.assign(split="TEST", prob=pte, pred=pred_te),
        ],
        ignore_index=True,
    )[
        [
            "window_id",
            "sample_id",
            "split",
            "start_frame_relative",
            "end_frame_relative",
            "label",
            "prob",
            "pred",
        ]
    ].to_csv(pred_path, index=False)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_nod_3s",
            "feature_set": "C_xyz_deriv",
            "window_sec": 3.0,
            "stride_sec": 2.0,
            "window_frames": WINDOW_FRAMES,
            "selection": "DEV window F1; trained on all DEV windows; TEST once",
            "n_dev_windows": int(len(dev)),
            "n_test_windows": int(len(tes)),
            "n_dev_pos": int((ydv == 1).sum()),
            "n_test_pos": int((yte == 1).sum()),
            "pos_weight": float(neg / pos),
            "best_epoch": int(best["epoch"]),
            "dev_window_f1": float(best["dev_f1"]),
            "dev_probability_threshold": thr,
            "test_window": test_w,
            "test_clip_any_window": test_c,
            "test_clip_ids": clip_ids,
            "majority_always1_window": always1,
            "majority_always0_window": always0,
        },
    )
    print("=====================================")
    print("windowed nod pose CNN (3 s, feature C)")
    print(f"  best epoch (DEV): {best['epoch']}   DEV window F1: {best['dev_f1']:.3f}")
    print(
        f"  TEST window P {test_w['precision']:.3f}  R {test_w['recall']:.3f}  "
        f"F1 {test_w['f1']:.3f}  (TP{test_w['tp']} FP{test_w['fp']} TN{test_w['tn']} FN{test_w['fn']})"
    )
    print(
        f"  TEST clip-any  P {test_c['precision']:.3f}  R {test_c['recall']:.3f}  "
        f"F1 {test_c['f1']:.3f}"
    )
    print(f"  artifacts: {out_dir}")
    print("  locked 60 s folders were not written.")


if __name__ == "__main__":
    main()
