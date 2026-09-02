#!/usr/bin/env python3
"""Nod-only 3 s pose MIL development experiment.

TRAIN consists of 80 independent 60 s pseudo clips with one weak label per
clip. Each clip is represented as a bag of 29 overlapping 3 s windows. The
model learns window scores through a top-2 multiple-instance pooling loss:
a positive bag should contain at least one high-scoring window, while a
negative bag should contain none.

Human DEV windows are used only for epoch and probability-threshold selection.
This script deliberately has no TEST input and writes ``metrics_dev.json``.
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

from check_split_leakage import assert_unlocked_out_dir, run as leakage_gate  # noqa: E402
from src.clip_metrics import choose_dev_threshold, clip_binary_metrics  # noqa: E402
from src.pose_cnn import load_npz  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402

PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
PSEUDO_DIR = ROOT / "features" / "pseudo"
DEV_WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
GOLD_DIR = ROOT / "features" / "gold"
DEFAULT_OUT = ROOT / "results" / "windowed_nod" / "pose_mil_pseudo80_dev"
DEFAULT_CHECKPOINT = ROOT / "models" / "windowed_nod_pose_mil_pseudo80_dev.pt"
TRAIN_IDS = {f"pseudo_{i:05d}" for i in range(1, 81)}
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
WINDOW_STARTS = tuple(range(0, 1401, 50))
WINDOW_FRAMES = 75
TOP_K = 2


def feature_windows(rotation: np.ndarray) -> np.ndarray:
    """Return 29 windows of xyz rotation plus first differences."""
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape != (1500, 3):
        raise SystemExit(f"STOP: expected rotation_xyz (1500, 3), got {rotation.shape}")
    windows = []
    for start in WINDOW_STARTS:
        chunk = rotation[start : start + WINDOW_FRAMES]
        delta = np.vstack(
            [np.zeros((1, 3), dtype=np.float32), np.diff(chunk, axis=0)]
        )
        windows.append(np.concatenate([chunk, delta], axis=1))
    return np.stack(windows).astype(np.float32)


def load_train_bags() -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_csv(PSEUDO_LABELS)
    needed = {"sample_id", "pseudo_label"}
    missing = needed - set(labels.columns)
    if missing:
        raise SystemExit(f"STOP: {PSEUDO_LABELS.name} missing {sorted(missing)}")
    labels["sample_id"] = labels["sample_id"].astype(str)
    if set(labels["sample_id"]) != TRAIN_IDS or len(labels) != len(TRAIN_IDS):
        raise SystemExit("STOP: expected exactly pseudo_00001–pseudo_00080")
    labels = labels.set_index("sample_id")
    bags = []
    ids = sorted(TRAIN_IDS)
    y = []
    for sid in ids:
        path = PSEUDO_DIR / f"{sid}.npz"
        if not path.exists():
            raise SystemExit(f"STOP: missing TRAIN pose {path}")
        pose = load_npz(path)
        embedded = str(np.asarray(pose["sample_id"]).item())
        if embedded != sid:
            raise SystemExit(f"STOP: {path.name} embeds sample_id {embedded}")
        bags.append(feature_windows(pose["rotation_xyz"]))
        y.append(int(labels.loc[sid, "pseudo_label"]))
    y_array = np.asarray(y, dtype=np.int64)
    if set(np.unique(y_array)) != {0, 1}:
        raise SystemExit("STOP: weak TRAIN labels need both classes")
    return np.stack(bags), y_array, ids


def load_dev_windows() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(DEV_WINDOWS)
    needed = {
        "window_id",
        "sample_id",
        "split",
        "start_frame_relative",
        "end_frame_relative",
        "label",
    }
    missing = needed - set(frame.columns)
    if missing:
        raise SystemExit(f"STOP: {DEV_WINDOWS.name} missing {sorted(missing)}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["split"] = frame["split"].astype(str).str.upper()
    if (frame["split"] != "DEV").any() or set(frame["sample_id"]) != DEV_IDS:
        raise SystemExit("STOP: human DEV window split is incomplete or contaminated")
    if len(frame) != 15 * 29:
        raise SystemExit("STOP: expected 435 human DEV windows")
    cache: dict[str, np.ndarray] = {}
    windows = []
    for row in frame.itertuples(index=False):
        sid = str(row.sample_id)
        if sid not in cache:
            cache[sid] = np.asarray(
                load_npz(GOLD_DIR / f"{sid}.npz")["rotation_xyz"],
                dtype=np.float32,
            )
        start = int(row.start_frame_relative)
        end = int(row.end_frame_relative)
        chunk = cache[sid][start:end]
        if chunk.shape != (WINDOW_FRAMES, 3):
            raise SystemExit(f"STOP: bad DEV slice {sid} {start}:{end}")
        delta = np.vstack(
            [np.zeros((1, 3), dtype=np.float32), np.diff(chunk, axis=0)]
        )
        windows.append(np.concatenate([chunk, delta], axis=1))
    return (
        np.stack(windows).astype(np.float32),
        frame["label"].to_numpy(dtype=np.int64),
        frame.reset_index(drop=True),
    )


def normalise(
    train: np.ndarray, dev: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalisation on weak TRAIN only, then apply it to DEV."""
    mean = train.mean(axis=(0, 1, 2))
    std = train.std(axis=(0, 1, 2)) + 1e-6
    return (train - mean) / std, (dev - mean) / std, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = assert_unlocked_out_dir(args.out_dir)
    metrics_path = out_dir / "metrics_dev.json"
    if metrics_path.exists():
        raise SystemExit(
            f"STOP: {metrics_path} exists. Use a new out-dir for another DEV experiment."
        )
    checkpoint = args.checkpoint.resolve()
    if checkpoint.parent != (ROOT / "models").resolve():
        raise SystemExit("STOP: pose MIL checkpoint must be stored directly under models/")

    leakage_gate(
        gold_csv=ROOT / "data" / "gold_annotations.csv",
        pseudo_labels=PSEUDO_LABELS,
        labelled_train_only=True,
    )
    set_seed(args.seed)
    train_bags, y_train, train_ids = load_train_bags()
    dev_windows, y_dev, dev_frame = load_dev_windows()
    train_bags, dev_windows, mean, std = normalise(train_bags, dev_windows)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "STOP: torch missing. Run on otter with /scratch/db01550/venv/bin/python"
        ) from exc

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    class PoseMIL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(6, 32, 5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(64, 1)

        def window_logits(self, windows):
            shape = windows.shape
            flat = windows.reshape(-1, shape[-2], shape[-1]).transpose(1, 2)
            encoded = self.encoder(flat).squeeze(-1)
            return self.head(encoded).squeeze(-1).reshape(shape[:-2])

        def forward(self, bags):
            logits = self.window_logits(bags)
            k = min(TOP_K, logits.shape[1])
            return torch.topk(logits, k=k, dim=1).values.mean(dim=1)

    model = PoseMIL().to(device)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_bags),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_tensor = torch.from_numpy(dev_windows).to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        out_dir / "normalization.json",
        {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "TRAIN"},
    )
    history = []
    best = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bags, labels in loader:
            bags, labels = bags.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(bags), labels)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model.window_logits(dev_tensor).cpu().numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        threshold, dev_metrics = choose_dev_threshold(y_dev, probabilities)
        row = {
            "epoch": epoch,
            "train_bag_loss": float(np.mean(losses)),
            "dev_window_f1": float(dev_metrics["f1"]),
            "dev_balanced_accuracy": float(dev_metrics["balanced_accuracy"]),
            "dev_probability_threshold": float(threshold),
        }
        history.append(row)
        print(
            f"epoch {epoch} loss={row['train_bag_loss']:.4f} "
            f"DEV F1={row['dev_window_f1']:.3f} "
            f"bacc={row['dev_balanced_accuracy']:.3f}"
        )
        key = (row["dev_window_f1"], row["dev_balanced_accuracy"])
        if best is None or key > best["key"]:
            best = {"key": key, **row}
            torch.save(model.state_dict(), checkpoint)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best is not None
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model.window_logits(dev_tensor).cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= float(best["dev_probability_threshold"])).astype(int)
    dev_metrics = clip_binary_metrics(y_dev, predictions)
    dev_out = dev_frame[
        [
            "window_id",
            "sample_id",
            "split",
            "start_frame_relative",
            "end_frame_relative",
            "label",
        ]
    ].copy()
    dev_out["probability"] = probabilities
    dev_out["prediction"] = predictions
    dev_out.to_csv(out_dir / "predictions_dev.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_nod_3s_pose_mil",
            "development_only": True,
            "test_scored": False,
            "training": (
                "80 weakly labelled 60 s TRAIN bags; 29 windows per bag; "
                "top-2 multiple-instance pooling"
            ),
            "selection": "human DEV window F1 and threshold only",
            "feature_set": "rotation_xyz_plus_first_difference",
            "window_sec": 3.0,
            "stride_sec": 2.0,
            "n_train_clips": len(train_ids),
            "n_train_positive_bags": n_pos,
            "n_train_negative_bags": n_neg,
            "n_train_instances": int(len(train_ids) * len(WINDOW_STARTS)),
            "weak_label_pos_weight": float(n_neg / n_pos),
            "n_dev_windows": int(len(y_dev)),
            "n_dev_positive": int(y_dev.sum()),
            "best_epoch": int(best["epoch"]),
            "dev_probability_threshold": float(best["dev_probability_threshold"]),
            "dev_window": dev_metrics,
            "device": str(device),
            "seed": args.seed,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        },
    )
    print("=====================================")
    print("windowed nod pose MIL — DEV only")
    print(
        f"TRAIN {len(train_ids)} weak bags ({n_pos} positive/{n_neg} negative); "
        f"{len(train_ids) * len(WINDOW_STARTS)} window instances"
    )
    print(
        f"best epoch {best['epoch']}  DEV P {dev_metrics['precision']:.3f} "
        f"R {dev_metrics['recall']:.3f} F1 {dev_metrics['f1']:.3f}"
    )
    print("TEST was not loaded or scored.")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
