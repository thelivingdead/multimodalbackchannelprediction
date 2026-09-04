#!/usr/bin/env python3
"""DEV LOCO Pose CNN with a separate amplitude/return-ratio scalar branch.

Does not broadcast return ratio across time. Does not load TEST.
Does not overwrite results/windowed_nod/pose_cnn_loco_dev.

Otter::

    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_windowed_pose_cnn_scalar_branch_dev.py
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

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from crossval_windowed_pose_cnn_dev import (  # noqa: E402
    WINDOW_FRAMES,
    window_tensor,
)
from evaluate_windowed_nod_motion_ablation import return_ratio  # noqa: E402
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.pose_cnn import load_npz  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    load_windows,
    rule_score_function,
    select_dev_threshold,
)

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
GOLD_DIR = ROOT / "features" / "gold"
OUT_DIR = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_scalar_branch"
BLOCKED = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
FIXED_THRESHOLD = 0.5


def scalar_features(frame: pd.DataFrame) -> np.ndarray:
    rule_score = rule_score_function()
    cache: dict[str, np.ndarray] = {}
    rows = []
    for rec in frame.itertuples(index=False):
        sid = str(rec.sample_id)
        if sid in TEST_IDS:
            raise SystemExit(f"STOP: TEST id {sid}")
        if sid not in cache:
            cache[sid] = np.asarray(
                load_npz(GOLD_DIR / f"{sid}.npz")["rotation_xyz"], dtype=np.float32
            )
        chunk = cache[sid][int(rec.start_frame_relative) : int(rec.end_frame_relative)]
        if len(chunk) != WINDOW_FRAMES:
            raise SystemExit(f"STOP: {sid} window is not 75 frames")
        rows.append([float(rule_score(chunk, 0)), float(return_ratio(chunk))])
    return np.asarray(rows, dtype=np.float32)


def build_two_branch(nn):
    class PoseCNNScalarBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Sequential(
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
            self.scalar = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
            )
            self.fc = nn.Linear(64 + 4, 1)

        def forward(self, seq, scalars):
            import torch

            h = self.temporal(seq).squeeze(-1)
            s = self.scalar(scalars)
            return self.fc(torch.cat([h, s], dim=1)).squeeze(-1)

    return PoseCNNScalarBranch()


def main() -> None:
    print("POSE CNN SCALAR-BRANCH RUN")
    print("Temporal: 75 x 6  (xyz + dxyz), fold z-score on train clips")
    print("Scalar: amplitude + return_ratio, fold z-score on train clips, MLP 2→8→4")
    print("Return ratio is NOT copied across timesteps")
    print("TEST will not be read")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if Path(args.out_dir).resolve() == BLOCKED.resolve():
        raise SystemExit("STOP: will not overwrite the locked original CNN")
    out_dir = assert_unlocked_out_dir(args.out_dir)
    metrics_path = out_dir / "metrics_dev.json"
    if metrics_path.exists():
        raise SystemExit(f"STOP: {metrics_path} exists.")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "STOP: torch missing. On otter use /scratch/db01550/venv/bin/python"
        ) from exc

    frame = load_windows(WINDOWS, "DEV", DEV_IDS)
    if set(frame["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id present")
    seq = window_tensor(frame, return_ratio_channel=False)
    if seq.shape[-1] != 6:
        raise SystemExit("STOP: temporal branch must stay 6 channels")
    scalars = scalar_features(frame)
    labels = frame["label"].to_numpy(dtype=np.int64)
    sample_ids = frame["sample_id"].to_numpy()
    fold_ids = sorted(set(sample_ids))
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    oof = np.full(len(labels), np.nan)
    fold_rows = []
    for position, held in enumerate(fold_ids, start=1):
        held_mask = sample_ids == held
        train = ~held_mask
        seq_mean = seq[train].mean(axis=(0, 1))
        seq_std = seq[train].std(axis=(0, 1)) + 1e-6
        sc_mean = scalars[train].mean(axis=0)
        sc_std = scalars[train].std(axis=0)
        sc_std = np.where(sc_std < 1e-8, 1.0, sc_std)
        x_seq = (seq[train] - seq_mean) / seq_std
        x_sc = (scalars[train] - sc_mean) / sc_std
        h_seq = (seq[held_mask] - seq_mean) / seq_std
        h_sc = (scalars[held_mask] - sc_mean) / sc_std
        y_train = labels[train]
        model = build_two_branch(nn)
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.transpose(x_seq, (0, 2, 1))),
                torch.from_numpy(x_sc),
                torch.from_numpy(y_train.astype(np.float32)),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + position),
        )
        last_loss = float("nan")
        for _ in range(args.epochs):
            model.train()
            losses = []
            for xb, sb, yb in loader:
                opt.zero_grad()
                loss = criterion(model(xb, sb), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            last_loss = float(np.mean(losses)) if losses else float("nan")
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(np.transpose(h_seq, (0, 2, 1))),
                torch.from_numpy(h_sc),
            ).numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        oof[held_mask] = probabilities
        fold_rows.append(
            {
                "fold": position,
                "held_out_clip": held,
                "final_train_loss": last_loss,
                "held_mean_probability": float(np.mean(probabilities)),
                "scalar_train_mean": [float(v) for v in sc_mean],
                "scalar_train_std": [float(v) for v in sc_std],
            }
        )
        print(
            f"fold {position}/15 {held} loss={last_loss:.4f} "
            f"held_mean_p={np.mean(probabilities):.3f}",
            flush=True,
        )
    pred = (oof >= FIXED_THRESHOLD).astype(int)
    metrics = clip_binary_metrics(labels, pred)
    boot = clip_bootstrap(sample_ids, labels, pred)
    pr_auc = average_precision(labels, oof)
    _, best_metrics, sweep = select_dev_threshold(labels, oof, "balanced_accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_summary.csv", index=False)
    sweep.to_csv(out_dir / "oof_threshold_sweep.csv", index=False)
    pd.DataFrame(
        {
            "window_id": frame["window_id"].astype(str),
            "sample_id": sample_ids,
            "label": labels,
            "oof_probability": oof,
            "pred_at_0.5": pred,
        }
    ).to_csv(out_dir / "predictions_oof_dev.csv", index=False)
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_nod_3s_pose_cnn_loco_scalar_branch",
            "development_only": True,
            "test_scored": False,
            "return_ratio_broadcast": False,
            "temporal_channels": 6,
            "scalar_features": ["amplitude", "return_ratio"],
            "scalar_mlp": [2, 8, 4],
            "normalisation": "train-fold z-score; sequence over time+windows; scalars over windows",
            "epochs_per_fold": args.epochs,
            "threshold": FIXED_THRESHOLD,
            "n_windows_scored": int(len(labels)),
            "n_positive": int(labels.sum()),
            "at_fixed_threshold_0.5": metrics,
            "pr_auc_out_of_fold": pr_auc,
            "clip_bootstrap_at_0.5": boot,
            "always_no": always_predict(labels, 0),
            "always_yes": always_predict(labels, 1),
            "best_threshold_upper_bound": {
                "metrics": best_metrics,
                "caveat": "optimistic: threshold chosen on scored predictions",
            },
            "seed": args.seed,
        },
    )
    print(
        f"scalar-branch BA {metrics['balanced_accuracy']:.3f}  "
        f"F1 {metrics['f1']:.3f}  PR AUC {pr_auc:.3f}  "
        f"TP{metrics['tp']} FP{metrics['fp']} TN{metrics['tn']} FN{metrics['fn']}"
    )
    print("TEST not loaded.")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
