#!/usr/bin/env python3
"""Leave-one-clip-out pose CNN cross-validation on the 3 s DEV windows.

Same folds, metrics and output format as crossval_windowed_videomae_dev.py,
so the pose CNN and the amplitude rule are compared on identical windows.
Feature set C (rotation xyz + first differences). Normalisation statistics
are fitted on the 14 training clips of each fold, never on the held-out clip.

``--return-ratio`` adds pitch return-ratio as a constant 7th channel. That
run writes a new directory and must not overwrite pose_cnn_loco_dev.

DEV only: no TEST input, writes metrics_dev.json.

Otter::

    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_windowed_pose_cnn_dev.py --task nod --return-ratio
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
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.pose_cnn import _build_cnn, load_npz, resample_seq  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    load_windows,
    select_dev_threshold,
)
from evaluate_windowed_nod_motion_ablation import return_ratio as pitch_return_ratio  # noqa: E402

DEV_WINDOWS = {
    "nod": ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv",
    "shake": ROOT / "data" / "windowed_annotations" / "shake_windows_dev.csv",
}
GOLD_DIR = ROOT / "features" / "gold"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
WINDOW_FRAMES = 75
FIXED_THRESHOLD = 0.5


def channels_for_window(
    chunk: np.ndarray, *, return_ratio_channel: bool = False
) -> np.ndarray:
    """Feature set C, optionally with pitch return-ratio broadcast over time."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 3:
        raise SystemExit(f"STOP: expected (T, 3) rotation, got {chunk.shape}")
    diff = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(chunk, axis=0)])
    feat = np.concatenate([chunk, diff], axis=1)
    if return_ratio_channel:
        rr = np.full((len(chunk), 1), pitch_return_ratio(chunk), dtype=np.float32)
        feat = np.concatenate([feat, rr], axis=1)
    return feat.astype(np.float32)


def window_tensor(frame: pd.DataFrame, *, return_ratio_channel: bool = False) -> np.ndarray:
    cache: dict[str, dict] = {}
    rows = []
    for r in frame.itertuples(index=False):
        sid = str(r.sample_id)
        if sid not in cache:
            path = GOLD_DIR / f"{sid}.npz"
            if not path.exists():
                raise SystemExit(f"STOP: missing pose file {path}")
            cache[sid] = load_npz(path)
        rot = np.asarray(cache[sid]["rotation_xyz"], dtype=np.float32)
        i0, i1 = int(r.start_frame_relative), int(r.end_frame_relative)
        if i0 < 0 or i1 > len(rot) or i1 <= i0:
            raise SystemExit(f"STOP: bad slice {i0}:{i1} on {sid}")
        chunk = rot[i0:i1]
        if len(chunk) != WINDOW_FRAMES:
            chunk = resample_seq(chunk, t=WINDOW_FRAMES)
        rows.append(channels_for_window(chunk, return_ratio_channel=return_ratio_channel))
    stacked = np.stack(rows).astype(np.float32)
    expected = 7 if return_ratio_channel else 6
    if stacked.shape[-1] != expected:
        raise SystemExit(f"STOP: expected {expected} channels, got {stacked.shape}")
    return stacked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("nod", "shake"), default="nod")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all 15 folds")
    ap.add_argument(
        "--return-ratio",
        action="store_true",
        help="Add pitch return-ratio as a 7th channel. Writes a new out-dir.",
    )
    args = ap.parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.return_ratio:
        out_dir = (
            ROOT / "results" / f"windowed_{args.task}" / "pose_cnn_loco_dev_return_ratio"
        )
    else:
        out_dir = ROOT / "results" / f"windowed_{args.task}" / "pose_cnn_loco_dev"
    original_loco = ROOT / "results" / f"windowed_{args.task}" / "pose_cnn_loco_dev"
    if args.return_ratio and Path(out_dir).resolve() == original_loco.resolve():
        raise SystemExit("STOP: return-ratio run must not overwrite the original CNN dir")
    out_dir = assert_unlocked_out_dir(out_dir)
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

    frame = load_windows(DEV_WINDOWS[args.task], "DEV", DEV_IDS)
    if set(frame["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id present in a DEV-only script")
    features = window_tensor(frame, return_ratio_channel=args.return_ratio)
    labels = frame["label"].to_numpy(dtype=np.int64)
    sample_ids = frame["sample_id"].to_numpy()
    fold_ids = sorted(set(sample_ids))
    if args.max_folds:
        fold_ids = fold_ids[: args.max_folds]

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"task {args.task} | {len(labels)} windows ({int(labels.sum())} positive) "
        f"from {len(set(sample_ids))} clips | {len(fold_ids)} folds | "
        f"channels {features.shape[-1]}"
        f"{' + return_ratio' if args.return_ratio else ''}"
    )

    oof = np.full(len(labels), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for position, held_out in enumerate(fold_ids, start=1):
        test_mask = sample_ids == held_out
        train_mask = ~test_mask
        x_train, y_train = features[train_mask], labels[train_mask]
        x_held = features[test_mask]

        mean = x_train.mean(axis=(0, 1))
        std = x_train.std(axis=(0, 1)) + 1e-6
        x_train = (x_train - mean) / std
        x_held = (x_held - mean) / std

        model = _build_cnn(nn, int(features.shape[-1]))
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.transpose(x_train, (0, 2, 1))),
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
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            last_loss = float(np.mean(losses)) if losses else float("nan")

        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(np.transpose(x_held, (0, 2, 1)))
            ).numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        oof[test_mask] = probabilities
        fold_rows.append(
            {
                "fold": position,
                "held_out_clip": held_out,
                "n_train_windows": int(train_mask.sum()),
                "n_train_positive": int((y_train == 1).sum()),
                "n_held_windows": int(test_mask.sum()),
                "n_held_positive": int(labels[test_mask].sum()),
                "final_train_loss": last_loss,
                "held_mean_probability": float(np.mean(probabilities)),
            }
        )
        print(
            f"fold {position}/{len(fold_ids)} {held_out} "
            f"loss={last_loss:.4f} held_mean_p={np.mean(probabilities):.3f}",
            flush=True,
        )

    scored = ~np.isnan(oof)
    y_scored, p_scored = labels[scored], oof[scored]
    ids_scored = sample_ids[scored]
    fixed_pred = (p_scored >= FIXED_THRESHOLD).astype(int)
    fixed_metrics = clip_binary_metrics(y_scored, fixed_pred)
    best_threshold, best_metrics, sweep = select_dev_threshold(
        y_scored, p_scored, "balanced_accuracy"
    )
    pr_auc = average_precision(y_scored, p_scored)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_summary.csv", index=False)
    sweep.to_csv(out_dir / "oof_threshold_sweep.csv", index=False)
    pd.DataFrame(
        {
            "window_id": frame.loc[scored, "window_id"].astype(str).to_numpy(),
            "sample_id": ids_scored,
            "label": y_scored,
            "oof_probability": p_scored,
            "pred_at_0.5": fixed_pred,
        }
    ).to_csv(out_dir / "predictions_oof_dev.csv", index=False)
    dump_json(
        metrics_path,
        {
            "protocol": (
                f"windowed_{args.task}_3s_pose_cnn_loco"
                + ("_return_ratio" if args.return_ratio else "")
            ),
            "feature_set": (
                "C_xyz_deriv_return_ratio" if args.return_ratio else "C_xyz_deriv"
            ),
            "return_ratio_channel": bool(args.return_ratio),
            "zero_crossings_used": False,
            "n_channels": int(features.shape[-1]),
            "development_only": True,
            "test_scored": False,
            "cross_validation": "leave-one-clip-out over DEV clips",
            "normalisation": "fitted on training clips of each fold",
            "n_folds": len(fold_rows),
            "folds_complete": len(fold_rows) == 15,
            "epochs_per_fold": args.epochs,
            "epoch_policy": "fixed a priori; no per-fold early stopping",
            "headline_metric": "pr_auc_out_of_fold",
            "pr_auc_out_of_fold": pr_auc,
            "prevalence": float(y_scored.mean()),
            "at_fixed_threshold_0.5": fixed_metrics,
            "best_threshold_upper_bound": {
                "threshold": best_threshold,
                "metrics": best_metrics,
                "caveat": "optimistic: threshold chosen on the scored predictions",
            },
            "always_no": always_predict(y_scored, 0),
            "always_yes": always_predict(y_scored, 1),
            "clip_bootstrap_at_0.5": clip_bootstrap(ids_scored, y_scored, fixed_pred),
            "n_windows_scored": int(scored.sum()),
            "n_positive": int(y_scored.sum()),
            "seed": args.seed,
        },
    )

    print("=====================================")
    print(f"windowed {args.task} pose CNN leave-one-clip-out — DEV only")
    print(
        f"{len(fold_rows)} folds; {int(scored.sum())} out-of-fold windows "
        f"({int(y_scored.sum())} positive, prevalence {y_scored.mean():.3f})"
    )
    print(f"out-of-fold PR AUC {pr_auc:.3f}  (chance = prevalence)")
    print(
        f"at threshold 0.5: balanced accuracy "
        f"{fixed_metrics['balanced_accuracy']:.3f}  "
        f"P {fixed_metrics['precision']:.3f} R {fixed_metrics['recall']:.3f} "
        f"F1 {fixed_metrics['f1']:.3f}"
    )
    print(
        f"best-threshold upper bound (optimistic): "
        f"{best_metrics['balanced_accuracy']:.3f} at {best_threshold:.3f}"
    )
    print("TEST not loaded.")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
