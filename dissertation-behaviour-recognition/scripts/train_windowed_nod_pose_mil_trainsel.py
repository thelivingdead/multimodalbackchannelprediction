#!/usr/bin/env python3
"""Nod 3 s pose MIL trained on the 80 TRAIN bags, with DEV scored properly.

The earlier run (``pose_mil_pseudo80_dev_bacc``) chose both the epoch and the
probability threshold on human DEV windows and then reported those same
windows, so its 0.533 balanced accuracy is selection-contaminated and reads as
an upper bound rather than an estimate.

Here every selection decision is made inside TRAIN, using only the weak
clip-level pseudo labels:

  stage 1  five-fold cross-validation over the 80 bags picks the epoch, scored
           by out-of-fold bag-level balanced accuracy
  stage 2  the window threshold is swept on those same out-of-fold bag scores,
           calling a bag positive when any of its 29 windows clears it
  stage 3  the model is refitted on all 80 bags for the chosen epoch and the
           frozen threshold is applied to human DEV windows exactly once

DEV therefore behaves as a held-out set: it is read after selection is closed,
never during it. TEST is not loaded. The DEV-selected oracle is also reported,
clearly labelled, so the cost of honest selection can be quantified.

Otter, nod (already run)::

    /scratch/db01550/venv/bin/python scripts/train_windowed_nod_pose_mil_trainsel.py

Otter, shake::

    /scratch/db01550/venv/bin/python scripts/train_windowed_nod_pose_mil_trainsel.py --task shake
    bash scripts/run_windowed_shake_3s_otter.sh
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
from src.utils import dump_json, set_seed  # noqa: E402
from src.windowed_baselines import clip_bootstrap  # noqa: E402
from train_windowed_nod_pose_mil_dev import (  # noqa: E402
    TOP_K,
    WINDOW_STARTS,
    load_dev_windows,
    load_train_bags,
)

TASKS = {
    "nod": {
        "gold_csv": ROOT / "data" / "gold_annotations.csv",
        "labels": ROOT / "results" / "pseudo_labels.csv",
        "windows": ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv",
        "out": ROOT / "results" / "windowed_nod" / "pose_mil_pseudo80_trainsel",
        "checkpoint": ROOT / "models" / "windowed_nod_pose_mil_pseudo80_trainsel.pt",
    },
    "shake": {
        "gold_csv": ROOT / "data" / "gold" / "shake_annotation_sheet.csv",
        "labels": ROOT / "results" / "shake" / "pseudo_balanced" / "manifest_40_40.csv",
        "windows": ROOT / "data" / "windowed_annotations" / "shake_windows_dev.csv",
        "out": ROOT / "results" / "windowed_shake" / "pose_mil_balanced40_trainsel",
        "checkpoint": ROOT / "models" / "windowed_shake_pose_mil_balanced40_trainsel.pt",
    },
}
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Fold assignment that keeps the 70/10 bag balance in every fold."""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(labels), dtype=int)
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        rng.shuffle(members)
        assignment[members] = np.arange(len(members)) % n_folds
    return [np.flatnonzero(assignment == fold) for fold in range(n_folds)]


def bag_balanced_accuracy(
    bag_labels: np.ndarray, window_probabilities: np.ndarray, threshold: float
) -> float:
    """Bag is positive when any window clears the threshold."""
    predicted = (window_probabilities >= threshold).any(axis=1).astype(int)
    return float(clip_binary_metrics(bag_labels, predicted)["balanced_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("nod", "shake"), default="nod")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    spec = TASKS[args.task]
    labels_path = (args.labels or spec["labels"]).resolve()
    windows_path = spec["windows"]
    out_dir = assert_unlocked_out_dir(args.out_dir or spec["out"])
    metrics_path = out_dir / "metrics_dev.json"
    if metrics_path.exists():
        raise SystemExit(
            f"STOP: {metrics_path} exists. Use a new out-dir for another experiment."
        )
    checkpoint = (args.checkpoint or spec["checkpoint"]).resolve()
    if checkpoint.parent != (ROOT / "models").resolve():
        raise SystemExit("STOP: checkpoint must be stored directly under models/")

    leakage_gate(
        gold_csv=spec["gold_csv"],
        pseudo_labels=labels_path,
        labelled_train_only=True,
    )
    set_seed(args.seed)
    train_bags, y_train, train_ids = load_train_bags(labels_path)
    dev_windows, y_dev, dev_frame = load_dev_windows(windows_path)

    # Normalisation is fitted on TRAIN bags only, never on DEV.
    mean = train_bags.mean(axis=(0, 1, 2))
    std = train_bags.std(axis=(0, 1, 2)) + 1e-6
    train_bags = (train_bags - mean) / std
    dev_windows = (dev_windows - mean) / std

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
        """Identical architecture to the DEV-selected run."""

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

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    def fit(
        fit_index: np.ndarray, score_bags: np.ndarray, n_epochs: int
    ) -> tuple[np.ndarray, "torch.nn.Module"]:
        """Train on fit_index, returning per-epoch window probabilities."""
        model = PoseMIL().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_bags[fit_index]),
                torch.from_numpy(y_train[fit_index].astype(np.float32)),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        held = torch.from_numpy(score_bags).to(device)
        per_epoch = []
        for _ in range(n_epochs):
            model.train()
            for bags, labels in loader:
                bags, labels = bags.to(device), labels.to(device)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(model(bags), labels)
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                logits = model.window_logits(held).cpu().numpy()
            per_epoch.append(1.0 / (1.0 + np.exp(-logits)))
        return np.stack(per_epoch), model

    # ---- stages 1 and 2: epoch and threshold from TRAIN bags only -----------
    print(f"stage 1: {args.folds}-fold CV over {len(train_ids)} TRAIN bags")
    folds = stratified_folds(y_train, args.folds, args.seed)
    oof = np.zeros((args.epochs, len(train_ids), len(WINDOW_STARTS)), dtype=np.float64)
    for number, held_index in enumerate(folds, start=1):
        fit_index = np.setdiff1d(np.arange(len(train_ids)), held_index)
        probabilities, _ = fit(fit_index, train_bags[held_index], args.epochs)
        oof[:, held_index, :] = probabilities
        print(
            f"  fold {number}/{len(folds)}: {len(fit_index)} fit bags, "
            f"{len(held_index)} held out"
        )

    sweep = []
    for epoch in range(args.epochs):
        for threshold in THRESHOLD_GRID:
            sweep.append(
                {
                    "epoch": epoch + 1,
                    "threshold": float(threshold),
                    "oof_bag_balanced_accuracy": bag_balanced_accuracy(
                        y_train, oof[epoch], float(threshold)
                    ),
                }
            )
    sweep_frame = pd.DataFrame(sweep)
    best = sweep_frame.sort_values(
        ["oof_bag_balanced_accuracy", "epoch"], ascending=[False, True]
    ).iloc[0]
    best_epoch = int(best["epoch"])
    best_threshold = float(best["threshold"])
    print(
        f"stage 2: epoch {best_epoch}, threshold {best_threshold:.2f} "
        f"(out-of-fold bag balanced accuracy "
        f"{best['oof_bag_balanced_accuracy']:.3f}); DEV not yet read"
    )

    # ---- stage 3: refit on all 80 bags, then read DEV once -------------------
    print(f"stage 3: refit on all {len(train_ids)} bags for {best_epoch} epochs")
    all_index = np.arange(len(train_ids))
    dev_probability_stack, model = fit(all_index, dev_windows, best_epoch)
    probabilities = dev_probability_stack[-1]
    torch.save(model.state_dict(), checkpoint)

    predictions = (probabilities >= best_threshold).astype(int)
    dev_metrics = clip_binary_metrics(y_dev, predictions)
    sample_ids = dev_frame["sample_id"].astype(str).to_numpy()
    boot = clip_bootstrap(sample_ids, y_dev, predictions)

    # Reported only to show what DEV-based selection would have bought.
    oracle_threshold, oracle_metrics = choose_dev_threshold(
        y_dev, probabilities, criterion="balanced_accuracy"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    sweep_frame.to_csv(out_dir / "train_oof_selection_sweep.csv", index=False)
    dump_json(
        out_dir / "normalization.json",
        {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "TRAIN"},
    )
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
    dump_json(
        metrics_path,
        {
            "protocol": f"windowed_{args.task}_3s_pose_mil_train_selected",
            "task": args.task,
            "weak_labels": str(labels_path),
            "development_only": True,
            "test_scored": False,
            "training": (
                f"{len(train_ids)} weakly labelled 60 s TRAIN bags; "
                f"{len(WINDOW_STARTS)} windows per bag; top-{TOP_K} "
                "multiple-instance pooling"
            ),
            "selection": (
                f"TRAIN only: {args.folds}-fold cross-validation over the weak bags "
                "chose the epoch and the window threshold, by out-of-fold bag-level "
                "balanced accuracy. DEV was read once, after selection closed."
            ),
            "why_this_run_exists": (
                "The pose_mil_pseudo80_dev_bacc run selected its epoch and threshold "
                "on the same DEV windows it reported, so its balanced accuracy is an "
                "upper bound. This run makes DEV a genuine held-out estimate."
            ),
            "headline_metric": "balanced_accuracy",
            "headline_metric_floor": 0.5,
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
            "dev_prevalence": float(y_dev.mean()),
            "selected_epoch": best_epoch,
            "selected_threshold": best_threshold,
            "train_oof_bag_balanced_accuracy": float(
                best["oof_bag_balanced_accuracy"]
            ),
            "dev_window": dev_metrics,
            "dev_clip_bootstrap": boot,
            "dev_selected_oracle": {
                "note": (
                    "Not a result. Best DEV balanced accuracy reachable by tuning the "
                    "threshold on DEV itself, quoted to size the selection effect."
                ),
                "threshold": float(oracle_threshold),
                "balanced_accuracy": float(oracle_metrics["balanced_accuracy"]),
            },
            "device": str(device),
            "seed": args.seed,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        },
    )

    interval = boot["balanced_accuracy"]
    print("=====================================")
    print(f"windowed {args.task} pose MIL, TRAIN selected. DEV scored once")
    print(
        f"TRAIN {len(train_ids)} weak bags ({n_pos} positive/{n_neg} negative); "
        f"{len(train_ids) * len(WINDOW_STARTS)} window instances"
    )
    print(f"selected on TRAIN: epoch {best_epoch}, threshold {best_threshold:.2f}")
    print(
        f"DEV balanced accuracy {dev_metrics['balanced_accuracy']:.3f} "
        f"[{interval['ci_lower_95']:.3f}, {interval['ci_upper_95']:.3f}]  "
        f"P {dev_metrics['precision']:.3f} R {dev_metrics['recall']:.3f} "
        f"F1 {dev_metrics['f1']:.3f}"
    )
    print(
        f"for reference, DEV-selected oracle would give "
        f"{oracle_metrics['balanced_accuracy']:.3f} at threshold "
        f"{float(oracle_threshold):.2f} (not a result)"
    )
    if interval["ci_lower_95"] <= 0.5:
        print(
            "VERDICT: the interval contains 0.500, so training on the 80 weak bags "
            "does not demonstrate above-chance window detection."
        )
    else:
        print(
            "VERDICT: the interval clears 0.500. Weak supervision at scale helps; "
            "this is the system to take to TEST."
        )
    print("TEST was not loaded or scored.")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
