#!/usr/bin/env python3
"""DEV-only Pose CNN receptive-field ablation on the 3 s windows.

Same data, Feature set C (6 channels), LOCO folds, optimiser, class
weighting, epochs and threshold as the locked windowed CNN. Only the
three Conv1d kernel sizes change.

The locked windowed input is 75 frames at 25 fps (3.0 s). It is not the
128-step resample used by the 60 s clip CNN in src/pose_cnn.py. At 25 fps
RF 11 / 25 / 47 steps = 0.44 s / 1.00 s / 1.88 s. The 128-step claim
(47 steps ≈ 1.1 s) does not apply to this protocol.

Does not add return-ratio. Does not use the scalar branch. Does not
overwrite pose_cnn_loco_dev. Fusion search and the TEST return-ratio
rule are untouched.

Otter::

    bash scripts/run_pose_cnn_rf_ablation_otter.sh
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

from check_split_leakage import (  # noqa: E402
    assert_unlocked_out_dir,
    path_under,
    resolve_repo_path,
)
from crossval_windowed_pose_cnn_dev import WINDOW_FRAMES, window_tensor  # noqa: E402
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.pose_cnn import (  # noqa: E402
    DEFAULT_KERNELS,
    WINDOWED_FPS,
    _build_cnn,
    conv_paddings,
    conv_receptive_field,
    receptive_field_seconds,
)
from src.utils import dump_json, set_seed  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    load_windows,
    select_dev_threshold,
)

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
LOCKED_CNN_DIR = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev"
LOCKED_METRICS = LOCKED_CNN_DIR / "metrics_dev.json"
OUT_DIR = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_rf_ablation"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
FIXED_THRESHOLD = 0.5
FPS = WINDOWED_FPS
LOCKED_CNN_BA = 0.5234233781883912
LOCKED_CNN_CI = (0.46853887717234227, 0.5831911636045494)
TEST_MARGIN = 0.03

CONFIGS = (
    {"name": "k5_5_3", "kernels": (5, 5, 3), "role": "baseline"},
    {"name": "k11_9_7", "kernels": (11, 9, 7), "role": "proposed"},
    {"name": "k21_15_13", "kernels": (21, 15, 13), "role": "larger"},
)


def refuse_test_ids(sample_ids) -> None:
    hit = set(map(str, sample_ids)) & TEST_IDS
    if hit:
        raise SystemExit(f"STOP: TEST id present: {sorted(hit)}")


def assert_ablation_out_dir(out_dir: Path) -> Path:
    resolved = resolve_repo_path(out_dir)
    locked = resolve_repo_path(LOCKED_CNN_DIR)
    if resolved == locked or path_under(resolved, locked):
        raise SystemExit("STOP: will not overwrite the locked original CNN")
    return assert_unlocked_out_dir(resolved)


def describe_config(cfg: dict) -> dict:
    kernels = tuple(int(k) for k in cfg["kernels"])
    steps = conv_receptive_field(kernels)
    seconds = receptive_field_seconds(kernels, fps=FPS)
    return {
        "name": cfg["name"],
        "role": cfg["role"],
        "kernels": list(kernels),
        "paddings": list(conv_paddings(kernels)),
        "rf_steps": int(steps),
        "rf_seconds": float(seconds),
        "fps": float(FPS),
        "input_frames": int(WINDOW_FRAMES),
        "input_seconds": float(WINDOW_FRAMES / FPS),
    }


def decide(rows: list[dict], locked_ba: float = LOCKED_CNN_BA, margin: float = TEST_MARGIN) -> dict:
    larger = [row for row in rows if row.get("role") != "baseline"]
    if not larger:
        raise SystemExit("STOP: no larger-kernel row to compare")
    best = max(larger, key=lambda row: float(row["balanced_accuracy"]))
    delta = float(best["balanced_accuracy"]) - float(locked_ba)
    clear = delta >= float(margin)
    text = (
        f"Do not score TEST. {best['name']} is {delta:+.3f} vs locked CNN "
        f"{locked_ba:.3f} (margin {margin:.2f})."
    )
    if not clear:
        text = (
            f"Do not score TEST. No larger-kernel config beats the locked CNN "
            f"{locked_ba:.3f} by {margin:.2f} BA."
        )
    return {
        "locked_cnn_ba": float(locked_ba),
        "locked_cnn_ba_ci": [float(LOCKED_CNN_CI[0]), float(LOCKED_CNN_CI[1])],
        "margin_required": float(margin),
        "best_larger_kernel": best["name"],
        "best_larger_kernel_ba": float(best["balanced_accuracy"]),
        "delta_vs_locked": float(delta),
        "clear_improvement": bool(clear),
        "test_authorised": False,
        "text": text,
    }


def figure_ba(rows: list[dict], stem: Path, locked_ba: float = LOCKED_CNN_BA) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figure_style import BLUE, GREY, INK, MUTED, ORANGE, PAPER, SIZE_FULL, save

    colours = {"baseline": GREY, "proposed": BLUE, "larger": ORANGE}
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    xs = np.arange(len(rows))
    ba = [float(row["balanced_accuracy"]) for row in rows]
    lo = [float(row["ci_lower"]) for row in rows]
    hi = [float(row["ci_upper"]) for row in rows]
    bar_colours = [colours.get(row.get("role"), BLUE) for row in rows]
    labels = [
        f"{','.join(str(k) for k in row['kernels'])}\nRF {row['rf_steps']}  ({row['rf_seconds']:.2f} s)"
        for row in rows
    ]
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, zorder=0)
    ax.axhline(float(locked_ba), color=MUTED, ls=":", lw=1.0, zorder=1)
    ax.bar(xs, ba, color=bar_colours, width=0.55, edgecolor=PAPER, zorder=2)
    ax.errorbar(
        xs,
        ba,
        yerr=[np.array(ba) - np.array(lo), np.array(hi) - np.array(ba)],
        fmt="none",
        ecolor=INK,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("DEV leave-one-clip-out. Kernel / receptive-field ablation.")
    ax.text(len(rows) - 0.55, 0.51, "chance  0.500", color=MUTED, ha="right", va="bottom")
    ax.text(
        len(rows) - 0.55,
        float(locked_ba) + 0.012,
        f"locked CNN  {locked_ba:.3f}",
        color=MUTED,
        ha="right",
        va="bottom",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    fig.text(
        0.12,
        0.035,
        "75 frames at 25 fps (3.0 s), not 128 resampled steps. Same folds and threshold 0.5.\n"
        "Whiskers are 95% clip-level intervals. Fusion search and the TEST return-ratio "
        "rule were not changed. TEST was not scored.",
        color=MUTED,
    )
    save(fig, stem)


def run_loco(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    window_ids: np.ndarray,
    kernels: tuple[int, int, int],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    nn,
    torch,
    DataLoader,
    TensorDataset,
) -> dict:
    metrics_path = out_dir / "metrics_dev.json"
    if metrics_path.exists():
        raise SystemExit(f"STOP: {metrics_path} exists.")
    if features.shape[1] != WINDOW_FRAMES or features.shape[2] != 6:
        raise SystemExit(
            f"STOP: expected ({WINDOW_FRAMES}, 6) window tensor, got {features.shape}"
        )
    fold_ids = sorted(set(sample_ids))
    set_seed(seed)
    torch.manual_seed(seed)
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
        model = _build_cnn(nn, 6, kernels=kernels)
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
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + position),
        )
        last_loss = float("nan")
        for _ in range(epochs):
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
            logits = model(torch.from_numpy(np.transpose(x_held, (0, 2, 1)))).numpy()
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
    boot = clip_bootstrap(ids_scored, y_scored, fixed_pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_summary.csv", index=False)
    sweep.to_csv(out_dir / "oof_threshold_sweep.csv", index=False)
    pd.DataFrame(
        {
            "window_id": window_ids[scored],
            "sample_id": ids_scored,
            "label": y_scored,
            "oof_probability": p_scored,
            "pred_at_0.5": fixed_pred,
        }
    ).to_csv(out_dir / "predictions_oof_dev.csv", index=False)
    info = describe_config({"name": out_dir.name, "role": "", "kernels": kernels})
    payload = {
        "protocol": "windowed_nod_3s_pose_cnn_loco_rf_ablation",
        "feature_set": "C_xyz_deriv",
        "return_ratio_channel": False,
        "scalar_branch": False,
        "development_only": True,
        "test_scored": False,
        "cross_validation": "leave-one-clip-out over DEV clips",
        "normalisation": "fitted on training clips of each fold",
        "n_channels": 6,
        "sequence_length": int(WINDOW_FRAMES),
        "fps": float(FPS),
        "kernels": list(kernels),
        "paddings": info["paddings"],
        "receptive_field_steps": info["rf_steps"],
        "receptive_field_seconds": info["rf_seconds"],
        "n_folds": len(fold_rows),
        "folds_complete": len(fold_rows) == 15,
        "epochs_per_fold": epochs,
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
        "clip_bootstrap_at_0.5": boot,
        "n_windows_scored": int(scored.sum()),
        "n_positive": int(y_scored.sum()),
        "seed": seed,
    }
    dump_json(metrics_path, payload)
    print(
        f"{out_dir.name} BA {fixed_metrics['balanced_accuracy']:.3f}  "
        f"F1 {fixed_metrics['f1']:.3f}  PR AUC {pr_auc:.3f}"
    )
    return payload


def comparison_row(cfg: dict, payload: dict) -> dict:
    info = describe_config(cfg)
    metrics = payload["at_fixed_threshold_0.5"]
    boot = payload["clip_bootstrap_at_0.5"]["balanced_accuracy"]
    return {
        **info,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "pr_auc": float(payload["pr_auc_out_of_fold"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "ci_lower": float(boot["ci_lower_95"]),
        "ci_upper": float(boot["ci_upper_95"]),
    }


def write_comparison(out_dir: Path, rows: list[dict]) -> dict:
    decision = decide(rows)
    payload = {
        "protocol": "windowed_nod_3s_pose_cnn_loco_rf_ablation",
        "development_only": True,
        "test_scored": False,
        "test_authorised": False,
        "fusion_search_untouched": True,
        "test_return_ratio_rule_untouched": True,
        "input_protocol": {
            "sequence_length": int(WINDOW_FRAMES),
            "fps": float(FPS),
            "window_seconds": float(WINDOW_FRAMES / FPS),
            "not_128_resampled_steps": True,
            "note": (
                "128-step resampling is the 60 s clip CNN in src/pose_cnn.py, "
                "not this windowed LOCO protocol. RF 47 steps is 1.88 s at "
                "25 fps, not 1.1 s."
            ),
        },
        "locked_cnn": {
            "path": "results/windowed_nod/pose_cnn_loco_dev/metrics_dev.json",
            "balanced_accuracy": LOCKED_CNN_BA,
            "ci_95": [LOCKED_CNN_CI[0], LOCKED_CNN_CI[1]],
            "kernels": list(DEFAULT_KERNELS),
            "receptive_field_steps": conv_receptive_field(DEFAULT_KERNELS),
            "receptive_field_seconds": receptive_field_seconds(DEFAULT_KERNELS, fps=FPS),
        },
        "configs": rows,
        "decision": decision,
    }
    dump_json(out_dir / "comparison.json", payload)
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    figure_ba(rows, out_dir / "figure_ba_comparison")
    return payload


def main() -> None:
    print("POSE CNN RECEPTIVE-FIELD ABLATION")
    print(
        f"Input: {WINDOW_FRAMES} frames at {FPS:.0f} fps "
        f"({WINDOW_FRAMES / FPS:.1f} s). Not 128 resampled steps."
    )
    print("Feature set C, 6 channels. No return-ratio. No scalar branch.")
    print("TEST will not be read. Fusion search / TEST return-ratio rule untouched.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = assert_ablation_out_dir(args.out_dir)
    comparison_path = out_dir / "comparison.json"
    if comparison_path.exists():
        raise SystemExit(f"STOP: {comparison_path} exists.")
    for cfg in CONFIGS:
        metrics_path = out_dir / cfg["name"] / "metrics_dev.json"
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
    refuse_test_ids(frame["sample_id"])
    features = window_tensor(frame, return_ratio_channel=False)
    if features.shape != (len(frame), WINDOW_FRAMES, 6):
        raise SystemExit(f"STOP: expected (N, {WINDOW_FRAMES}, 6), got {features.shape}")
    labels = frame["label"].to_numpy(dtype=np.int64)
    sample_ids = frame["sample_id"].to_numpy()
    window_ids = frame["window_id"].astype(str).to_numpy()
    refuse_test_ids(sample_ids)

    rows = []
    for cfg in CONFIGS:
        info = describe_config(cfg)
        print(
            f"{info['name']} kernels {','.join(str(k) for k in info['kernels'])}  "
            f"RF {info['rf_steps']} steps = {info['rf_seconds']:.2f} s at {FPS:.0f} fps  "
            f"(padding {','.join(str(p) for p in info['paddings'])})",
            flush=True,
        )
        payload = run_loco(
            features=features,
            labels=labels,
            sample_ids=sample_ids,
            window_ids=window_ids,
            kernels=tuple(cfg["kernels"]),
            out_dir=out_dir / cfg["name"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            nn=nn,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
        )
        rows.append(comparison_row(cfg, payload))

    summary = write_comparison(out_dir, rows)
    print("=====================================")
    for row in rows:
        print(
            f"{row['name']}  BA {row['balanced_accuracy']:.3f}  "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  "
            f"RF {row['rf_steps']} steps / {row['rf_seconds']:.2f} s"
        )
    print(summary["decision"]["text"])
    print("TEST not loaded.")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
