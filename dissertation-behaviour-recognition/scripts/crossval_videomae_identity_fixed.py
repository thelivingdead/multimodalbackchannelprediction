#!/usr/bin/env python3
"""DEV-only leave-one-clip-out VideoMAE on identity-fixed 3 s crops.

Refuses to start unless audit_pass.json exists. Never loads TEST.
Does not overwrite results/windowed_nod/videomae_*.

    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_videomae_identity_fixed.py --mode frozen
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_videomae_identity_fixed.py --mode last_blocks
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402
from src.windowed_baselines import average_precision, clip_bootstrap, load_windows  # noqa: E402

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

FIXED = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
CHECKPOINT = "MCG-NJU/videomae-base"
FIXED_THRESHOLD = 0.5
MAX_MISSING_FRAC = 0.25


def require_audit(out_root: Path) -> dict:
    path = out_root / "audit_pass.json"
    if not path.exists():
        raise SystemExit(
            f"STOP: {path} is missing. Run scripts/audit_target_person_crops.py "
            "and inspect crop_audit/ first. VideoMAE must not start."
        )
    report = json.loads(path.read_text())
    if not report.get("pass"):
        raise SystemExit("STOP: audit_pass.json exists but pass is not true")
    if report.get("n_resolved_on_wrong_half", 1) != 0:
        raise SystemExit("STOP: audit still reports wrong-half crops")
    return report


def load_rgb(window_id: str, rgb_dir: Path) -> np.ndarray | None:
    path = rgb_dir / f"{window_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        rgb = payload["rgb"]
        status = str(payload["crop_status"]) if "crop_status" in payload else ""
    if status and status != "resolved":
        return None
    if rgb.shape != (16, 224, 224, 3) or rgb.dtype != np.uint8:
        raise SystemExit(f"STOP: {path.name} has rgb {rgb.shape} {rgb.dtype}")
    return rgb


def confusion_png(metrics: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title("Out of fold confusion, threshold 0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("frozen", "last_blocks"), required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=FIXED)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    audit = require_audit(out_root)
    rgb_dir = args.rgb_dir.resolve()
    name = "frozen_encoder" if args.mode == "frozen" else "last_blocks_unfrozen"
    out_dir = assert_unlocked_out_dir(out_root / name)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(f"STOP: {metrics_path} exists. This run is already written.")

    frame = load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    if set(frame["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id present in a DEV-only script")
    clips, labels, keep, missing = [], [], [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        rgb = load_rgb(str(row.window_id), rgb_dir)
        if rgb is None:
            missing.append(str(row.window_id))
            continue
        clips.append(rgb)
        labels.append(int(row.label))
        keep.append(i)
    if len(frame) and len(missing) / len(frame) > MAX_MISSING_FRAC:
        raise SystemExit(
            f"STOP: {len(missing)}/{len(frame)} windows have no resolved crop"
        )
    frame = frame.iloc[keep].reset_index(drop=True)
    labels = np.asarray(labels, dtype=np.int64)
    sample_ids = frame["sample_id"].to_numpy()
    fold_ids = sorted(set(sample_ids.tolist()))
    if args.max_folds:
        fold_ids = fold_ids[: args.max_folds]

    try:
        import torch
        import transformers
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "STOP: torch/transformers missing. Use /scratch/db01550/venv/bin/python"
        ) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    mean_t = torch.tensor(processor.image_mean, dtype=torch.float32)
    std_t = torch.tensor(processor.image_std, dtype=torch.float32)

    def preprocess(rgb_u8: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).to(torch.float32) / 255.0
        x = (x - mean_t) / std_t
        return x.permute(0, 3, 1, 2).contiguous()

    class WinDS(Dataset):
        def __init__(self, items, ys, flip: bool):
            self.items, self.ys, self.flip = items, ys, flip

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            x = preprocess(self.items[i])
            if self.flip and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            return x, float(self.ys[i])

    def build_model():
        config = VideoMAEConfig.from_pretrained(CHECKPOINT)
        config.num_labels = 1
        model = VideoMAEForVideoClassification.from_pretrained(
            CHECKPOINT, config=config, ignore_mismatched_sizes=True
        )
        model.to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        heads = [model.classifier]
        if model.fc_norm is not None:
            heads.append(model.fc_norm)
        for module in heads:
            for param in module.parameters():
                param.requires_grad_(True)
        layers = model.videomae.encoder.layer
        trainable_layers = []
        if args.mode == "last_blocks":
            trainable_layers = list(layers[-args.unfreeze_blocks :])
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad_(True)
        groups = [
            {
                "params": [p for m in heads for p in m.parameters()],
                "lr": 1e-4,
            }
        ]
        if trainable_layers:
            groups.insert(
                0,
                {
                    "params": [p for layer in trainable_layers for p in layer.parameters()],
                    "lr": 1e-5,
                },
            )
        return model, torch.optim.AdamW(groups)

    def predict(model, items) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(items), args.batch_size):
                batch = torch.stack(
                    [preprocess(c) for c in items[i : i + args.batch_size]]
                ).to(device)
                logits = model(pixel_values=batch).logits.squeeze(-1)
                out.append(1.0 / (1.0 + np.exp(-logits.float().cpu().numpy())))
        return np.concatenate(out) if out else np.asarray([])

    print(
        f"mode {args.mode} | device {device} | transformers {transformers.__version__} | "
        f"{len(labels)} resolved DEV windows, {len(missing)} excluded"
    )
    use_amp = device.type == "cuda"
    oof = np.full(len(labels), np.nan, dtype=float)
    fold_rows = []
    for position, held_out in enumerate(fold_ids, start=1):
        test_mask = sample_ids == held_out
        train_mask = ~test_mask
        train_y = labels[train_mask]
        if train_y.min() == train_y.max():
            raise SystemExit(f"STOP: fold {held_out} train labels are one class")
        model, opt = build_model()
        pos = int((train_y == 1).sum())
        neg = int((train_y == 0).sum())
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device)
        )
        loader = DataLoader(
            WinDS(
                [c for c, keep in zip(clips, train_mask) if keep],
                train_y,
                flip=True,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + position),
        )
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        loss = criterion(model(pixel_values=xb).logits.squeeze(-1), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = criterion(model(pixel_values=xb).logits.squeeze(-1), yb)
                    loss.backward()
                    opt.step()
                losses.append(float(loss.item()))
            print(
                f"fold {position}/{len(fold_ids)} {held_out} epoch {epoch} "
                f"loss={np.mean(losses):.4f} pos_weight={neg / pos:.3f}",
                flush=True,
            )
        held_items = [c for c, keep in zip(clips, test_mask) if keep]
        probabilities = predict(model, held_items)
        oof[test_mask] = probabilities
        fold_rows.append(
            {
                "fold": position,
                "held_out_clip": held_out,
                "n_train_windows": int(train_mask.sum()),
                "n_train_positive": int((train_y == 1).sum()),
                "pos_weight": float(neg / pos),
                "n_held_windows": int(test_mask.sum()),
                "n_held_positive": int(labels[test_mask].sum()),
                "held_mean_probability": float(np.mean(probabilities)),
            }
        )
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = ~np.isnan(oof)
    y = labels[scored]
    p = oof[scored]
    ids = sample_ids[scored]
    pred = (p >= FIXED_THRESHOLD).astype(int)
    metrics = clip_binary_metrics(y, pred)
    boot = clip_bootstrap(ids, y, pred)
    pr_auc = average_precision(y, p)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "window_id": frame.loc[scored, "window_id"].astype(str).to_numpy(),
            "sample_id": ids,
            "label": y,
            "oof_probability": p,
            "pred_at_0.5": pred,
        }
    ).to_csv(out_dir / "oof_predictions.csv", index=False)
    confusion_png(metrics, out_dir / "confusion_matrix.png")
    dump_json(
        metrics_path,
        {
            "protocol": f"windowed_nod_3s_videomae_identity_fixed_{args.mode}",
            "development_only": True,
            "test_scored": False,
            "test_touched": False,
            "crop_source": str(rgb_dir),
            "audit": audit,
            "mode": args.mode,
            "unfreeze_blocks": 0 if args.mode == "frozen" else args.unfreeze_blocks,
            "threshold": FIXED_THRESHOLD,
            "threshold_note": "fixed at 0.5 before any fold was scored",
            "n_dev_windows": 15 * 29,
            "n_excluded_unresolved": int(len(missing)),
            "n_windows_scored": int(scored.sum()),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "prevalence": float(y.mean()),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": pr_auc,
            "confusion": {
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            },
            "clip_bootstrap": boot,
            "always_no": always_predict(y, 0),
            "always_yes": always_predict(y, 1),
            "epochs_per_fold": args.epochs,
            "seed": args.seed,
            "device": str(device),
        },
    )
    interval = boot["balanced_accuracy"]
    print("=====================================")
    print(f"identity-fixed VideoMAE {args.mode}, DEV out of fold, TEST untouched")
    print(
        f"balanced accuracy {metrics['balanced_accuracy']:.3f} "
        f"[{interval['ci_lower_95']:.3f}, {interval['ci_upper_95']:.3f}]  "
        f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
        f"F1 {metrics['f1']:.3f} PR AUC {pr_auc:.3f}"
    )
    print(f"excluded unresolved crops: {len(missing)}")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
