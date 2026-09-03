#!/usr/bin/env python3
"""Leave-one-clip-out VideoMAE cross-validation on the 3 s DEV windows.

Each fold fits weights on 14 DEV clips and predicts the held-out clip once.
DEV only: no TEST input, writes metrics_dev.json. Protocol and metric choices
are documented in reports/methods_chapter_draft.md.

Otter::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_windowed_videomae_dev.py --task nod --max-folds 2
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    load_windows,
    select_dev_threshold,
)

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

DEV_WINDOWS = {
    "nod": ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv",
    "shake": ROOT / "data" / "windowed_annotations" / "shake_windows_dev.csv",
}
RGB_DIR = ROOT / "features" / "rgb16_windowed"
CHECKPOINT = "MCG-NJU/videomae-base"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
MAX_MISSING_FRAC = 0.10
FIXED_THRESHOLD = 0.5


def load_rgb(window_id: str, rgb_dir: Path) -> np.ndarray | None:
    path = rgb_dir / f"{window_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        rgb = z["rgb"]
    if rgb.shape != (16, 224, 224, 3) or rgb.dtype != np.uint8:
        raise SystemExit(f"STOP: {path.name} has rgb {rgb.shape} {rgb.dtype}")
    return rgb


def pack(df: pd.DataFrame, rgb_dir: Path, max_missing_frac: float = MAX_MISSING_FRAC):
    clips, labels, keep = [], [], []
    missing: list[str] = []
    for i, row in enumerate(df.itertuples(index=False)):
        rgb = load_rgb(str(row.window_id), rgb_dir)
        if rgb is None:
            missing.append(str(row.window_id))
            continue
        clips.append(rgb)
        labels.append(int(row.label))
        keep.append(i)
    if missing:
        print(f"NOTE: {len(missing)} DEV windows missing rgb crops")
    if len(df) and len(missing) / len(df) > max_missing_frac:
        raise SystemExit(
            f"STOP: missing {len(missing)}/{len(df)} rgb crops "
            f"(limit {max_missing_frac:.0%}). Finish the crop fetch first."
        )
    return clips, np.asarray(labels, dtype=np.int64), df.iloc[keep].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("nod", "shake"), default="nod")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all 15 folds")
    ap.add_argument("--rgb-dir", type=Path, default=RGB_DIR)
    ap.add_argument(
        "--max-missing-frac",
        type=float,
        default=MAX_MISSING_FRAC,
        help="Allow this fraction of DEV windows to lack an rgb npz.",
    )
    args = ap.parse_args()
    rgb_dir = args.rgb_dir.resolve()
    if not rgb_dir.is_dir():
        raise SystemExit(f"STOP: no crop directory {rgb_dir}")
    try:
        crop_source = str(rgb_dir.relative_to(ROOT))
    except ValueError:
        crop_source = str(rgb_dir)

    out_dir = args.out_dir or (
        ROOT / "results" / f"windowed_{args.task}" / "videomae_loco_dev"
    )
    out_dir = assert_unlocked_out_dir(out_dir)
    metrics_path = out_dir / "metrics_dev.json"
    if metrics_path.exists():
        raise SystemExit(f"STOP: {metrics_path} exists.")

    frame = load_windows(DEV_WINDOWS[args.task], "DEV", DEV_IDS)
    if set(frame["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id present in a DEV-only script")
    clips, labels, frame = pack(frame, rgb_dir, max_missing_frac=args.max_missing_frac)
    sample_ids = frame["sample_id"].to_numpy()
    fold_ids = sorted(set(sample_ids))
    if args.max_folds:
        fold_ids = fold_ids[: args.max_folds]

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
    except ImportError as exc:
        raise SystemExit(
            "STOP: torch/transformers missing. On otter use "
            "/scratch/db01550/venv/bin/python"
        ) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    mean_t = torch.tensor(processor.image_mean, dtype=torch.float32)
    std_t = torch.tensor(processor.image_std, dtype=torch.float32)
    print(
        f"task {args.task} | device {device} | transformers {transformers.__version__}"
    )
    print(
        f"DEV {len(labels)} windows ({int((labels == 1).sum())} positive) "
        f"from {len(set(sample_ids))} clips; running {len(fold_ids)} folds"
    )

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
        layers = model.videomae.encoder.layer
        for p in model.parameters():
            p.requires_grad_(False)
        heads = [model.classifier] + (
            [model.fc_norm] if model.fc_norm is not None else []
        )
        for module in heads:
            for p in module.parameters():
                p.requires_grad_(True)
        for layer in layers[-4:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        opt = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for layer in layers[-4:] for p in layer.parameters()
                    ],
                    "lr": 1e-5,
                },
                {
                    "params": [p for m in heads for p in m.parameters()],
                    "lr": 1e-4,
                },
            ]
        )
        return model, opt

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

    use_amp = device.type == "cuda"
    oof = np.full(len(labels), np.nan, dtype=float)
    fold_rows: list[dict] = []
    started = time.time()

    for position, held_out in enumerate(fold_ids, start=1):
        test_mask = sample_ids == held_out
        train_mask = ~test_mask
        train_items = [c for c, keep in zip(clips, train_mask) if keep]
        train_y = labels[train_mask]
        held_items = [c for c, keep in zip(clips, test_mask) if keep]
        held_y = labels[test_mask]

        model, opt = build_model()
        pos = max(int((train_y == 1).sum()), 1)
        neg = max(int((train_y == 0).sum()), 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32).to(device)
        )
        loader = DataLoader(
            WinDS(train_items, train_y, flip=True),
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
                        loss = criterion(
                            model(pixel_values=xb).logits.squeeze(-1), yb
                        )
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = criterion(model(pixel_values=xb).logits.squeeze(-1), yb)
                    loss.backward()
                    opt.step()
                losses.append(float(loss.item()))
            print(
                f"fold {position}/{len(fold_ids)} {held_out} "
                f"epoch {epoch} loss={np.mean(losses):.4f}",
                flush=True,
            )

        probabilities = predict(model, held_items)
        oof[test_mask] = probabilities
        fold_rows.append(
            {
                "fold": position,
                "held_out_clip": held_out,
                "n_train_windows": int(train_mask.sum()),
                "n_train_positive": int((train_y == 1).sum()),
                "n_held_windows": int(test_mask.sum()),
                "n_held_positive": int((held_y == 1).sum()),
                "held_mean_probability": float(np.mean(probabilities)),
                "elapsed_sec": float(time.time() - started),
            }
        )
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = ~np.isnan(oof)
    y_scored = labels[scored]
    p_scored = oof[scored]
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
            "protocol": f"windowed_{args.task}_3s_videomae_loco",
            "crop_source": crop_source,
            "development_only": True,
            "test_scored": False,
            "cross_validation": "leave-one-clip-out over DEV clips",
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
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
        },
    )

    print("=====================================")
    print(f"windowed {args.task} VideoMAE leave-one-clip-out — DEV only")
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
