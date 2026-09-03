#!/usr/bin/env python3
"""DEV leave-one-clip-out VideoMAE on identity-fixed 1.5 s nod windows.

Does not read TEST. Does not write into the 3 s identity-fixed directory.

    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/crossval_videomae_identity_1p5s.py \\
        --rgb-dir /scratch/db01550/rgb16_windowed_identity_dev_1p5s \\
        --hflip
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
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.clip_metrics import always_predict, choose_dev_threshold, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402
from src.windowed_baselines import average_precision, clip_bootstrap  # noqa: E402

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev_1p5s.csv"
OUT_ROOT = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed_1p5s"
BLOCKED_3S = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
CHECKPOINT = "MCG-NJU/videomae-base"
FIXED_THRESHOLD = 0.5
MAX_MISSING_FRAC = 0.25


def require_audit(out_root: Path) -> dict:
    path = out_root / "audit_pass.json"
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}. Run the 1.5 s identity audit first.")
    report = json_load(path)
    if not report.get("pass") and not report.get("audit_pass"):
        raise SystemExit("STOP: 1.5 s identity audit did not pass")
    if report.get("n_resolved_on_wrong_half", 1) != 0:
        raise SystemExit("STOP: audit still reports wrong-half crops")
    return report


def json_load(path: Path) -> dict:
    import json
    return json.loads(path.read_text())


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


def load_dev_windows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["split"] = frame["split"].astype(str).str.upper()
    if (frame["split"] != "DEV").any():
        raise SystemExit(f"STOP: {path.name} contains a non-DEV row")
    if set(frame["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id in 1.5 s windows")
    if set(frame["sample_id"]) != DEV_IDS:
        raise SystemExit("STOP: 1.5 s windows are not gold_001 to gold_015")
    return frame.reset_index(drop=True)


def write_bundle(
    out_dir: Path,
    *,
    protocol: str,
    audit: dict,
    rgb_dir: Path,
    y: np.ndarray,
    p: np.ndarray,
    pred: np.ndarray,
    ids: np.ndarray,
    window_ids: np.ndarray,
    n_dev: int,
    n_missing: int,
    epochs: int,
    seed: int,
    hflip: bool,
    threshold: float,
    threshold_note: str,
    fold_rows: list,
    extra: dict,
    device: str,
) -> dict:
    metrics = clip_binary_metrics(y, pred)
    boot = clip_bootstrap(ids, y, pred)
    pr_auc = average_precision(y, p)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "window_id": window_ids,
            "sample_id": ids,
            "label": y,
            "oof_probability": p,
            "pred": pred,
            "threshold": threshold,
        }
    ).to_csv(out_dir / "oof_predictions.csv", index=False)
    payload = {
        "protocol": protocol,
        "development_only": True,
        "test_scored": False,
        "test_touched": False,
        "window_sec": 1.5,
        "stride_sec": 1.0,
        "n_rgb_frames": 16,
        "effective_fps": 16 / 1.5,
        "crop_source": str(rgb_dir),
        "audit": audit,
        "hflip_train_only": hflip,
        "threshold": float(threshold),
        "threshold_note": threshold_note,
        "n_dev_windows": int(n_dev),
        "n_excluded_unresolved": int(n_missing),
        "n_windows_scored": int(len(y)),
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
        "epochs_per_fold": epochs,
        "seed": seed,
        "device": device,
        **extra,
    }
    dump_json(out_dir / "metrics.json", payload)
    interval = boot["balanced_accuracy"]
    print(
        f"{protocol}  BA {metrics['balanced_accuracy']:.3f} "
        f"[{interval['ci_lower_95']:.3f}, {interval['ci_upper_95']:.3f}]  "
        f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
        f"F1 {metrics['f1']:.3f} PR AUC {pr_auc:.3f}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, default=WINDOWS)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hflip", action="store_true", default=True)
    parser.add_argument("--no-hflip", dest="hflip", action="store_false")
    parser.add_argument("--write-train-threshold", action="store_true")
    parser.add_argument("--run-name", type=str, default="last_blocks_unfrozen")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    if out_root == BLOCKED_3S.resolve() or BLOCKED_3S.resolve() in out_root.parents:
        raise SystemExit(
            "STOP: refusing to write 1.5 s results into the 3 s identity-fixed directory"
        )
    audit = require_audit(out_root)
    rgb_dir = args.rgb_dir.resolve()
    out_dir = assert_unlocked_out_dir(out_root / args.run_name)
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"STOP: {out_dir / 'metrics.json'} exists")

    frame = load_dev_windows(args.windows)
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
        trainable = list(layers[-args.unfreeze_blocks:])
        for layer in trainable:
            for param in layer.parameters():
                param.requires_grad_(True)
        groups = [
            {"params": [p for layer in trainable for p in layer.parameters()], "lr": 1e-5},
            {"params": [p for m in heads for p in m.parameters()], "lr": 1e-4},
        ]
        return model, torch.optim.AdamW(groups)

    def predict(model, items) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(items), args.batch_size):
                batch = torch.stack(
                    [preprocess(c) for c in items[i:i + args.batch_size]]
                ).to(device)
                logits = model(pixel_values=batch).logits.squeeze(-1)
                out.append(1.0 / (1.0 + np.exp(-logits.float().cpu().numpy())))
        return np.concatenate(out) if out else np.asarray([])

    print(
        f"1.5 s last_blocks | device {device} | hflip={args.hflip} | "
        f"{len(labels)} resolved, {len(missing)} excluded | "
        f"transformers {transformers.__version__}"
    )
    use_amp = device.type == "cuda"
    oof = np.full(len(labels), np.nan, dtype=float)
    oof_thr = np.full(len(labels), np.nan, dtype=float)
    fold_rows = []
    train_thr_rows = []
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
            WinDS([c for c, keep in zip(clips, train_mask) if keep], train_y, args.hflip),
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
        selected_thr = FIXED_THRESHOLD
        if args.write_train_threshold:
            train_items = [c for c, keep in zip(clips, train_mask) if keep]
            train_p = predict(model, train_items)
            selected_thr, _ = choose_dev_threshold(
                train_y, train_p, criterion="balanced_accuracy"
            )
            oof_thr[test_mask] = selected_thr
            train_thr_rows.append(
                {
                    "fold": position,
                    "held_out_clip": held_out,
                    "train_selected_threshold": float(selected_thr),
                    "n_train_windows": int(train_mask.sum()),
                    "n_held_windows": int(test_mask.sum()),
                }
            )
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
                "train_selected_threshold": float(selected_thr),
            }
        )
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = ~np.isnan(oof)
    y = labels[scored]
    p = oof[scored]
    ids = sample_ids[scored]
    window_ids = frame.loc[scored, "window_id"].astype(str).to_numpy()
    pred = (p >= FIXED_THRESHOLD).astype(int)
    write_bundle(
        out_dir,
        protocol="windowed_nod_1p5s_videomae_identity_fixed_last_blocks",
        audit=audit,
        rgb_dir=rgb_dir,
        y=y,
        p=p,
        pred=pred,
        ids=ids,
        window_ids=window_ids,
        n_dev=len(pd.read_csv(args.windows)),
        n_missing=len(missing),
        epochs=args.epochs,
        seed=args.seed,
        hflip=args.hflip,
        threshold=FIXED_THRESHOLD,
        threshold_note="fixed at 0.5 before any fold was scored",
        fold_rows=fold_rows,
        extra={"unfreeze_blocks": args.unfreeze_blocks, "mode": "last_blocks"},
        device=str(device),
    )
    if args.write_train_threshold:
        thr_dir = assert_unlocked_out_dir(out_root / f"{args.run_name}_train_threshold")
        if (thr_dir / "metrics.json").exists():
            raise SystemExit(f"STOP: {thr_dir / 'metrics.json'} exists")
        fold_thr = oof_thr[scored]
        pred_thr = (p >= fold_thr).astype(int)
        pd.DataFrame(train_thr_rows).to_csv(thr_dir / "fold_thresholds.csv", index=False)
        write_bundle(
            thr_dir,
            protocol="windowed_nod_1p5s_videomae_identity_fixed_train_threshold",
            audit=audit,
            rgb_dir=rgb_dir,
            y=y,
            p=p,
            pred=pred_thr,
            ids=ids,
            window_ids=window_ids,
            n_dev=len(pd.read_csv(args.windows)),
            n_missing=len(missing),
            epochs=args.epochs,
            seed=args.seed,
            hflip=args.hflip,
            threshold=float("nan"),
            threshold_note=(
                "Per-fold threshold selected on training clips only, "
                "by balanced accuracy. The held-out clip was not used."
            ),
            fold_rows=fold_rows,
            extra={
                "unfreeze_blocks": args.unfreeze_blocks,
                "mode": "last_blocks",
                "same_weights_as": str(out_dir),
                "per_fold_threshold": True,
            },
            device=str(device),
        )
    print(f"artifacts: {out_root}")
    print("TEST was not loaded.")


if __name__ == "__main__":
    main()
