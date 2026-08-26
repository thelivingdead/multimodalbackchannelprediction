#!/usr/bin/env python3
"""DEV-only VideoMAE last-4-block fine-tune on a balanced shake manifest.

Never writes locked ``results/shake/videomae_finetuned/``. Never scores
GOLD TEST. ``score_test=False``. Window length is cached rgb16 (16 frames);
no second window (would need a new fetch).

Otter95 (RTX A4000, no Docker)::

    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/finetune_videomae_shake_dev.py \\
        --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \\
        --out-dir results/shake/dev_search/vmae_ft4_40_40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select-dev",
        choices=("f1", "balanced_accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_split_leakage
    from finetune_videomae import main as finetune

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    check_split_leakage.assert_unlocked_out_dir(out)
    finetune(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=args.pseudo_labels if args.pseudo_labels.is_absolute()
        else ROOT / args.pseudo_labels,
        out_dir=out,
        force=args.force,
        unfreeze_blocks=args.unfreeze_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        flip=args.flip,
        score_test=False,
        select_dev=args.select_dev,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
