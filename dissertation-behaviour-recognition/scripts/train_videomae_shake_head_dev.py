#!/usr/bin/env python3
"""DEV-only frozen VideoMAE head on a balanced shake manifest (new out-dir).

Never writes locked ``results/shake/videomae_frozen_head/``. Never scores
GOLD TEST. ``score_test=False``.

Otter95 (CPU ok)::

    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_videomae_shake_head_dev.py \\
        --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \\
        --out-dir results/shake/dev_search/vmae_frozen_40_40
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select-dev",
        choices=("f1", "balanced_accuracy"),
        default="balanced_accuracy",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_split_leakage
    from train_videomae_head import main as train_head

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    check_split_leakage.assert_unlocked_out_dir(out)
    train_head(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=args.pseudo_labels if args.pseudo_labels.is_absolute()
        else ROOT / args.pseudo_labels,
        out_dir=out,
        force=args.force,
        score_test=False,
        select_dev=args.select_dev,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
