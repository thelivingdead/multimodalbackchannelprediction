#!/usr/bin/env python3
"""Head-shake VideoMAE partial fine-tune with balanced TRAIN (new protocol).

Scientific test of the 75 pos / 5 neg pseudo-label collapse. Gold DEV/TEST
still use ``shake_label``. TRAIN is still ``results/shake/pseudo_labels.csv``
(frozen-rule 0/1, not gold), then subsampled to 1:1 (default) or oversampled.

Out-dir (locked originals are never written)::

    results/shake/videomae_finetuned_balanced/

Refuses if that ``metrics.json`` already exists. Never ``--force`` the
locked ``results/shake/videomae_finetuned/`` run (nod 0.82 and shake 0.60).

Otter95 (RTX A4000, ``/scratch`` CUDA venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/finetune_videomae_shake_balanced.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
SHAKE_OUT = ROOT / "results" / "shake" / "videomae_finetuned_balanced"
LOCKED = ROOT / "results" / "shake" / "videomae_finetuned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow re-scoring TEST under the NEW balanced out-dir only",
    )
    parser.add_argument(
        "--balance-train",
        choices=("subsample", "oversample"),
        default="subsample",
        help="subsample (default, 1:1) or oversample minority negatives",
    )
    parser.add_argument("--balance-ratio", type=float, default=1.0)
    parser.add_argument("--pos-weight-boost", type=float, default=1.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_split_leakage
    from finetune_videomae import main as finetune

    if SHAKE_OUT.resolve() == LOCKED.resolve():
        raise SystemExit("STOP: refusing locked results/shake/videomae_finetuned/")
    check_split_leakage.assert_unlocked_out_dir(SHAKE_OUT)

    finetune(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=SHAKE_PSEUDO,
        out_dir=SHAKE_OUT,
        force=args.force,
        unfreeze_blocks=args.unfreeze_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        flip=args.flip,
        balance_train=args.balance_train,
        balance_ratio=args.balance_ratio,
        pos_weight_boost=args.pos_weight_boost,
    )


if __name__ == "__main__":
    main()
