#!/usr/bin/env python3
"""Head-shake frozen VideoMAE MLP head with balanced TRAIN (new protocol).

Same gold / pseudo as the locked 75/5 run; TRAIN is subsampled to 1:1
(default) or oversampled. Writes only::

    results/shake/videomae_frozen_head_balanced/

Refuses if that ``metrics.json`` exists. Does not touch
``results/shake/videomae_frozen_head/``.

Otter95 (CPU is fine; ``/scratch`` venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_videomae_shake_head_balanced.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
SHAKE_OUT = ROOT / "results" / "shake" / "videomae_frozen_head_balanced"
LOCKED = ROOT / "results" / "shake" / "videomae_frozen_head"


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
    )
    parser.add_argument("--balance-ratio", type=float, default=1.0)
    parser.add_argument("--pos-weight-boost", type=float, default=1.0)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_split_leakage
    from train_videomae_head import main as train_head

    if SHAKE_OUT.resolve() == LOCKED.resolve():
        raise SystemExit(
            "STOP: refusing locked results/shake/videomae_frozen_head/"
        )
    check_split_leakage.assert_unlocked_out_dir(SHAKE_OUT)

    train_head(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=SHAKE_PSEUDO,
        out_dir=SHAKE_OUT,
        force=args.force,
        balance_train=args.balance_train,
        balance_ratio=args.balance_ratio,
        pos_weight_boost=args.pos_weight_boost,
    )


if __name__ == "__main__":
    main()
