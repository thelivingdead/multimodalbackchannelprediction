#!/usr/bin/env python3
"""DEV-only shake 1D CNN on a balanced pseudo manifest (new out-dir).

Never writes ``results/shake/cnn/`` or ``results/shake/pseudo_labels.csv``.
Never scores GOLD TEST. Gold labels: ``shake_annotation_sheet.csv`` /
``shake_label``. Feature set C = Euler xyz + first differences.

::

    OMP_NUM_THREADS=1 python scripts/train_shake_cnn_dev.py \\
        --pseudo-labels results/shake/pseudo_balanced/A_40_40.csv \\
        --out-dir results/shake/dev_balanced/cnn_A_40_40 \\
        --select-dev balanced_accuracy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--select-dev",
        choices=("f1", "balanced_accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument(
        "--score-test",
        action="store_true",
        default=False,
        help="refused. GOLD TEST is locked.",
    )
    args = parser.parse_args()
    if args.score_test:
        raise SystemExit(
            "STOP: DEV-only wrapper refuses --score-test. Do not score GOLD TEST."
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_split_leakage

    check_split_leakage.assert_unlocked_out_dir(
        args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    )
    argv = [
        "--dev-only",
        "--pseudo-labels", str(args.pseudo_labels),
        "--out-dir", str(args.out_dir),
        "--epochs", str(int(args.epochs)),
        "--seed", str(int(args.seed)),
        "--seq-len", str(int(args.seq_len)),
        "--select-dev", str(args.select_dev),
        "--workdir", str(args.workdir),
    ]
    if args.force:
        argv.append("--force")
    from train_shake_cnn import main as train_cnn
    train_cnn(argv)


if __name__ == "__main__":
    main()
