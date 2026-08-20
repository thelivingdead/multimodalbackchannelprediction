#!/usr/bin/env python3
"""Train the pseudo-labelled 1D CNN on EMOCA pose features (standalone entry).

This is the same training routine that ``scripts/run_full_experiment.py`` runs
end-to-end; the implementation lives in ``src/pose_cnn.py`` and the rule
amplitude used for pseudo-labelling is imported from
``scripts/run_full_experiment.py`` so there is a single source of truth.

Protocol (unchanged from the recorded experiment):
  1. Pseudo-labels: frozen DEV-tuned rule (results/rule_selected_config.json)
     applied to the 80 committed pseudo pose clips.
  2. Training: 1D CNN on feature set C (Euler xyz + first differences),
     feature normalisation computed on the pseudo TRAIN clips only.
  3. Selection: best epoch and decision threshold by DEV F1 only.
  4. TEST is scored once with the best-on-DEV checkpoint; ablation feature
     sets A-D are then trained with the same protocol.

Requires the committed features (features/gold, features/pseudo) and the
frozen rule config. If features are missing, run scripts/run_full_experiment.py
first (it streams EMOCA without saving the archive).

For run-to-run deterministic CPU training, pin threads:
  OMP_NUM_THREADS=1 python scripts/train_pose_cnn.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_experiment import load_gold, rule_score  # noqa: E402  (single source for the rule)

from src.pose_cnn import train_pseudo_cnn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-csv", type=Path, default=ROOT / "data" / "gold_annotations.csv")
    ap.add_argument("--workdir", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    work = args.workdir.resolve()
    cfg = work / "results" / "rule_selected_config.json"
    if not cfg.exists():
        raise SystemExit(
            "Frozen rule config missing: results/rule_selected_config.json. "
            "Run scripts/run_full_experiment.py first so the rule is tuned on DEV and frozen."
        )
    n_gold = len(list((work / "features" / "gold").glob("*.npz")))
    n_pseudo = len(list((work / "features" / "pseudo").glob("*.npz")))
    if n_gold < 30 or n_pseudo < 8:
        raise SystemExit(
            f"Missing pose features (gold={n_gold}/30, pseudo={n_pseudo}). "
            "Run scripts/run_full_experiment.py first to extract them."
        )
    gold = load_gold(args.gold_csv)
    out = train_pseudo_cnn(gold, work, epochs=args.epochs, seed=args.seed, smoke=args.smoke_test, rule_score_fn=rule_score)
    if out is None:
        raise SystemExit("Training did not run (see messages above).")
    tm = out["test_metrics"]
    print("=====================================")
    print("1D CNN (feature set C = xyz + first differences)")
    print(f"  best epoch (DEV): {out['best_epoch']}   DEV F1: {out['dev_f1']:.3f}")
    print(f"  TEST P {tm['precision']:.3f}  R {tm['recall']:.3f}  F1 {tm['f1']:.3f}  "
          f"(TP{tm['tp']} FP{tm['fp']} TN{tm['tn']} FN{tm['fn']})")
    print("  artifacts: results/classifier_test_metrics.json, "
          "results/classifier_test_predictions.csv, results/ablation_results.csv")
    print("  note: the locked dissertation artifacts record the single official TEST scoring;")


if __name__ == "__main__":
    main()
