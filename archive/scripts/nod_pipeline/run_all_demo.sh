#!/usr/bin/env bash
# Prove the full nod pipeline on synthetic 10×1min clips (no RealTalk download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python scripts/nod_pipeline/01_make_tiny_subset.py --mode demo
python scripts/nod_pipeline/02_extract_pose.py
python scripts/nod_pipeline/03_detect_nod_candidates.py
python scripts/nod_pipeline/04_label_ground_truth.py --demo-fill --validate
python scripts/nod_pipeline/05_baseline_experiment.py --model all
python scripts/nod_pipeline/06_visual_demo.py
echo
echo "Pitch plots:  data/tiny_subset/*/pitch_p0.png"
echo "Metrics:      outputs/nod_pipeline/metrics.json"
echo "Demo figure:  outputs/nod_pipeline/demo_*.png"
