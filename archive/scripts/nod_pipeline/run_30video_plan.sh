#!/usr/bin/env bash
# 30-video gold (15 dev / 15 test) + unlabeled train → rule baseline → pseudo-train → test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

START=$(date +%s)
echo "=== START $(date) ==="

python scripts/nod_pipeline/08_build_30_video_experiment.py --n-gold 30 --n-train 50
python scripts/nod_pipeline/02_extract_pose.py --subset data/nod30
python scripts/nod_pipeline/03_detect_nod_candidates.py --subset data/nod30
python scripts/nod_pipeline/09_analyse_splits.py --subset data/nod30
python scripts/nod_pipeline/10_rule_baseline.py
python scripts/nod_pipeline/11_pseudo_label.py
python scripts/nod_pipeline/12_train_pseudo_classifier.py
python scripts/nod_pipeline/13_compare_and_figures.py

END=$(date +%s)
echo "=== TOTAL WALL CLOCK $((END-START)) seconds ===" | tee -a outputs/nod_pipeline/time_log.txt
echo "Gold labels:     outputs/nod_pipeline/gold_labels.csv"
echo "Splits:          outputs/nod_pipeline/splits.json"
echo "Rule baseline:   outputs/nod_pipeline/rule_baseline_metrics.csv"
echo "HP search:       outputs/nod_pipeline/hyperparam_search.csv"
echo "Learned test:    outputs/nod_pipeline/learned_test_metrics.json"
echo "Comparison:      outputs/nod_pipeline/baseline_vs_learned.csv"
du -sh data/nod30 outputs/nod_pipeline 2>/dev/null || true
df -h .
