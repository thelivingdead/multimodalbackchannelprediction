#!/usr/bin/env bash
# Stage B — after you have typed 1/0 labels.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python scripts/05_validate_annotations.py --mode pilot
python scripts/06_create_gold_split.py
python scripts/08_plot_feature_examples.py --behaviour nod --split pilot
python scripts/09_tune_rule.py --behaviour nod --split pilot
python scripts/10_evaluate_rule.py --behaviour nod --split pilot
python scripts/make_figures.py --all
python -m pytest -q tests
echo
echo "Read:"
echo "  results/pilot_nod_rule_metrics.json"
echo "  reports/pilot_nod_findings.md"
echo "  figures/pilot_nod/"
echo "  figures/rule_baseline/"
echo "  figures/annotations/"
df -h .
du -sh . data reports results figures 2>/dev/null || true
