# Shake DEV-only jobs on otter95. Do NOT score GOLD TEST.
# Do NOT --force locked results/shake/{cnn,videomae_*}/ dirs.

cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
PY=/scratch/db01550/venv/bin/python

$PY scripts/audit_shake_axis_dev.py
$PY scripts/build_shake_balanced_pseudo.py

$PY scripts/train_shake_cnn_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \
    --out-dir results/shake/dev_search/cnn_40_40_seq64 --seq-len 64

$PY scripts/train_shake_cnn_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \
    --out-dir results/shake/dev_search/cnn_40_40
$PY scripts/train_videomae_shake_head_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \
    --out-dir results/shake/dev_search/vmae_frozen_40_40
$PY scripts/finetune_videomae_shake_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_40_40.csv \
    --out-dir results/shake/dev_search/vmae_ft4_40_40 --unfreeze-blocks 4

$PY scripts/train_shake_cnn_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_20_20_highconf.csv \
    --out-dir results/shake/dev_search/cnn_20_20_highconf
$PY scripts/train_videomae_shake_head_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_20_20_highconf.csv \
    --out-dir results/shake/dev_search/vmae_frozen_20_20_highconf
$PY scripts/finetune_videomae_shake_dev.py --pseudo-labels results/shake/pseudo_balanced/manifest_20_20_highconf.csv \
    --out-dir results/shake/dev_search/vmae_ft4_20_20_highconf --unfreeze-blocks 4

$PY scripts/compare_shake_dev_search.py
echo 'DEV search done. Do not score GOLD TEST.'
