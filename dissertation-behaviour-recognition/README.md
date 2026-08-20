# Weakly supervised head-nod recognition

MSc dissertation package for **head-nod recognition** on Columbia RealTalk (Geng et al., 2023).

The submitted pipeline is **pose-only**: a small human gold set tunes a frozen EMOCA-pose nod rule, which then generates pseudo-labels for a pose-based temporal classifier (a 1D CNN over EMOCA `rotation_xyz` sequences — not an RGB vision model). A first frozen VideoMAE-head run exists as a follow-up check (see "Visual experiment" below); it scores below the pose models and is not a headline. Synthetic pilot clips are not reported as RealTalk results.

## Labels

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (only this class is a gold positive) |
| `0` | Unclear / not a nod |

Metric: **clip-level** precision, recall, and F1 on the 30 binary windows (one 0/1 decision per ~60 s window). An event-level F1 at temporal IoU 0.30 is implemented in `src/metrics.py` but is **not** the protocol of this study; do not headline it, and do not headline accuracy.

RealTalk convention: **p0 = LEFT**, **p1 = RIGHT**, 25 fps.

## Gold split

30 RealTalk videos, one ~1-minute watch window each:

- **15 DEV** — used to tune the rule and to pick the CNN epoch/threshold
- **15 TEST** — labelled, scored once, never used for tuning or training

Times and labels are in `data/gold/annotation_sheet.csv` (YouTube clock). The same events in seconds are in `data/gold/events.csv`. The split lists are `data/splits/gold_dev.txt` and `data/splits/gold_test.txt`.

```bash
python scripts/import_annotation_sheet.py
python scripts/export_predicted_vs_annotated.py
```

Predicted columns stay empty until matching EMOCA pose exists for that `video_id`.

## Layout

```
configs/          rule parameters and planned-experiment notes (see configs/rule_nod.yaml header)
data/gold/        human labels (annotation sheet, events, watch list, annotation log)
data/splits/      DEV / TEST video ids
scripts/          pipeline: 00-10 audit + rule stages, run_full_experiment.py (main),
                  train_pose_cnn.py (CNN stage only), 19-21 error analysis / efficiency / figures
src/              events, metrics, rules, loaders, pose_cnn.py (1D CNN training)
results/          every saved metric and table (see results/README.md)
figures/          dissertation figures (see figures/README.md)
reports/          audits, chapter drafts, dissertation evidence notes
tests/            invariants (no TEST leakage)
```

Large binaries stay off git: `.mp4`, `.avi`, `.pkl`, `.venv`, Hugging Face tars, `emoca.tar.gz`.

## Pipeline (executed)

Entry point: `python scripts/run_full_experiment.py` (end to end), or
`python scripts/train_pose_cnn.py` for the CNN stage alone.

1. Gold annotation — 30 human-labelled windows (done)
2. EMOCA / FLAME pose for those windows — **streamed** from Hugging Face, never saved as `emoca.tar.gz`; EMOCA was not trained (done)
3. Nod rule tuned on DEV only — Savitzky–Golay amplitude rule, axis x, threshold 16.35° (done, frozen in `results/rule_selected_config.json`)
4. TEST scored once — rule: **F1 0.67** (P 0.64, R 0.70; TP 7, FP 4, TN 1, FN 3)
5. Pseudo-labels from the frozen rule — 80 TRAIN clips, 70 nod / 10 unclear (done)
6. Pose 1D CNN (`src/pose_cnn.py`) on pseudo-labels, epoch and probability threshold chosen on DEV — TEST scored once: **F1 0.70** (P 0.70, R 0.70; TP 7, FP 3, TN 2, FN 3)

Feature ablations A–C (x-only / xyz / xyz + derivatives) all give TEST F1 0.70 at two decimals; ablation D (expression) diverged (`loss = nan`) and is omitted. DEV F1 (0.86 rule, 0.89 CNN) are tuning numbers, not results. Locked scores: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`; the citable table is `results/final_results.csv`.

Do not train or tune on GOLD TEST.

## Visual experiment: VideoMAE

The planned visual arm is a **frozen CPU VideoMAE encoder with a small trained head** (no fine-tuning). A first frozen-head run landed on 20 August 2026 (`scripts/extract_videomae_embeddings.py` + `scripts/train_videomae_head.py`): 768-d embeddings for all 110 clips, an MLP head trained on the same 80 rule pseudo-labels with epoch/threshold chosen on DEV. Artifacts: `results/videomae_frozen_head/`, `figures/videomae_training_curve.png`.

Outcome: DEV F1 0.90 (tuning split) but **TEST F1 0.57** (P 0.55, R 0.60; TP 6, FP 5, TN 0, FN 4) — below both pose models on the same held-out TEST set, so the submitted headline story remains pose-only (0.67 / 0.70). VideoMAE fine-tuning and the pose+VideoMAE fusion were not run (lab storage constraint; `reports/videomae_preflight_lab.md`).

## Audits and validation

- `reports/annotation_audit.md` — two annotation outliers inspected (no ground truth changed)
- `reports/split_integrity.md` — DEV/TEST leakage audit (none found)
- `reports/repository_validation.md` — pytest output and reproduction of the 0.67 / 0.70 TEST scores

## Setup

```bash
cd dissertation-behaviour-recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
