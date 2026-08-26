# Multimodal Backchannel Prediction
## Head-nod recognition on RealTalk (plus locked head-shake)

MSc dissertation code and gold annotations for **weakly supervised head-nod recognition** on Columbia RealTalk (Geng et al., 2023), with a second locked experiment on **head shake**.

The primary **task** is detecting a listener **head nod**. Head **shake** uses the **same 30 gold videos** and a separate `shake_label`. The main pose input is official **EMOCA** 3D head rotation **x, y, z** (FLAME face parameters shipped with RealTalk). EMOCA and FLAME were **not trained** in this project: the published tracks were streamed and used as features. **RGB** is the other input (16-frame face crops → VideoMAE). TRAIN uses automatic labels from a frozen pose rule (**not gold**); DEV/TEST are 30 human labels. TEST is scored once per model; DEV is tuning only.

## Headline TEST results (n = 15, scored once)

Citable table: `dissertation-behaviour-recognition/results/tables/main_results.md`.

| method | input | TRAIN | P | R | F1 | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Pose rule (frozen amplitude) | EMOCA/FLAME rotation x, y, z (rule uses x) | — | 0.64 | 0.70 | **0.67** | [0.35, 0.87] |
| Pose 1D CNN (xyz + derivatives) | EMOCA/FLAME rotation x, y, z + derivatives | 80 pseudo | 0.70 | 0.70 | **0.70** | [0.40, 0.89] |
| Frozen VideoMAE head | RGB 16-frame face crops | 80 pseudo | 0.55 | 0.60 | **0.57** | [0.24, 0.75] |
| Fine-tuned VideoMAE (last 4 blocks) | RGB 16-frame face crops | 80 pseudo | 0.75 | 0.90 | **0.82** | [0.60, 0.96] |
| Fine-tuned VideoMAE (scaling) | RGB 16-frame face crops | 200 pseudo | 0.67 | 0.60 | **0.63** | [0.31, 0.84] |

Canonical RGB result is **n = 80, F1 0.82**. n = 200 is a scaling ablation (point estimate fell). All CIs overlap at n = 15 — no significance claims. The EMOCA/FLAME results are the pose rule (**F1 0.67**) and the pose 1D CNN (**F1 0.70**). **Nod headline: RGB fine-tune F1 0.82.**

## Head-shake TEST (n = 15, scored once)

Same 30 videos / 15–15 splits; gold is `shake_label` in `dissertation-behaviour-recognition/data/gold/shake_annotation_sheet.csv`. No shake CIs (none computed).

**Shake headline: pose rule F1 0.70.** RGB did not beat pose. Shake TRAIN is **75/5** frozen-rule pseudo-labels. Fine-tuned VideoMAE **F1 0.60** is below always-predict-shake **F1 0.64**.

| method | input | TRAIN | P | R | F1 |
| --- | --- | --- | --- | --- | --- |
| Pose rule (axis z, τ≈11.15°) | EMOCA/FLAME rotation x, y, z (rule uses z) | — | 0.54 | 1.00 | **0.70** |
| Pose 1D CNN (xyz + derivatives) | EMOCA/FLAME rotation x, y, z + derivatives | 80 (75/5) | 0.47 | 1.00 | **0.64** |
| Always-predict-shake | — | — | 0.47 | 1.00 | **0.64** |
| Frozen VideoMAE head | RGB 16-frame face crops | 80 (75/5) | 0.46 | 0.86 | **0.60** |
| Frozen VideoMAE head (balanced TRAIN) | RGB 16-frame face crops | 10 | 0.47 | 1.00 | **0.64** |
| Fine-tuned VideoMAE (last 4 blocks) | RGB 16-frame face crops | 80 (75/5) | 0.46 | 0.86 | **0.60** |
| Fine-tuned VideoMAE (balanced TRAIN) | RGB 16-frame face crops | 10 | 0.50 | 0.86 | **0.63** |
| Fine-tuned VideoMAE (DEV-threshold protocol) | RGB 16-frame face crops | 80 (75/5) | 0.60 | 0.43 | **0.50** |
| Late fusion pose + RGB | pose-rule scores + VideoMAE probs | — | 0.54 | 1.00 | **0.70** |

Late fusion matches the pose rule (F1 **0.70**). Joint two-head VideoMAE (TEST once per head): shake F1 **0.64** (always-shake, TP 7 FP 8); nod F1 **0.70** (does not beat dedicated nod **0.82**). Joint frozen-head: nod F1 **0.53** (P 0.56 R 0.50), shake F1 **0.67** (P 0.50 R 1.00).

## Shake DEV-only search (TEST not scored)

GOLD TEST was scored **once** and is **locked**. This search used GOLD DEV only (`test_scored: false`). It is **not** a replacement TEST number. Published shake TEST headline remains **pose rule F1 0.70**. DEV CNN 40/40 is a new protocol result (balanced pseudo-labels).

Source: `dissertation-behaviour-recognition/results/shake/dev_search/comparison_dev.md`. Best config: pose 1D CNN, 40/40, `results/shake/dev_search/cnn_40_40`. DEV n=15 (10 shake+ / 5 shake−). Always-shake on DEV is F1 **0.80**. Frozen VideoMAE 40/40 was conservative (F1 0.462). Fine-tune last-4 40/40 (F1 0.706) did not beat the CNN on DEV.

| method | TRAIN | P | R | F1 | confusion | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| Pose 1D CNN (best DEV) | 40/40 | 0.75 | 0.90 | **0.818** | TP 9, FP 3, TN 2, FN 1 | false |
| Always-predict-shake | — | 0.67 | 1.00 | **0.80** | TP 10, FP 5, TN 0, FN 0 | true |
| Frozen VideoMAE head | 40/40 | 1.00 | 0.30 | **0.462** | TP 3, FP 0, TN 5, FN 7 | false |
| Fine-tuned VideoMAE (last 4 blocks) | 40/40 | 0.86 | 0.60 | **0.706** | TP 6, FP 1, TN 4, FN 4 | false |

After `as_euler('xyz')`: **x** = pitch/nod, **y** = yaw/shake, **z** = roll. Locked TEST rule used **z** (τ≈11.15°); new TRAIN ranks on **y**. Do not retune or rescore locked GOLD TEST. Do not `--force` existing `dev_search/` dirs. Axis audit: `dissertation-behaviour-recognition/results/shake/dev_search/axis_audit.md` and `figures/shake_axis_audit/`.

## Labels

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (the only gold positive) |
| `0` | Unclear / not a nod |
| `shake_label` `1` / `0` | Clear shake / unclear or not a shake (same 30 clips) |

Metric: **clip-level** precision, recall, and F1 (not event IoU 0.30). RealTalk: **p0 = left listener**, **p1 = right listener**, 25 fps.

## Where to look

The submitted package is **`dissertation-behaviour-recognition/`** (README there has the pipeline, scripts, and figure list).

- Gold: `dissertation-behaviour-recognition/data/gold/`
- Locked metrics: `dissertation-behaviour-recognition/results/`
- Tests: `dissertation-behaviour-recognition/tests/`

## Not in this repository

Videos, EMOCA `.pkl` files, RGB `.npz` windows, VideoMAE `best_model.pt` (~345 MB), and virtual environments.

## Other folders (not the submitted experiment)

`api/` and `web/` are a leftover **7-class heuristic demo** (BERT/HuBERT were never used). `scripts/nod_pipeline/` and `notebooks/` are early lab sketches. The submitted study lives only in **`dissertation-behaviour-recognition/`**.

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
