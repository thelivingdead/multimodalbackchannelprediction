# Multimodal Backchannel Prediction
## Head-nod recognition on RealTalk

MSc dissertation code and gold annotations for **weakly supervised head-nod recognition** on Columbia RealTalk (Geng et al., 2023).

The **task** is detecting a listener **head nod**. The main pose input is official **EMOCA** 3D head rotation **x, y, z** (FLAME face parameters shipped with RealTalk). EMOCA and FLAME were **not trained** in this project: the published tracks were streamed and used as features. **RGB** is the other input (16-frame face crops → VideoMAE). TRAIN uses automatic labels from a frozen pose rule; DEV/TEST are 30 human labels. TEST is scored once per model.

## Headline TEST results (n = 15, scored once)

Citable table: `dissertation-behaviour-recognition/results/tables/main_results.md`.

| method | input | TRAIN | P | R | F1 | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Pose rule (frozen amplitude) | EMOCA/FLAME rotation x, y, z (rule uses x) | — | 0.64 | 0.70 | **0.67** | [0.35, 0.87] |
| Pose 1D CNN (xyz + derivatives) | EMOCA/FLAME rotation x, y, z + derivatives | 80 pseudo | 0.70 | 0.70 | **0.70** | [0.40, 0.89] |
| Frozen VideoMAE head | RGB 16-frame face crops | 80 pseudo | 0.55 | 0.60 | **0.57** | [0.24, 0.75] |
| Fine-tuned VideoMAE (last 4 blocks) | RGB 16-frame face crops | 80 pseudo | 0.75 | 0.90 | **0.82** | [0.60, 0.96] |
| Fine-tuned VideoMAE (scaling) | RGB 16-frame face crops | 200 pseudo | 0.67 | 0.60 | **0.63** | [0.31, 0.84] |

Canonical RGB result is **n = 80, F1 0.82**. n = 200 is a scaling ablation (point estimate fell). All CIs overlap at n = 15 — no significance claims. The EMOCA/FLAME results are the pose rule (**F1 0.67**) and the pose 1D CNN (**F1 0.70**).

## Labels

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (the only gold positive) |
| `0` | Unclear / not a nod |

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
