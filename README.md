# Head-nod recognition on RealTalk

MSc dissertation code and gold annotations for **weakly supervised head-nod recognition** on Columbia RealTalk (Geng et al., 2023).

The **task** is detecting a listener **head nod**. **Pose** is one input (EMOCA 3D head rotation). **RGB** is the other (16-frame face crops → VideoMAE). TRAIN uses automatic labels from a frozen pose rule; DEV/TEST are 30 human labels. TEST is scored once per model.

## Headline TEST results (n = 15, scored once)

Citable table: `dissertation-behaviour-recognition/results/tables/main_results.md`.

| method | TRAIN | F1 | 95% CI |
| --- | --- | --- | --- |
| Pose rule (frozen amplitude) | — | **0.67** | [0.35, 0.87] |
| Pose 1D CNN (xyz + derivatives) | 80 pseudo | **0.70** | [0.40, 0.89] |
| Frozen VideoMAE head | 80 pseudo | **0.57** | [0.24, 0.75] |
| Fine-tuned VideoMAE (last 4 blocks) | 80 pseudo | **0.82** | [0.60, 0.96] |
| Fine-tuned VideoMAE (scaling) | 200 pseudo | **0.63** | [0.31, 0.84] |

Canonical RGB result is **n = 80, F1 0.82**. n = 200 is a scaling ablation (point estimate fell). All CIs overlap at n = 15 — no significance claims.

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
- Chapter drafts: `dissertation-behaviour-recognition/reports/`
- Tests: `dissertation-behaviour-recognition/tests/`

## Not in this repository

Videos, EMOCA `.pkl` files, RGB `.npz` windows, VideoMAE `best_model.pt` (~345 MB), and virtual environments.

## Other folders (not the submitted experiment)

`scripts/nod_pipeline/`, `api/`, `web/` are earlier prototypes. Proposal-era notes: `docs/archive/`. Do not cite those as TEST scores.

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
