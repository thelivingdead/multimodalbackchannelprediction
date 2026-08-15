# Weakly supervised head-nod recognition

MSc dissertation package for **head-nod recognition** on Columbia RealTalk (Geng et al., 2023).

A small human gold set is used to tune a pose-based nod rule, then to generate pseudo-labels for a pose classifier. VideoMAE is reserved until real rule metrics exist. Synthetic pilot clips are not reported as RealTalk results.

## Labels

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (only this class is a gold positive) |
| `0` | Unclear / not a nod |

Primary event metric: **F1 at temporal IoU 0.30** (one-to-one matching). Do not headline accuracy.

RealTalk convention: **p0 = LEFT**, **p1 = RIGHT**, 25 fps.

## Gold split

30 RealTalk videos, one ~1-minute watch window each:

- **15 DEV** — may be used to tune the rule
- **15 TEST** — labelled, scored once, never used for tuning or training

Times and labels are in `data/gold/annotation_sheet.csv` (YouTube clock). The same events in seconds are in `data/gold/events.csv`. The split lists are `data/splits/gold_dev.txt` and `data/splits/gold_test.txt`.

```bash
python scripts/import_annotation_sheet.py
python scripts/export_predicted_vs_annotated.py
```

Predicted columns stay empty until matching EMOCA pose exists for that `video_id`.

## Layout

```
configs/          nod rule and later model settings
data/gold/        human labels (sheet, events, watch list)
data/splits/      DEV / TEST video ids
scripts/          numbered pipeline + annotation import/export
src/              events, metrics, rules, loaders
results/          predicted vs annotated table
reports/          dissertation evidence notes
tests/            invariants (no TEST leakage)
```

Large binaries stay off git: `.mp4`, `.avi`, `.pkl`, `.venv`, Hugging Face tars, `emoca.tar.gz`.

## Pipeline (intended)

1. Gold annotation (done for the 30-video sheet)
2. EMOCA / FLAME pose for those videos
3. Tune the nod rule on DEV only
4. Score TEST once
5. Pseudo-labels from the frozen rule
6. Pose classifier, then VideoMAE if rule metrics exist

Do not train or tune on GOLD TEST.

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
