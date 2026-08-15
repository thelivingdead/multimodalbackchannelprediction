# Head-nod recognition on RealTalk

MSc dissertation code and gold annotations for **weakly supervised head-nod recognition** on Columbia RealTalk (Geng et al., 2023).

A small human gold set is used to tune a pose-based nod rule. The frozen rule can then produce pseudo-labels for a pose classifier. VideoMAE is not used until real rule metrics exist.

## Labels

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (the only gold positive) |
| `0` | Unclear / not a nod |

Primary metric: event **F1 at IoU 0.30**. RealTalk: **p0 = left listener**, **p1 = right listener**, 25 fps.

## Current status

- 30 RealTalk videos labelled (15 DEV, 15 TEST)
- Gold times and labels: `dissertation-behaviour-recognition/data/gold/`
- DEV may be used to tune the rule; TEST is scored once and is not used for training

## Main package

```
dissertation-behaviour-recognition/
  configs/     rule and model settings
  data/gold/   annotation sheet, events, watch list
  data/splits/ DEV and TEST video ids
  scripts/     pipeline and annotation import
  src/         events, metrics, nod rule
  results/     predicted vs annotated table
  tests/       split and label checks
```

```bash
cd dissertation-behaviour-recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Videos, EMOCA `.pkl` files, and virtual environments are not stored in this repository.

## Other folders

Earlier notes and prototypes are kept for reference: `docs/`, `scripts/nod_pipeline/`, `api/`, `web/`. The dissertation experiments use `dissertation-behaviour-recognition/`.

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
