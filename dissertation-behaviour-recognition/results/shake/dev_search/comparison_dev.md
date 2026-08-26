# Shake DEV-only comparison

GOLD DEV class counts: **10 shake+ / 5 shake−** (n=15).
Selection uses DEV only. GOLD TEST was **not** scored for this search.

Collapse rule: predicted-positive rate on DEV **> 0.85** or **TN=0**.
Always-shake on this DEV split has F1 **0.80** (TP10 FP5 TN0 FN0) and is not a trained system.

## Headline rows

| system | P | R | F1 | bal-acc | TP FP TN FN | collapse | train pos/neg | path |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| always-shake baseline (DEV) | 0.667 | 1.000 | 0.800 | 0.500 | 10 5 0 0 | True |  | `(no model)` |
| locked 75/5 pose CNN | 0.769 | 1.000 | 0.870 | 0.700 | 10 3 2 0 | True | 75/5 | `results/shake/cnn/metrics.json` |
| locked 75/5 frozen VideoMAE | nan | nan | 0.833 | 0.600 | confusion not stored | False | 75/5 | `results/shake/videomae_frozen_head/metrics.json` |
| locked 75/5 fine-tuned VideoMAE | 0.714 | 1.000 | 0.833 | 0.600 | 10 4 1 0 | True | 75/5 | `results/shake/videomae_finetuned/metrics.json` |
| search:cnn_40_40 | 0.750 | 0.900 | 0.818 | 0.650 | 9 3 2 1 | False | 40/40 | `results/shake/dev_search/cnn_40_40/dev_metrics.json` |
| search:vmae_frozen_40_40 | 1.000 | 0.300 | 0.462 | 0.650 | 3 0 5 7 | False | 40/40 | `results/shake/dev_search/vmae_frozen_40_40/dev_metrics.json` |
| search:vmae_ft4_40_40 | 0.857 | 0.600 | 0.706 | 0.700 | 6 1 4 4 | False | 40/40 | `results/shake/dev_search/vmae_ft4_40_40/dev_metrics.json` |
| search:vmae_frozen_20_20_highconf | 0.750 | 0.900 | 0.818 | 0.650 | 9 3 2 1 | False | 20/20 | `results/shake/dev_search/vmae_frozen_20_20_highconf/dev_metrics.json` |

## All DEV-search runs

| system | P | R | F1 | collapse | pred+ rate | path |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| search:cnn_20_20_highconf | 0.769 | 1.000 | 0.870 | True | 0.867 | `results/shake/dev_search/cnn_20_20_highconf/dev_metrics.json` |
| search:cnn_40_40 | 0.750 | 0.900 | 0.818 | False | 0.800 | `results/shake/dev_search/cnn_40_40/dev_metrics.json` |
| search:vmae_frozen_20_20_highconf | 0.750 | 0.900 | 0.818 | False | 0.800 | `results/shake/dev_search/vmae_frozen_20_20_highconf/dev_metrics.json` |
| search:vmae_frozen_40_40 | 1.000 | 0.300 | 0.462 | False | 0.200 | `results/shake/dev_search/vmae_frozen_40_40/dev_metrics.json` |
| search:vmae_ft4_40_40 | 0.857 | 0.600 | 0.706 | False | 0.467 | `results/shake/dev_search/vmae_ft4_40_40/dev_metrics.json` |

## Window length

Pose CNN default is 128 resampled steps; one cheap variant uses `--seq-len 64` (`cnn_*_seq64`). VideoMAE rgb16 stays 16×224×224 (no second window without a new fetch).

## One best config (DEV only — not scored on TEST)

**search:cnn_40_40**  F1=0.818  P=0.750  R=0.900  bAcc=0.650  collapse=False  (highest DEV F1 among non-collapsed search runs)
Path: `results/shake/dev_search/cnn_40_40/dev_metrics.json`

## Do not report a winner TEST F1

Locked TEST numbers from the 75/5 protocol are already known and are **not** a selection criterion. Next evaluation should be a **fresh 10–15 clip holdout** from videos that are **not** in the gold 30.

Footnote (already-known locked TEST, not for selection): shake rule TEST F1 0.70; 75/5 CNN TEST F1 0.64 with TN=0; 75/5 VideoMAE TEST F1 0.60. Ignore these when picking the best new run.

Student git-pushes themselves. Do not score GOLD TEST yet.

