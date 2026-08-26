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
| best balanced pose CNN (DEV) | — | — | — | — | — | — | — | not run yet |
| best balanced frozen VideoMAE (DEV) | — | — | — | — | — | — | — | not run yet |
| best balanced fine-tuned VideoMAE (DEV) | — | — | — | — | — | — | — | not run yet |
| best high-confidence balanced VideoMAE (DEV) | — | — | — | — | — | — | — | not run yet |

## All DEV-search runs

| system | P | R | F1 | collapse | pred+ rate | path |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| *(no `dev_search/*/dev_metrics.json` yet)* | | | | | | |

## Window length

Pose CNN default is 128 resampled steps; one cheap variant uses `--seq-len 64` (`cnn_*_seq64`). VideoMAE rgb16 stays 16×224×224 (no second window without a new fetch).

## One best config (DEV only — not scored on TEST)

No new `dev_search/*/dev_metrics.json` yet. Run otter jobs.

## Do not report a winner TEST F1

Locked TEST numbers from the 75/5 protocol are already known and are **not** a selection criterion. Next evaluation should be a **fresh 10–15 clip holdout** from videos that are **not** in the gold 30.

Footnote (already-known locked TEST, not for selection): shake rule TEST F1 0.70; 75/5 CNN TEST F1 0.64 with TN=0; 75/5 VideoMAE TEST F1 0.60. Ignore these when picking the best new run.

Student git-pushes themselves. Do not score GOLD TEST yet.

