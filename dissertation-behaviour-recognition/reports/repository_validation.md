# Repository validation

**Updated:** 1 September 2026. Metrics below are recomputed from **saved TEST prediction CSVs** with `src.metrics.binary_metrics`. Models were **not** re-run on TEST.

Canonical headlines (rounded as in the README): pose rule **0.67**, pose CNN **0.70**, frozen VideoMAE **0.57**, fine-tuned VideoMAE **0.82**. Exact saved F1s are 0.6667 / 0.7000 / 0.5714 / 0.8182.

## 1. Test suite (1 Sep 2026)

```
$ python -m pytest -q
22 passed, 14 warnings in 9.14s
```

Warnings are matplotlib/pyparsing deprecations, not test failures. Coverage now includes split leakage, VideoMAE/shake path isolation, locked-directory refusal (`results/joint/videomae_finetuned` and the rest of `LOCKED_OUT_DIRS`), audio DEV-only guards, and no `/user/` or `/home/` prefixes in `results/`.

The 20 August 2026 run (`5 passed`) is superseded by this suite; it is not a regression in the study results.

## 2. Metrics recomputed from stored TEST predictions

| artefact | P | R | F1 | TP FP TN FN | matches saved JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `results/rule_test_predictions.csv` | 0.6364 | 0.7000 | **0.6667** | 7 4 1 3 | yes (`rule_test_metrics.json`) |
| `results/classifier_test_predictions.csv` | 0.7000 | 0.7000 | **0.7000** | 7 3 2 3 | yes (`classifier_test_metrics.json`) |
| `results/videomae_frozen_head/predictions.csv` (TEST rows) | 0.5455 | 0.6000 | **0.5714** | 6 5 0 4 | yes (`videomae_frozen_head/metrics.json`) |
| `results/videomae_finetuned/predictions_test.csv` | 0.7500 | 0.9000 | **0.8182** | 9 3 2 1 | yes (`videomae_finetuned/metrics.json`) |
| `results/videomae_finetuned_n200/predictions.csv` (TEST rows) | 0.6667 | 0.6000 | **0.6316** | 6 3 2 4 | yes (`videomae_finetuned_n200/metrics.json`) |

These are **recomputed from stored predictions**, not a new TEST inference.

## 3. TEST-lock protections

`scripts/check_split_leakage.py` `LOCKED_OUT_DIRS` includes nod VideoMAE (`videomae_finetuned`, `videomae_finetuned_n120`, `videomae_finetuned_n200`, `videomae_frozen_head`), shake TEST dirs, and `results/joint/videomae_finetuned`. `assert_unlocked_out_dir` refuses writes into those paths. Unit tests assert that refusal.

## 4. Historical note (20 Aug 2026) — CNN training numerics

The 20 August cleanup established that the pose **rule** reproduces bit-for-bit from committed `features/gold/*.npz`, and that extracting the CNN into `src/pose_cnn.py` did not change stored 0.70 artefacts. Retraining the CNN on another machine can yield a different TEST F1 because of CPU thread/BLAS order; **the locked 0.70 CSV/JSON remain canonical** and were not overwritten. See `reports/dissertation_evidence/limitations.md`.

## 5. Verdict

| claim | status |
| --- | --- |
| pytest | **22 passed** (1 Sep 2026) |
| rule TEST F1 0.67 | matches stored predictions |
| CNN TEST F1 0.70 | matches stored predictions |
| frozen VideoMAE TEST F1 0.57 | matches stored predictions |
| fine-tuned VideoMAE TEST F1 0.82 | matches stored predictions |
| n=200 ablation F1 0.63 | matches stored predictions |
| TEST inference rerun | **no** |
