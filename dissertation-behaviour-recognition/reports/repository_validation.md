# Repository validation

Post-cleanup validation of the dissertation repository (Mac, 2026-08-20).
Question answered: **after the cleanup/refactor, do the submitted TEST numbers —
rule F1 0.67 and pose-CNN F1 0.70 — still stand, and does the code still reproduce them?**

Short answer: yes. The rule reproduces bit-for-bit; the CNN artifacts are internally
consistent and the CNN refactor is proven numerically neutral; one training-numerics
caveat is documented below (section 4) and the saved 0.70 artifacts remain canonical.

## 1. Test suite

```
$ python3 -m pytest -q
.....                                                                    [100%]
5 passed in 4.32s
```

`tests/test_invariants.py` covers IoU/event matching, clock parsing, label parsing,
and the figure-log invariant. The no-TEST-leakage invariant is additionally covered by
`scripts/check_split_leakage.py` and the audit in `reports/split_integrity.md`.

## 2. Locked artifacts are internally consistent

Recomputing binary metrics from the saved prediction CSVs with `src.metrics.binary_metrics`
reproduces the saved metric JSONs exactly:

| artifact | P | R | F1 | TP | FP | TN | FN | matches saved JSON |
|---|---|---|---|---|---|---|---|---|
| `results/rule_test_predictions.csv` | 0.6364 | 0.7000 | **0.6667** | 7 | 4 | 1 | 3 | yes |
| `results/classifier_test_predictions.csv` | 0.7000 | 0.7000 | **0.7000** | 7 | 3 | 2 | 3 | yes |

## 3. Rule reproduction: exact

Re-ran the frozen rule over the committed pose features:

- config: `results/rule_selected_config.json` (axis x, savgol window 11 poly 2,
  movement 5–50 frames, amplitude threshold 16.3538328545911)
- code path: `scripts/run_full_experiment.py::rule_score`
- inputs: `features/gold/gold_016..gold_030.npz` (committed)

Result: all 15 TEST amplitude scores identical to `results/rule_test_predictions.csv`
at `atol = 1e-9`, all 15 predictions identical → **TEST F1 0.6667 reproduces
end-to-end and deterministically.** The rule stage involves no randomness.

## 4. CNN: refactor neutrality and a training-numerics caveat

### 4.1 The refactor is numerically neutral

Full pipeline runs with pinned CPU threads (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`)
on identical committed features:

- `/tmp/dbr_p1`, `/tmp/dbr_p2` — two independent **pre-refactor** runs
  (`maybe_train_cnn` inside `run_full_experiment.py`)
- `/tmp/dbr_ref` — **post-refactor** run (`src/pose_cnn.py` + thin wrapper)

Byte-identical between pre- and post-refactor: `classifier_test_predictions.csv`,
`training_history.csv`, `ablation_results.csv`, `rule_test_predictions.csv`,
`pseudo_labels.csv`. The two pre-refactor runs are also byte-identical to each other,
confirming run-to-run determinism under pinned threads. (Only `experiment_config.json`,
`final_results_summary.json`, `storage_after.json` differ across runs, in absolute paths
and disk readings.) **Extracting the CNN into `src/pose_cnn.py` changed no numerical
behaviour.**

### 4.2 CNN training is thread-sensitive; the saved 0.70 is canonical

Retraining on this Mac yields TEST F1 **0.80** (P 0.67, R 1.00; TP 10, FP 5, TN 0,
FN 0) — for the *original* code and the *refactored* code alike, pinned or unpinned.
The submitted realisation scored **0.70** (TP 7, FP 3, TN 2, FN 3).

The inputs are proven unchanged: rule scores and pseudo-labels agree with the locked
CSVs to ~1e-13, so the drift is confined to the CNN gradient-training step — CPU
thread/BLAS reduction order (epoch-1 loss already differs in the 4th decimal between
realisations, then selection diverges). With 15 TEST windows, one borderline window
crossing the DEV-chosen threshold moves F1 by ~0.1.

Consequences, per the submission rules:

- The locked artifacts in `results/` were **not** overwritten; 0.67 / 0.70 stand.
- The dissertation reports the single submitted realisation, with this caveat noted in
  `reports/dissertation_evidence/limitations.md`.
- `src/pose_cnn.py` documents the thread-pinning recipe for deterministic reruns.

### 4.3 Checkpoint lifecycle note

`models/best_1dcnn.pt` and `models/normalization.json` are overwritten per ablation
mode, so a full run leaves mode D's files (mode D diverged, `loss = nan`); the headline
mode-C weights are not retained in the workdir. The durable record of the CNN result is
therefore the prediction CSV + metrics JSON pair. Re-inference from the surviving
mode-D checkpoint yields `nan` probabilities — expected, and consistent with the
recorded ablation note.

## 5. Supporting scripts re-run

- `scripts/19_error_analysis.py` → `results/error_analysis.csv` (15 TEST clips:
  7 both-correct, 3 shared FP, 2 shared FN, 1 rule-only FP, 1 rule-only FN, 1 CNN-only FN)
  — consistent with both locked confusion matrices.
- `scripts/20_annotation_efficiency.py` → `results/annotation_efficiency.json`
  (30 videos, 1800 s watched, 30 events, timing not recorded).

## 6. Late-arriving VideoMAE artifacts (left untouched)

During this cleanup (20 Aug, after the 13:00 brief), a first frozen-VideoMAE-head run
landed in `results/videomae_frozen_head/` (+ `videomae_embeddings_meta.json`,
`figures/videomae_training_curve.png`). The cleanup did not run, modify, or overwrite
any of it. Consistency check only: metrics recomputed from
`videomae_frozen_head/predictions.csv` match `metrics.json` exactly (TEST F1 0.5714,
P 0.5455, R 0.60; TP 6, FP 5, TN 0, FN 4). This is below both pose models and does not
affect the locked 0.67 / 0.70 headlines. Docs that previously claimed "no VideoMAE
score exists" were corrected (READMEs, `SCOPE_MAP.md`, the two preflight reports,
`configs/videomae_frozen.yaml`, and the `scripts/15–17` pointer docstrings).

## 7. Verdict

| claim | status |
|---|---|
| pytest passes after cleanup | yes (5/5) |
| rule TEST F1 0.67 reproduces | yes — bit-identical predictions from frozen config |
| CNN TEST F1 0.70 reproduces | locked artifacts unchanged and internally consistent; retraining is thread-sensitive (documented); refactor proven neutral byte-for-byte |
| ground truth untouched | yes — audits in `reports/annotation_audit.md`, `reports/split_integrity.md` |
