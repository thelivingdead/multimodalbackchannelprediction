# Head-shake DEV-only improvement — parent report

Do **not** score GOLD TEST. Student git-pushes themselves. This agent did not `git add` / commit / push.

## Verified shake rotation axis

**y** (yaw-like / left–right). **z was not geometrically correct** for shake.

- EMOCA `rotation_xyz` = `Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)`.
- GOLD DEV exclusive labels: shake-only peaks on **y** (25.3°) not **z** (15.7°); nod-only peaks on **x** (38.3°).
- Shake+ minus shake− rule-score gap is largest on **y** (+7.75°); 1–5 Hz band energy agrees.
- Locked TEST rule stays **z**, τ ≈ **11.15°** (`rule_selected_config.json` not rewritten).
- Video compare: no `features/rgb16/` on this Mac; needs otter or Mac playback.

Plots: `figures/shake_axis_audit/` (DEV shake+ gold_004, 010, 011, 012, 001; all five shake− gold_003, 005, 006, 009, 014).

## DEV class distribution

GOLD DEV: **10 shake / 5 no-shake** (n=15). TEST (locked, not used): 7 / 8.

Always-shake on DEV: P 0.667, R 1.000, F1 **0.800**, bAcc 0.500, TP10 FP5 TN0 FN0 (collapse).

## Locked 75/5 DEV numbers (read from existing json; TEST not rescored)

| system | P | R | F1 | bAcc | TP FP TN FN | collapse |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| pose CNN | 0.769 | 1.000 | 0.870 | 0.700 | 10 3 2 0 | yes (13/15 pred+) |
| frozen VideoMAE | — | — | 0.833 | 0.600 | not stored | likely (bAcc 0.6) |
| FT VideoMAE | 0.714 | 1.000 | 0.833 | 0.600 | 10 4 1 0 | yes |

## Pseudo train distributions

Locked `results/shake/pseudo_labels.csv` **untouched**: **75 pos / 5 neg**, 80 unique `video_id`s, **0 leakage** vs gold DEV/TEST videos. No dyad column; exclusion is by `video_id`.

New ranking axis **y** (not frozen-z 0/1). Frozen-z on this pool would still be 75/5.

| file | pos/neg | axis | leakage | hard-neg (nod / turn / static) |
| --- | --- | --- | --- | --- |
| `pseudo_balanced/manifest_40_40.csv` | **40 / 40** | y | PASS | 14 / 4 / 0 |
| `pseudo_balanced/manifest_20_20_highconf.csv` | **20 / 20** | y | PASS | 4 / 2 / 0 |
| 80/80, 100/100, 200/200 | skipped | — | — | Mac has only 80 npz (`pseudo_00081+` missing) |

## Best DEV F1 of *new* models

Quoted from `comparison_dev.md` / `cnn_40_40/metrics_dev.json`. GOLD TEST was **not** scored (`test_scored: false`).

| system | P | R | F1 | bAcc | TP FP TN FN | collapse |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| search:cnn_40_40 | 0.750 | 0.900 | **0.818** | 0.650 | 9 3 2 1 | False |
| always-shake baseline (DEV) | 0.667 | 1.000 | 0.800 | 0.500 | 10 5 0 0 | True |
| search:vmae_frozen_40_40 | 1.000 | 0.300 | 0.462 | 0.650 | 3 0 5 7 | False |
| search:vmae_ft4_40_40 | 0.857 | 0.600 | 0.706 | 0.700 | 6 1 4 4 | False |

**search:cnn_40_40** is the best non-collapsed search run. Frozen VideoMAE 40/40 is conservative (R 0.30). Fine-tune last-4 40/40 (F1 0.706) did not beat the CNN on DEV. `cnn_20_20_highconf` DEV F1 0.870 is collapsed (pred+ 0.867) and is not eligible.

## Best configuration

**search:cnn_40_40** — pose 1D CNN, 40/40 balanced pseudo, seed 42.

- path: `results/shake/dev_search/cnn_40_40/metrics_dev.json`
- DEV F1 **0.818** (P 0.750, R 0.900, bAcc 0.650; TP9 FP3 TN2 FN1)
- collapse **false**; `test_scored` **false**
- ranking axis **y** (new TRAIN); locked TEST rule stays **z**
- **do not score TEST**; do not `--force` this out-dir

Always-shake on this DEV split is F1 **0.80** (10 pos / 5 neg) and is not a trained win.

## Recommend fresh holdout

**Yes — annotate 10–15 new untouched clips** (new `video_id`s, not in the gold 30). DEV n=15 is noisy; GOLD TEST has already been seen; 75/5 collapse is the failure mode.

## Next command

DEV search already wrote `results/shake/dev_search/` (`comparison_dev.md`). Do **not** `--force` existing metrics dirs. Do **not** rescore GOLD TEST.

## Files created / updated (scripts)

- `scripts/audit_shake_axis_dev.py`
- `scripts/build_shake_balanced_pseudo.py` (y-ranking; never glob-deletes 40/40)
- `scripts/shake_v2_common.py`
- `scripts/train_shake_cnn.py` (`--dev-only` / `--no-test`)
- `scripts/train_shake_cnn_dev.py`
- `scripts/train_videomae_head.py` / `finetune_videomae.py` (`score_test=False` path)
- `scripts/train_videomae_shake_head_dev.py` / `finetune_videomae_shake_dev.py`
- `scripts/run_shake_dev_search.py` / `.sh`
- `scripts/compare_shake_dev_search.py`
- `scripts/check_split_leakage.py` (locked TEST dirs)

## Artefacts

- `figures/shake_axis_audit/`
- `results/shake/dev_search/axis_audit.md`, `axis_audit_conclusion.json`, `comparison_dev.md`, `summary.csv`, `best_config.json`, `cnn_40_40/metrics_dev.json`
- `results/shake/pseudo_balanced/manifest_40_40.csv`, `manifest_20_20_highconf.csv`

Locked nod and locked shake TEST dirs were not written.
