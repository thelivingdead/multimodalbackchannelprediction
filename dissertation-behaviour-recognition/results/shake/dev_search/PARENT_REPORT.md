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

**Not trained yet.** Mac has numpy/pandas/matplotlib only (no torch, no VideoMAE embeddings, no rgb16). No `dev_search/*/dev_metrics.json`.

Class collapse is **not yet shown to be fixed** in a trained model. The 40/40 y-ranking + hard negatives is the intended fix; otter must train to measure it.

## Best configuration

**Not picked** (no new DEV metrics). Intended protocol when otter finishes:

- out-dir under `results/shake/dev_search/`
- train size 40/40 (or 80/80 if extra npz)
- highconf vs all chosen on DEV balanced accuracy (not F1)
- frozen MLP vs last-4-block FT
- ranking axis **y**
- seed **42**
- threshold: DEV `balanced_accuracy` sweep
- **do not score TEST**

Reject if DEV pred+ ≥ 13/15 or TN=0 (always-1 F1 0.80 is not a win).

## Recommend fresh holdout

**Yes — annotate 10–15 new untouched clips** (new `video_id`s, not in the gold 30). DEV n=15 is noisy; GOLD TEST has already been seen; 75/5 collapse is the failure mode.

## Next command (otter95)

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
PY=/scratch/db01550/venv/bin/python
bash scripts/run_shake_dev_search.sh
```

Or the commands in `results/shake/dev_search/OTTER_COMMANDS.md`. If `features/pseudo/pseudo_00081.npz`… exist, the builder will add 80/80 (and 100/100 if 200 clips). Then `scripts/compare_shake_dev_search.py`. **Do not score TEST.**

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
- `results/shake/dev_search/axis_audit.md`, `axis_audit_conclusion.json`, `comparison_dev.md`, `summary.csv`, `OTTER_COMMANDS.md`
- `results/shake/pseudo_balanced/manifest_40_40.csv`, `manifest_20_20_highconf.csv`

Locked nod and locked shake TEST dirs were not written.
