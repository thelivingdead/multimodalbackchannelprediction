# Shake balanced pseudo-labels (new protocol)

Locked `results/shake/pseudo_labels.csv` (**75 pos / 5 neg**, frozen **z**) was **not** overwritten. Labels here are **pseudo**, not gold. GOLD TEST was not used.

## Axis

- Locked TEST rule (untouched): **z**, τ = 11.150°
- New ranking axis (DEV geometry): **y**
- Hard-negative nod-like axis: **x**
- The locked z 0/1 cut is **not** the TRAIN label here (that is the 75/5 collapse).

## Eligible pool on this disk

- `features/pseudo/*.npz`: **80**
- after dropping gold DEV/TEST `video_id`: **80**

40/40 needs 80 clips; 80/80 needs 160; 100/100 needs 200. This Mac has the original 80 pose files unless otter streamed `pseudo_00081.npz`…

## Hard negatives

Ranked among the non-positive remainder:

1. **Nod-like** — high Euler x oscillatory amplitude
2. **Turns** — large shake-axis range with weak oscillation (look-aside)
3. **Mid-range motion** — conversation, not a static head
4. Static (max ptp < 8°) last, never first

## Manifests

| file | n/class | highconf | status | leakage |
| --- | ---: | --- | --- | --- |
| manifest_40_40.csv | 40 | False | wrote | PASS |
| manifest_40_40_highconf.csv | 40 | True | skipped_insufficient_data | — |
| manifest_80_80.csv | 80 | False | skipped_insufficient_data | — |
| manifest_80_80_highconf.csv | 80 | True | skipped_insufficient_data | — |
| manifest_100_100.csv | 100 | False | skipped_insufficient_data | — |
| manifest_100_100_highconf.csv | 100 | True | skipped_insufficient_data | — |
| manifest_200_200.csv | 200 | False | skipped_insufficient_data | — |
| manifest_200_200_highconf.csv | 200 | True | skipped_insufficient_data | — |
| manifest_20_20_highconf.csv | 20 | True | wrote | PASS |

Leakage: `python scripts/check_split_leakage.py --gold-csv data/gold/shake_annotation_sheet.csv --pseudo-labels <manifest>`.
No dyad column exists on gold/pseudo; disjointness is by `video_id`.

Train with `--dev-only` / `--no-test` into `results/shake/dev_balanced/<config>/`.
Do not score GOLD TEST.

