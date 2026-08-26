# Shake axis audit (GOLD DEV only)

Selection uses **GOLD DEV only**. This file is not a GOLD TEST F1.

## DEV class counts

- gold DEV clips: **15**
- shake+ (`shake_label=1`): **10**
- shake− (`shake_label=0`): **5**
- nod+ on the same clips: **9**
- shake-only (shake=1, nod=0): **4**
- nod-only (shake=0, nod=1): **3**

Gold TEST (15 clips) was **not** used to choose the axis.

## Convention (do not assume names a priori)

EMOCA/FLAME pose is **used, not trained**. Stored `rotation_xyz` is
`Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)` →
**x, y, z**. Literature maps these to pitch / yaw / roll, but this
audit decides from DEV traces.

Locked **nod** rule used **x** (τ = 16.35°). Locked **shake** rule
used **z** (τ = 11.150°) by DEV F1, with a note
that yaw was *hypothesised* to be y.

## Clips plotted (not ranked by frozen z)

Shake+ (shake-only first, then mixed): gold_004, gold_010, gold_011, gold_012, gold_001.

Shake− (all five DEV negatives): gold_003, gold_005, gold_006, gold_009, gold_014.

No `features/rgb16/*.npz` on this Mac for the plotted gold ids. YouTube URLs are in `data/gold/shake_annotation_sheet.csv` but were not fetched. Anatomical left-right vs nod was **not** watched. On otter, rgb16 crops exist for the locked VideoMAE runs; they were not re-scored here. Video compare needs otter or Mac playback.

## Mean oscillatory amplitude on DEV (rule_score, degrees)

| axis | literature | mean shake+ | mean shake− | + minus − | shake-only | nod-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| x | pitch (nod-like) | 23.57 | 31.89 | -8.33 | 16.96 | 38.31 |
| y | yaw (shake-like) | 32.96 | 25.21 | 7.75 | 25.32 | 32.30 |
| z | roll (tilt-like) | 24.42 | 17.39 | 7.03 | 15.65 | 16.45 |

1–5 Hz band energy (qualitative, not a detector): largest shake+ minus shake− gap on **y**.

## Verdict

Geometric shake axis on GOLD DEV is **y** (yaw-like). Shake-only clips peak on y (25.3°) not z (15.7°). Nod-only clips peak on **x** (38.3°). Locked **z** is roll-like and is **not** geometrically supported as left-right. **New pseudo-labels use y.** Locked TEST artefacts stay on z (τ=11.150°) and are not rewritten.

### YouTube windows (videos were not on this Mac)

- `gold_001` (`IH6KWbTogT0`) shake=1 nod=1  watch LEFT 0:55–1:55  https://www.youtube.com/watch?v=IH6KWbTogT0&t=55
- `gold_003` (`xkHwlcDSOjc`) shake=0 nod=0  watch LEFT 2:31–3:31  https://www.youtube.com/watch?v=xkHwlcDSOjc&t=150
- `gold_004` (`RzIxWA-ll8g`) shake=1 nod=0  watch LEFT 11:41–12:41  https://www.youtube.com/watch?v=RzIxWA-ll8g&t=701
- `gold_005` (`D8K1AAxkg0g`) shake=0 nod=0  watch LEFT 0:48–1:48  https://www.youtube.com/watch?v=D8K1AAxkg0g&t=47
- `gold_006` (`niEsUBm1l98`) shake=0 nod=1  watch RIGHT 27:02–28:02  https://www.youtube.com/watch?v=niEsUBm1l98&t=1622
- `gold_009` (`GJtqigeWHV8`) shake=0 nod=1  watch RIGHT 2:04–3:04  https://www.youtube.com/watch?v=GJtqigeWHV8&t=123
- `gold_010` (`jg6y3LABwTs`) shake=1 nod=0  watch RIGHT 1:28–2:28  https://www.youtube.com/watch?v=jg6y3LABwTs&t=88
- `gold_011` (`FzCjvLU7u7Q`) shake=1 nod=0  watch LEFT 1:22–2:22  https://www.youtube.com/watch?v=FzCjvLU7u7Q&t=81
- `gold_012` (`6RDkdbgzeAI`) shake=1 nod=0  watch LEFT 6:08–7:08  https://www.youtube.com/watch?v=6RDkdbgzeAI&t=368
- `gold_014` (`f6aNo5Mod9I`) shake=0 nod=1  watch LEFT 0:53–1:53  https://www.youtube.com/watch?v=f6aNo5Mod9I&t=52

Per-clip scores: `axis_audit_dev_scores.csv`. TEST clips were not scored.

