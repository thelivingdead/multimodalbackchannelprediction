# Gold annotation results (30 videos)

Human labels plus streamed EMOCA pose. Clip-level TEST F1: rule **0.67**, 1D CNN **0.70**. Synthetic `pilot_*` clips are not used.

## Protocol

- Dataset: Columbia RealTalk (Geng et al., 2023), 25 fps. Listener: p0 = LEFT, p1 = RIGHT.
- 30 × ~1 min watch windows. Labels: `1` = clear nod (gold positive), `0` = unclear.
- Split: 15 DEV (tune later) / 15 TEST (score once, never train).
- Times from YouTube clock; imported to `data/gold/events.csv`.

## Counts

| | DEV | TEST | All |
| --- | ---: | ---: | ---: |
| Videos | 15 | 15 | 30 |
| Clear nod (`1`) | 9 | 10 | 19 |
| Unclear (`0`) | 6 | 5 | 11 |
| LEFT / RIGHT |  |  | 15 / 15 |

Mean labelled interval length: 1.1 s.

Two times sit outside the planned window and were kept as recorded: `Ak2Bm8mfL3w` (1:57–1:58 vs 13:34–14:34), `Zrer1sqWzOQ` (4:48–4:49 vs 4:56–5:56).

## TEST metrics (clip-level; n = 15)

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Frozen pose rule | 0.64 | 0.70 | 0.67 |
| 1D CNN (80 pseudo-labels) | 0.70 | 0.70 | 0.70 |

DEV was used only to freeze the rule and the CNN epoch. Full write-up: `reports/results_chapter_draft.md`.

Figures: `figures/gold_label_distribution.jpg`, `figures/rule_confusion_matrix.jpg`, `figures/classifier_confusion_matrix.jpg`.
