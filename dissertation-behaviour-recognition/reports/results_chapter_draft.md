# Results (paste-ready)

Use **only TEST numbers as headlines**. DEV scores froze the rule and picked the CNN epoch. They are not the result.

JSON sources: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`.
Figures: `dissertation-behaviour-recognition/figures/`.

If Methods already describes the protocol, start this chapter at §5.1. A six-sentence recap is included so the chapter can stand alone.

---

## 5. Results

### 5.1 Evaluation setting (recap)

Columbia RealTalk (Geng et al., 2023), 25 fps; listener **p0 = LEFT**, **p1 = RIGHT**. Thirty windows of about 60 s were **human-labelled**: **1 = clear nod** (gold positive), **0 = unclear**. Split: **15 DEV / 15 TEST**. TEST is scored once.

The frozen pose rule uses Euler axis **x** and amplitude threshold **16.35°**, chosen on DEV. The second system is a **pose-based temporal classifier**: a 1D CNN over the EMOCA `rotation_xyz` sequence, with no RGB input, trained on **80 pseudo-labels** from that rule (70 nod / 10 unclear). Feature set **C** (xyz + first differences, 6-D) is the reported network. Metrics are **clip-level** precision, recall, and F1 from the 2×2 counts on TEST (\(n=15\); 10 nod, 5 unclear). Event F1 at IoU 0.30 is not the metric of this protocol.

Gold, in this chapter, means the human label. It does not mean a prediction.

### 5.2 Gold set

| | DEV | TEST | All |
| --- | ---: | ---: | ---: |
| Videos | 15 | 15 | 30 |
| Clear nod (1) | 9 | 10 | 19 |
| Unclear (0) | 6 | 5 | 11 |
| LEFT / RIGHT (all 30) |  |  | 15 / 15 |

Mean marked gesture interval: 1.1 s. Two marked times lie outside the planned watch window and were kept as recorded (`Ak2Bm8mfL3w`, `Zrer1sqWzOQ`).

**Figures:** `gold_visuals/label_counts.jpg` (or `gold_label_distribution.jpg`); `gold_visuals/labels_by_split.jpg` (or `gold_split_distribution.jpg`); `gold_visuals/labels_by_person.jpg`.

### 5.3 TEST comparison (headline)

| Method | Precision | Recall | F1 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen pose rule | 0.64 | 0.70 | **0.67** | 7 | 4 | 1 | 3 |
| Pose 1D CNN (pseudo-labels) | 0.70 | 0.70 | **0.70** | 7 | 3 | 2 | 3 |

Both systems operate on EMOCA pose features only; no RGB or pixel model is part of this comparison. The CNN matches the rule’s recall (7/10 gold nods) and reduces false positives from 4 to 3. On 15 TEST clips that is **one extra correct rejection**. Secondary scores: accuracy 0.53 (rule) and 0.60 (CNN); balanced accuracy 0.45 (rule) and 0.55 (CNN). Accuracy is not the headline under class imbalance.

**Figures:** `rule_confusion_matrix.jpg`; `classifier_confusion_matrix.jpg`; `model_comparison_f1.jpg`.

DEV F1 was 0.86 (rule) and 0.89 (CNN, epoch 9, probability threshold 0.45). Those numbers describe the tuning set and must not be reported as generalisation.

### 5.4 Feature ablations (TEST)

Training and epoch selection follow Methods. TEST F1 after that selection:

| Feature set | Dim. | Best epoch (DEV) | TEST P | TEST R | TEST F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A single axis (\(x\)) | 1 | 4 | 0.70 | 0.70 | 0.70 |
| B xyz | 3 | 8 | 0.62 | 0.80 | 0.70 |
| C xyz + derivatives (**reported**) | 6 | 9 | 0.70 | 0.70 | **0.70** |
| D xyz + derivatives + expression | 26 | 1 | — | — | *diverged* |

Sets A–C are indistinguishable at this sample size (F1 0.70 at two decimals). Set B raises recall to 8/10 and lowers precision; it is not a second headline. Set D produced `loss = nan` and TEST F1 0 by failed training. **Do not report D as a result** and do not use `ablation_f1.jpg` unless the caption states that D is invalid.

### 5.5 TEST error pattern

| sample | video | gold | rule | CNN |
| --- | --- | ---: | ---: | ---: |
| gold_016 | Cusa1_4R_QI | 1 | TP | TP |
| gold_017 | oQNpe8uwSUA | 0 | FP | FP |
| gold_018 | Zrer1sqWzOQ | 1 | FN | FN |
| gold_019 | YDI27aeM2O8 | 1 | TP | TP |
| gold_020 | V1tcw5SLwmM | 1 | TP | TP |
| gold_021 | MGXtWqf1_BA | 1 | TP | TP |
| gold_022 | jn_3yDP58Ik | 0 | FP | FP |
| gold_023 | zS-xXIiLrWw | 0 | FP | **TN** |
| gold_024 | PM3oaJMiDd4 | 0 | TN | TN |
| gold_025 | J4XrvnkftL8 | 1 | TP | TP |
| gold_026 | G6tLY8FiheE | 1 | FN | FN |
| gold_027 | VSdVKQhnD9s | 1 | TP | TP |
| gold_028 | N-6L1u42cnw | 0 | FP | FP |
| gold_029 | PDd6qEv0_7c | 1 | TP | **FN** |
| gold_030 | ktR3_bXoxaE | 1 | FN | **TP** |

Shared false positives: `gold_017`, `gold_022`, `gold_028`. Shared false negatives: `gold_018`, `gold_026`. The CNN’s F1 gain is `gold_023` (unclear, correctly rejected) while it swaps `gold_029` (miss) for `gold_030` (hit). `gold_018` is the TEST clip whose marked nod time sits outside the pose window.

**Optional figures (not headlines):** `example_positive_rotation.jpg`, `example_negative_rotation.jpg`, `gold_visuals/pose_traces_extracted.jpg`. `training_loss.jpg` and `dev_f1_by_epoch.jpg` belong in an appendix; they are DEV/TRAIN curves.

### 5.6 Frozen VideoMAE head (RGB comparison)

A third system tests whether a generic frozen video representation can replace explicit pose features in this low-data regime. For each of the 110 windows (80 TRAIN pseudo-labelled, 15 DEV, 15 TEST), 16 RGB frames were fetched from the source videos via HTTP range reads, cropped to the listener face (Haar detector; for **12 of 110 clips** no face was found and a centre-crop fallback was used), resized to 224×224, and encoded by the **frozen** `MCG-NJU/videomae-base` VideoMAE encoder. Token embeddings were mean-pooled to a single **768-D** vector per clip, and a small **MLP head** was trained on the same **80 rule pseudo-labels** (70 nod / 10 unclear). Epoch and probability threshold were selected on DEV (early stopping; best epoch **10**, DEV F1 **0.90**, threshold 0.40); TEST was scored **once** under the frozen protocol. Split integrity passed: 80 pseudo clips are disjoint from DEV and TEST by id and video.

| Method | Precision | Recall | F1 | Accuracy | F1 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen VideoMAE head | 0.55 | 0.60 | **0.57** | 0.40 | [0.24, 0.75] |

TEST counts: TP 6, FP 5, TN 0, FN 4 — the head never correctly rejects an unclear clip. The full four-row comparison (including the raw-xyz CNN variant, which has no saved CI) is maintained in `results/tables/main_results.md` and should be cited as the master results table.

**Figure:** `videomae_training_curve.png` (loss and DEV F1 by epoch, best epoch 10 marked; a tuning diagnostic, not a headline) and `model_comparison_f1.png` (TEST F1 of all three systems with 95% CIs).

The VideoMAE head underperforms both pose systems on TEST F1 (0.57 vs 0.67 rule / 0.70 CNN). Three cautions keep this from being over-read. First, with \(n=15\) the 95% bootstrap CIs overlap widely — rule [0.35, 0.87], CNN [0.40, 0.89], VideoMAE [0.24, 0.75] — so the differences are **not statistically significant**. Second, supervision is the same 80 rule pseudo-labels, so label noise sets a ceiling that a larger input modality cannot lift. Third, the DEV–TEST gap (DEV F1 0.90 at epoch 10, TEST F1 0.57) shows the head fit the tuning set without generalising: frozen generic video features, trained for general action recognition, carry less of the nod signal than the explicit EMOCA pose sequence, and the 12 centre-crop fallbacks further degrade the RGB input. In this low-data regime, **explicit pose modelling is competitive with or better than frozen generic video features**; it is also cheaper, needing no pixel access at inference.

### 5.7 What is not in this chapter

- DEV F1 0.86 / 0.89 / 0.90 as a finding
- Ablation D as a valid F1
- Any synthetic `pilot_*` F1
- Event-level F1 at IoU 0.30 for this 30-window protocol
- A claim that F1 0.70 is significantly better than 0.67, or that 0.57 is significantly worse than either

---

**End of Results paste.** Continue with `discussion_conclusion_draft.md`.
