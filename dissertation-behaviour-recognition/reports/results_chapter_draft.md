# Results (paste-ready)

Use **only TEST numbers as headlines**. DEV scores froze the rule and picked the CNN epoch. They are not the result.

JSON sources: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/videomae_frozen_head/metrics.json`, `results/videomae_finetuned/metrics.json`. Master table: `results/tables/main_results.md`.
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

DEV F1 was 0.86 (rule) and 0.89 (CNN, epoch 9, probability threshold 0.45). Those numbers describe the tuning set and must not be reported as generalisation. Two RGB systems (frozen and fine-tuned VideoMAE) under the same protocol are reported in §5.6–5.7; the full five-model ordering is in the master table `results/tables/main_results.md`.

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

### 5.7 Fine-tuned VideoMAE (partial, GPU)

The frozen head of §5.6 underfits: it cannot adapt generic video features to the nod decision, and its DEV–TEST gap suggests the head memorised the tuning set. The fourth system therefore tests whether **task adaptation of the video backbone itself** closes the gap, using identical inputs and splits.

**Setup.** The same 16-frame, 224×224 listener face crops and the same `MCG-NJU/videomae-base` checkpoint were used, but as `VideoMAEForVideoClassification` with a single logit (`num_labels=1`). Patch embeddings and the first 8 encoder blocks were frozen; the **last 4 encoder blocks plus `fc_norm` and the classifier** were trained — **28.4M of 86.2M parameters**. AdamW with learning rates **1e-5 (backbone) / 1e-4 (head)**, batch size 8, `BCEWithLogitsLoss` with `pos_weight = 0.143` (TRAIN 70 nod / 10 unclear), and horizontal-flip augmentation on TRAIN only. Training used automatic mixed precision on a single RTX A4000 16 GB (lab host otter95). Because the 25 GB home quota could not hold a CUDA PyTorch stack, GPU PyTorch was installed on **`/scratch`** (local disk outside the quota); preprocessing was verified **bit-identical** to the frozen pipeline (max abs diff 0.0 against `VideoMAEImageProcessorPil`).

**Protocol.** Identical TRAIN/DEV/TEST splits as every other system; the leakage gate passed at startup. Epoch and probability threshold were selected on DEV only (early stopping; best epoch **5**, DEV F1 **0.857**, threshold **0.45**); TEST was scored **exactly once**.

|| Method | Precision | Recall | F1 | Accuracy | F1 95% CI |
|| --- | ---: | ---: | ---: | ---: | --- |
|| Fine-tuned VideoMAE (last 4 blocks) | 0.75 | 0.90 | **0.82** | 0.73 | [0.60, 0.96] |

TEST counts: TP 9, FP 3, TN 2, FN 1 — the highest TEST F1 point estimate of the five systems (rule 0.67, pose CNN 0.70, frozen head 0.57, fine-tuned 0.82; the raw-xyz CNN variant, 0.70, has no saved CI). The full comparison is the master table `results/tables/main_results.md`.

**Figure:** `videomae_finetuned_training_curve.png` (loss and DEV F1 by epoch, best epoch 5 marked; a tuning diagnostic, not a headline) and `model_comparison_f1.png` (TEST F1 of the four CI-backed systems with 95% CIs, fine-tuned bar highlighted).

Three cautions apply, exactly as for the frozen head. First, **no pairwise difference is statistically significant**: at \(n=15\) the fine-tuned 95% CI [0.60, 0.96] overlaps the pose CNN's [0.40, 0.89] and the rule's [0.35, 0.87]; 0.82 is the **highest point estimate**, not a proven win. Second, supervision is the same 80 rule pseudo-labels, so label noise still ceilings every system. Third, the frozen-vs-fine-tuned pair (0.57 vs 0.82 on identical inputs, splits, and labels) indicates that **task adaptation of the video backbone, not the RGB input alone, drove the gain** — though the frozen head's own CI [0.24, 0.75] overlaps the fine-tuned interval, so this contrast is also not significant at \(n=15\).

A process note for the record: an earlier bootstrap over all 110 saved predictions (`results/videomae_finetuned/predictions.csv`, which carries a `split` column) was **train-contaminated and invalid**; it was caught and corrected. The canonical CI file is the 15-row `results/videomae_finetuned/predictions_test.csv`, and every CI reported here uses TEST only.

### 5.8 Scaling ablation: 80 → 200 pseudo-labels

The same partial fine-tune recipe was repeated with **200** frozen-rule pseudo-labels (the original 80 rows left byte-identical, plus 120 new clips). DEV and TEST were unchanged. Outputs went to `results/videomae_finetuned_n200/` so the n=80 TEST score was not overwritten. Best-on-DEV: epoch **9**, threshold **0.80** (DEV F1 0.889 — tuning only). TEST was scored once.

| Method | TRAIN n | Precision | Recall | F1 | Accuracy | F1 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fine-tuned VideoMAE (canonical) | 80 | 0.75 | 0.90 | **0.82** | 0.73 | [0.60, 0.96] |
| Fine-tuned VideoMAE (scaling) | 200 | 0.67 | 0.60 | **0.63** | 0.53 | [0.31, 0.84] |

TEST counts at n=200: TP 6, FP 3, TN 2, FN 4. The point estimate **fell** relative to n=80. The 95% intervals overlap ([0.60, 0.96] vs [0.31, 0.84]), so the drop is **not statistically significant** at \(n=15\). The defensible reading is that extra automatic labels from the same noisy teacher did not help, and the canonical RGB result remains the 80-clip run. Do not treat n=200 as a second headline, and do not rerun it.

### 5.9 Three-second sliding-window protocol: nod baselines

Under the finer protocol of Section 4.11 the pitch amplitude rule does not separate nod windows from non-nod windows. The DEV sweep was run with balanced accuracy as the objective, as declared in Section 4.11, and TEST was then scored once at the selected threshold of 2.68 degrees. Balanced accuracy is the headline metric; the floor for any constant predictor is 0.500.

| Method | DEV balanced accuracy | TEST balanced accuracy | TEST P | TEST R | TEST F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always no | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Always yes | 0.500 | 0.500 | 0.159 | 1.000 | 0.274 |
| Frozen 60 s threshold, transferred | 0.486 | 0.494 | 0.148 | 0.130 | 0.138 |
| DEV-selected pitch rule | 0.580 | **0.549** | 0.179 | 0.710 | 0.287 |

The 95 percent clip-bootstrap interval for the selected rule is [0.520, 0.647] on DEV and **[0.480, 0.619] on TEST**. The TEST interval contains 0.500, so on 15 clips this rule is not distinguishable from chance. This is also the clearest illustration of why F1 was rejected as the headline: the selected rule's TEST F1 of 0.287 sits only 0.013 above the always-yes value of 0.274, a gap that invites over-reading, while the same operating point in balanced accuracy is 0.549 against a floor of exactly 0.500 with an interval that crosses it.

Two observations matter for the interpretation. First, balanced accuracy and F1 select the *same* DEV threshold, 2.68 degrees, so the near-always-yes behaviour of the rule is not an artefact of the selection criterion: at that threshold the rule fires on 64.8 percent of DEV windows to reach recall 0.788 at precision 0.145. Across the whole 434-point DEV sweep balanced accuracy never exceeds 0.580 and its median is 0.536, so no threshold on this score yields a useful operating point. Second, the ranking itself is close to uninformative on DEV: PR AUC is 0.131 against a prevalence of 0.120. On TEST, PR AUC is 0.207 against a prevalence of 0.159, a slightly wider but still small margin. The criterion change was worth making because it makes the comparison against the trivial baseline meaningful, but it does not rescue the feature.

The weakly supervised pose model does not improve on this. A multiple-instance pose network, trained on the 80 pseudo-labelled clips as bags of 29 windows with top-2 pooling on rotation and its first difference, and selected on DEV balanced accuracy, reaches DEV balanced accuracy 0.533 at epoch 4 and probability threshold 0.25, with precision 0.128 at recall 0.846 (TP 44, FP 299, TN 84, FN 8). It fires on 343 of 435 DEV windows, 78.9 percent, against a true prevalence of 12.0 percent, which is the always-yes collapse made visible: the same operating point reads as F1 0.223 and as balanced accuracy 0.533. It is therefore *worse* on DEV than the one-parameter amplitude rule at 0.580, so neither the additional capacity nor the 80 weakly labelled clips bought anything. TEST was not scored for this model, and results are in `results/windowed_nod/pose_mil_pseudo80_dev_bacc/metrics_dev.json`.

That a hand rule and a learned model both land just above chance, from opposite directions, points at the representation rather than at either fitting procedure. Peak-to-peak pitch amplitude over 3 s cannot distinguish an oscillation from a single downward glance or a postural drift of the same magnitude, and at 3 s many non-nod windows contain exactly such movements; at 60 s these were diluted by the rest of the window. A rule that also counted direction reversals or zero crossings of the smoothed pitch velocity would address this directly and is the natural next step, but it is not part of the present results.

The transferred 60 s threshold is reported unchanged: TEST F1 0.138 at balanced accuracy 0.494, marginally below chance. The mechanism is a scale mismatch rather than anything subtle. Peak-to-peak amplitude is a maximum over the window, so it grows with window length: the value frozen on 60 s windows is 16.35 degrees, whereas the DEV optimum for 3 s windows is 2.68 degrees, a factor of six. A 3 s window almost never accumulates 16 degrees of pitch excursion, so the transferred rule fires on only 61 of 435 TEST windows and recovers 13 percent of the nods. The transfer failure is therefore expected on dimensional grounds and is reported without refitting of any kind.

### 5.10 Three-second sliding-window protocol: shake baselines

The same protocol was applied to head shakes, using the shake event annotations for all 30 gold clips. Positive rates are lower than for nod: 39/435 (9.0 percent) on DEV and 40/435 (9.2 percent) on TEST. No frozen 60 s shake threshold exists, and the repository does not assume that EMOCA rotation channel 0 is anatomical pitch, so the axis was selected on shake DEV together with the amplitude threshold. That is one additional disclosed DEV decision relative to nod, which inherited its axis from the frozen 60 s configuration. The full three-axis DEV table is reported for transparency.

| Axis | DEV threshold | DEV balanced accuracy | DEV precision | DEV recall | DEV F1 | DEV PR AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.321 | 0.636 | 0.121 | 0.949 | 0.215 | 0.129 |
| 1 | 4.091 | **0.704** | 0.173 | 0.769 | 0.283 | 0.186 |
| 2 | 3.666 | 0.684 | 0.163 | 0.744 | 0.267 | 0.174 |

Axis 1 was selected and applied once to TEST at 4.091 degrees.

| Method | DEV balanced accuracy | TEST balanced accuracy | TEST P | TEST R | TEST F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always no | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Always yes | 0.500 | 0.500 | 0.092 | 1.000 | 0.168 |
| DEV-selected yaw rule | 0.704 | **0.654** | 0.153 | 0.700 | 0.251 |

The 95 percent clip-bootstrap interval is [0.611, 0.793] on DEV and **[0.525, 0.794] on TEST**. Unlike nod, the TEST interval excludes 0.500, so the shake rule is distinguishable from chance, although the lower bound clears the floor by only 0.025 on 40 positives from 15 clips. TEST counts are TP 28, FP 155, TN 240, FN 12. Ranking quality is correspondingly better than for nod: DEV PR AUC 0.186 against a prevalence of 0.090, and TEST 0.164 against 0.092, roughly twice chance in both cases, where the nod rule managed 0.131 against 0.120.

Two consequences follow. First, the axis sweep is an independent check on the pose features: shake selects axis 1 while nod's frozen configuration uses axis 0, so the rotation channels do separate vertical from horizontal head motion as an anatomical reading of pitch and yaw would require. Had shake also selected axis 0, the feature extraction itself would be in question. Second, and more importantly, the 3 s protocol is not inherently intractable and the pipeline is not at fault: a single amplitude parameter on the correct axis detects shakes but not nods. This is the amplitude-versus-oscillation argument in its sharpest form. A head shake is a distinctive horizontal oscillation with few competing sources of yaw excursion at this timescale, whereas vertical motion of nod-like magnitude arises constantly from downward glances and postural drift. The nod result of Section 5.9 is therefore a claim about which behaviour amplitude can separate at short timescales, not a claim that short-window recognition fails in general.

One further note for the methods. Here the two selection criteria disagree: balanced accuracy selects 4.091 degrees while F1 would select 7.783. F1 prefers the higher, more conservative cut because at 9 percent prevalence the precision gain outweighs the lost recall in its harmonic mean, whereas balanced accuracy prefers the lower cut on the specificity-weighted trade. For nod the two criteria agreed exactly, so the disagreement here is a property of this score distribution rather than of the criteria in general.

### 5.11 What is not in this chapter

- DEV F1 0.86 / 0.89 / 0.90 / 0.857 as a finding
- Ablation D as a valid F1
- Any synthetic `pilot_*` F1
- Event-level F1 at IoU 0.30 for this 30-window protocol
- A claim that F1 0.70 is significantly better than 0.67, that 0.57 is significantly worse than either, that 0.82 is significantly better than any pose system, or that n=200 is significantly worse than n=80 — at \(n=15\) all 95% CIs overlap

---

**End of Results paste.** Continue with `discussion_conclusion_draft.md`.
