## 4. Methods

This chapter describes the study that was run. The task is clip-level recognition of a clear listener head nod on Columbia RealTalk (Geng et al., 2023). Two pose systems are compared on a frozen TEST set of 15 windows: a deterministic pose-amplitude rule whose axis and threshold are chosen on DEV only, and a 1D convolutional network over the EMOCA `rotation_xyz` sequence trained on 80 pseudo-labels produced by that frozen rule. Two RGB systems based on VideoMAE were run under the same frozen protocol (Section 4.8): a frozen encoder with a trained head, and a partial fine-tune of the last four encoder blocks on a lab GPU. Full fine-tuning was not attempted, for the storage reasons given in Section 4.8.

### 4.1 Scope of the submitted study

The original project notes outlined a seven-class backchannel taxonomy (nod, shake, tilt, lean forward, lean back, eyebrow raise, neutral) and a multimodal predictor that would fuse text, audio, video, and FLAME/EMOCA sequences (see `docs/archive/02_research_methodology_and_roadmap.md`). That plan is not the submitted experiment.

The submitted study is narrower for two reasons. First, reliable human labels were obtained only for a binary decision: clear nod versus unclear. Second, the lab account used for pose extraction (otter) has a home quota of about 25 GB. After a CPU PyTorch install, free space was about 6.5 GB, which is not enough to store RealTalk video shards together with a VideoMAE checkpoint. Official EMOCA/FLAME pickles were streamed from Hugging Face rather than written to disk as `emoca.tar.gz`.

The research questions this chapter is designed to answer are:

1. **RQ1.** After the pose rule is frozen on 15 DEV windows, what clip-level precision, recall, and F1 does it obtain on 15 unseen TEST windows?
2. **RQ2.** Does a 1D CNN trained only on 80 automatic rule labels improve those TEST scores?
3. **RQ3.** At this sample size, do extra pose channels (full Euler xyz, first differences, expression coefficients) change TEST F1?

Two constrained VideoMAE variants, a frozen head and a partial fine-tune, were run and are reported as results in Section 4.8. Full VideoMAE fine-tuning, seven-class typing, and event-level F1 at temporal IoU 0.30 remain future work.

### 4.2 Dataset and listener convention

Columbia RealTalk comprises in-the-wild dyadic conversation videos at 25 frames per second, with per-frame EMOCA/FLAME-style head parameters for two face tracks (Geng et al., 2023). In the official convention used here, `p0` is the LEFT participant and `p1` is the RIGHT participant. The listener to be scored is the person named in the annotation sheet, not whoever is silent.

Thirty source videos were selected. For each video a single watch window of about 60 s was defined, which is 1,500 frames at 25 fps. The windows are listed in `data/gold/annotation_sheet.csv` by YouTube clock and in `data/gold_annotations.csv` by frame index. DEV and TEST source-video identifiers do not overlap.

Official pose archives are not redistributed in the code repository. Features used for modelling are compact NumPy clips (`rotation_xyz`, `expression`) written during a streaming pass over the Hugging Face EMOCA tar archive.

### 4.3 Human labels (gold)

Three terms are used throughout this dissertation and are kept distinct. A **gold** label is the binary decision written by the annotator. A **pseudo-label** is an automatic 0/1 assignment from the frozen pose rule on an unlabelled TRAIN window. A **prediction** is the 0/1 output of a scored system on DEV or TEST. Only the first is a human judgement.

The annotation protocol uses two classes only:

| Code | Name | Role in evaluation |
| --- | --- | --- |
| `1` | Clear nod | The only gold positive |
| `0` | Unclear | Gold negative, with no claim that the clip is motionless |

A single annotator, the author, watched each 60 s window on the public RealTalk YouTube copies, attending only to the named side. Class `1` was used when a nod was judged clearly visible in that window. Class `0` was used when the motion was absent, ambiguous, or not a nod. The annotator also marked a short clock interval for the gesture, mean duration 1.1 s. That interval documents where the nod was seen. The experimental unit remains the full 60 s window, classified as a whole.

The split is 15 DEV and 15 TEST. DEV may be used to choose a rotation axis, an amplitude threshold, a CNN epoch, and a probability threshold. TEST is labelled but is scored once. It is not used to choose any of those quantities and is not used as training data.

Label counts:

| | DEV | TEST | All |
| --- | ---: | ---: | ---: |
| Windows | 15 | 15 | 30 |
| Clear nod (`1`) | 9 | 10 | 19 |
| Unclear (`0`) | 6 | 5 | 11 |
| LEFT / RIGHT (all 30) |  |  | 15 / 15 |

Two marked clock times fall outside the planned watch window and were kept as recorded: video `Ak2Bm8mfL3w` (marked 1:57 to 1:58, planned window 13:34 to 14:34) and video `Zrer1sqWzOQ` (marked 4:48 to 4:49, planned window 4:56 to 5:56). Pose was extracted from the planned frame range. This is a protocol defect, is discussed as a limitation, and was not corrected after looking at TEST scores.

There is no second annotator and therefore no inter-annotator agreement. On ethics, RealTalk licence terms were followed, the study does not attempt identity re-identification, and only derived pose clips and public video identifiers appear in the write-up.

### 4.4 Pose features (EMOCA streamed, not trained)

For each gold window, per-frame EMOCA embeddings were read for the named person. The first three pose coefficients were treated as an axis-angle rotation and converted to Euler angles in degrees with

`Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)`.

The three resulting channels are stored as `rotation_xyz`. Without further evidence they are not assumed to be anatomical pitch, yaw, and roll, so the rule search is allowed to pick any one of the three axes on DEV. Expression coefficients, padded or truncated to 20 dimensions, are stored alongside pose and are used only in ablation D.

Missing Euler samples were linearly interpolated when at least some finite samples existed in the window. A window was discarded only if fewer than 5% of frames yielded a pose. All 30 gold windows produced usable pose clips.

Eighty additional unlabelled windows, drawn from other RealTalk conversations in the same archive and not from the 30 gold video identifiers, form the weakly supervised TRAIN pool. For each of those videos the first available 60 s of pose was stored, preferring `p0` and falling back to `p1` where `p0` was missing.

The Hugging Face object `emoca.tar.gz` was never downloaded as a file. Members were read from an HTTP stream, converted to compact `.npz` clips, and discarded. EMOCA was used as a pretrained feature extractor and was neither trained nor fine-tuned in this project.

### 4.5 Frozen pose rule

The scored rule is an amplitude detector on one Euler channel, implemented in `scripts/run_full_experiment.py` as `rule_score`. The 1 to 3 Hz band-pass cycle detector in `src/rules/nod.py` is earlier code that was not used as the TEST baseline.

For a clip and a chosen axis $a \in \{0,1,2\}$:

1. Take the finite samples of `rotation_xyz[:, a]`.
2. Smooth with a Savitzky-Golay filter, window length 11 and polynomial order 2. The window is shortened on very short series so that it remains odd and at least 5.
3. Find turning points of the smoothed series.
4. For turning-point pairs separated by 5 to 50 frames, which is 0.20 to 2.00 s at 25 fps, record the absolute difference in smoothed angle.
5. Take the clip score as the largest such amplitude, or the peak-to-peak range of the smoothed series if no pair qualifies.

A clip is predicted positive if $\mathrm{score} \ge \tau$.

Axis $a$ and threshold $\tau$ were selected on DEV only. For each axis, candidate thresholds were the unique 10th to 90th percentiles of the 15 DEV scores, giving 17 quantiles. Each candidate was scored with clip-level precision, recall, and F1 against DEV gold. The pair with the highest F1, with balanced accuracy as a tie-break, was frozen:

- axis x (channel 0);
- $\tau = 16.35^\circ$, recorded as 16.3538° in `results/rule_selected_config.json`;
- DEV F1 0.86, from 9 TP, 3 FP, 3 TN, and 0 FN on DEV.

DEV F1 is a tuning diagnostic rather than a result. After this freeze, the same $(a, \tau)$ was applied once to TEST and to the 80 TRAIN windows to make pseudo-labels. TEST was not used to try other axes or thresholds.

### 4.6 Pseudo-labels (TRAIN)

The frozen rule labelled the 80 unlabelled TRAIN clips, returning 70 predicted nod and 10 predicted unclear. These pseudo-labels inherit the rule's bias toward the positive class.

No gold TEST clip and no gold DEV clip is in TRAIN, so the CNN never sees a human TEST label during learning. It does see human DEV labels after each epoch, but only to choose an epoch and a probability threshold, as described in Section 4.7. That is model selection rather than TEST tuning.

### 4.7 1D CNN (pose-based temporal classifier)

This classifier is a temporal model over pose sequences. Its only input is the EMOCA `rotation_xyz` Euler series with first differences, resampled to a fixed length, and no video frame, face crop, or pixel tensor is read at any point in training or scoring. The pixel-based systems of this study are the two VideoMAE variants of Section 4.8.

**Architecture.** Each clip is resampled to 128 time steps. Feature set C, the reported model, concatenates Euler xyz with first differences, giving 6 input channels. Inputs are standardised with the mean and standard deviation of the TRAIN pseudo set, and the same statistics are applied to DEV and TEST.

The network is a three-layer 1D CNN in PyTorch:

- `Conv1d` 6 to 32, kernel 5, padding 2, batch norm, ReLU, dropout 0.2;
- `Conv1d` 32 to 64, kernel 5, padding 2, batch norm, ReLU, dropout 0.2;
- `Conv1d` 64 to 64, kernel 3, padding 1, ReLU;
- adaptive average pool over time to a 64-D vector;
- linear 64 to 1 logit.

**Training.** Loss is binary cross-entropy with logits. The positive-class weight is $n_{\mathrm{neg}}/n_{\mathrm{pos}}$ on TRAIN, which is 10/70, so the majority nod class is down-weighted. The optimiser is Adam at learning rate $10^{-3}$, batch size 16, maximum 15 epochs, seed 42, on CPU. Training stops if DEV F1 does not improve for four consecutive epochs, and the weights with the best DEV F1 are restored.

**Threshold.** After each epoch, DEV probabilities are swept over 13 thresholds in $[0.20, 0.80]$. The threshold that maximises DEV F1, with balanced accuracy as tie-break, is stored with that epoch. For feature set C the selected epoch is 9, with DEV F1 0.89 and probability threshold 0.45. TEST probabilities are then thresholded once at 0.45.

**Ablations (RQ3).** The same training recipe was repeated with four input designs. Only TEST F1 after DEV-based epoch selection is comparable, and even then $n=15$ cannot separate close scores.

| Set | Channels | Dim. | Role |
| --- | --- | --- | --- |
| A | Euler $x$ only | 1 | Single-axis pose |
| B | Euler $xyz$ | 3 | Full rotation |
| C | $xyz$ + first differences | 6 | Reported model |
| D | C + 20 expression coefficients | 26 | Training diverged (`loss = nan`, TEST F1 0) |

Sets A to C are described in the Results chapter. Set D records that expression was tried and failed numerically. It is not interpreted as evidence that expression is uninformative.

### 4.8 VideoMAE systems (RGB comparison)

A VideoMAE fine-tune on face crops was part of the original plan (Tong et al., 2022). It was initially blocked by storage: on otter, free disk after installing CPU PyTorch was about 6.5 GB against a 25 GB home quota, and RealTalk video shards plus a VideoMAE checkpoint do not fit that remainder. Two constrained variants were ultimately run under the same frozen protocol. Full fine-tuning and the multimodal fusion of text, audio, and video remain future work.

**Input pipeline, shared by both variants.** For each of the 110 windows, comprising 80 TRAIN pseudo-labelled, 15 DEV, and 15 TEST, 16 RGB frames were fetched from the source videos via HTTP range reads, cropped to the listener face with a Haar detector, and resized to 224×224 with the `VideoMAEImageProcessorPil` preprocessing of the checkpoint. For 12 of the 110 clips no face was found and a centre-crop fallback was used. The backbone is `MCG-NJU/videomae-base` in both variants.

**Variant 1: frozen head (CPU).** The encoder was kept frozen. Token embeddings were mean-pooled to a single 768-D vector per clip, and a small MLP head was trained on the 80 rule pseudo-labels. Epoch and threshold were selected on DEV, giving best epoch 10, DEV F1 0.90, and threshold 0.40. TEST was scored once.

**Variant 2: partial fine-tune (GPU).** The checkpoint was loaded as `VideoMAEForVideoClassification` with a single logit. Patch embeddings and the first 8 encoder blocks were frozen, and the last 4 encoder blocks with `fc_norm` and the classifier were trained, which is 28.4M of 86.2M parameters. The optimiser is AdamW at 1e-5 for the backbone and 1e-4 for the head, batch size 8, `BCEWithLogitsLoss` with `pos_weight = 0.143` from the TRAIN split of 70 nod and 10 unclear, horizontal-flip augmentation on TRAIN only, and automatic mixed precision on a single RTX A4000 16 GB (lab host otter95, torch 2.13.0+cu126, transformers 5.15.1, seed 42). Because the 25 GB home quota could not hold a CUDA PyTorch stack of roughly 5 to 8 GB installed, the GPU environment lived on `/scratch`, local disk outside the quota. Preprocessing was verified bit-identical to the frozen pipeline, with maximum absolute difference 0.0. Early stopping on DEV F1 selected epoch 5, with DEV F1 0.857 and threshold 0.45. TEST was scored exactly once.

For both variants the TRAIN, DEV, and TEST splits are identical to the pose systems, and a leakage gate checking that pseudo clips are disjoint from DEV and TEST by id and video passed at startup. A scaling repeat of variant 2 used 200 frozen-rule pseudo-labels with the same DEV and TEST splits, reported in Results §5.8, and the n=80 artefacts were left untouched. The locked TEST values for these systems are frozen 0.57, fine-tuned at n=80 0.82, and scaling at n=200 0.63, in Results §5.6 to §5.8.

Training the 1D CNN on a GPU would not change the reported TEST counts. The model is small and was already trained to the early-stopping epoch on CPU.

### 4.9 Evaluation protocol and metrics

The experimental unit is one watch window. A prediction is a single 0/1 decision for that window.

From the 2×2 counts on a split,

$$
P = \frac{TP}{TP + FP}, \qquad
R = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2PR}{P + R},
$$

with the convention that an undefined ratio is 0, which occurs when there are no positive predictions. Headline metrics on TEST are precision, recall, and F1. Accuracy and balanced accuracy are reported as secondary numbers because the TEST set is imbalanced at 10 nod and 5 unclear. Accuracy was not optimised.

The headline metric of this study is clip-level F1. An event-level F1 at temporal IoU 0.30, matching time intervals one to one, is implemented in `src/metrics.py` but was not used to freeze the rule or to score TEST for these 30 windows.

TEST at $n=15$ is small. A change of one clip in the confusion matrix moves F1 by several points, so a difference of a single false positive is described as such rather than as a statistically tested improvement.

### 4.10 Reproducibility and leakage controls

- Gold CSV: `data/gold_annotations.csv`. Splits: `data/splits/gold_dev.txt`, `gold_test.txt`.
- Frozen rule: `results/rule_selected_config.json`.
- TEST scores: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/videomae_frozen_head/metrics.json`, `results/videomae_finetuned/metrics.json`. Master table: `results/tables/main_results.md`. Intervals: `results/tables/bootstrap_ci.csv`, computed from TEST-only predictions, with `results/videomae_finetuned/predictions_test.csv` (15 rows) as the canonical file for the fine-tuned model.
- Seed 42 throughout. CPU PyTorch for the pose systems and the frozen head. GPU partial fine-tune on otter95 with CUDA PyTorch on `/scratch`. No TEST-based retraining.
- Repository tests include split-leakage checks in `tests/test_invariants.py`.
- Synthetic `pilot_*` clips, where present from earlier pipeline debugging, are not RealTalk results and are not tabulated.

### 4.11 Three-second sliding-window protocol (nod)

The 60 s protocol above asks whether a nod occurs anywhere in a watch window, which is a coarse question. A second protocol therefore re-cuts the same clips into 3 s windows with a 2 s stride, giving 29 overlapping windows per 60 s clip and 435 windows per split. Event onsets and offsets were annotated by hand for all 30 gold clips; a window is positive if it overlaps an annotated nod event. Under this protocol the positive rate is 52/435 (12.0 percent) on DEV and 69/435 (15.9 percent) on TEST. Window labels are in `data/windowed_annotations/nod_windows_dev.csv` and `nod_windows_test.csv`; the DEV and TEST files are generated by separate scripts and no script reads both.

**Selection criterion.** F1 is not a safe selection criterion at this prevalence. Precision has the predicted-positive count in its denominator, so when positives are rare the penalty for a false positive is small relative to the reward for an extra true positive, and a threshold sweep that maximises F1 can drift towards predicting positive almost everywhere without that being visible in the metric. The trivial always-yes rule already reaches F1 0.274 on TEST, so an F1 close to that value cannot be distinguished from collapse. Balanced accuracy, the mean of sensitivity and specificity, weights the two classes equally and has a fixed floor of 0.5 for any constant or uninformative predictor. Balanced accuracy is therefore the selection criterion and the headline metric for this protocol, for the amplitude threshold of the pitch rule and for the decision threshold of the 1D CNN. The change was made on DEV and stated before TEST was scored, for the reason given here rather than in response to a TEST outcome. DEV PR AUC is reported alongside as a threshold-free check on whether the score ranks nod windows above non-nod windows at all, since a criterion can only choose a good operating point if the underlying ranking carries signal.

**Intervals.** The 435 windows of a split come from 15 clips and overlap by 1 s, so they are not independent observations. Confidence intervals for this protocol are 95 percent percentile intervals over 2000 bootstrap resamples of the 15 clips with replacement, taking all windows of each drawn clip. Resamples that contain only one class are discarded and counted. Resampling windows instead would understate uncertainty.

**Files.** `scripts/evaluate_windowed_nod_baselines.py` writes `results/windowed_nod/baselines_bacc/`; the earlier F1-selected run is retained unchanged at `results/windowed_nod/baselines/` for comparison. The frozen 60 s threshold is transferred without refitting from `results/rule_selected_config.json`.
