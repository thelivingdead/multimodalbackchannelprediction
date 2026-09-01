# Methods 
Numbers here are protocol and implementation details, not TEST headlines. The locked TEST table lives in `results_chapter_draft.md`.

Suggested chapter number: **4** (if Data is a separate chapter) or **4–5** (if Data and Methods are combined). Figure files are under `dissertation-behaviour-recognition/figures/`.

---

## 4. Methods

This chapter describes the study that was actually run. The task is **clip-level recognition of a clear listener head nod** on Columbia RealTalk (Geng et al., 2023). Two systems are compared on a frozen TEST set of 15 windows: (i) a deterministic pose-amplitude rule whose axis and threshold are chosen on DEV only, and (ii) a small **pose-based temporal classifier** — a 1D convolutional network over the EMOCA `rotation_xyz` pose sequence, with no RGB input — trained on 80 **pseudo-labels** produced by that frozen rule. Human labels are never replaced by model outputs. Two RGB systems based on VideoMAE were additionally run under the same frozen protocol (Section 4.8): a frozen encoder with a trained head, and a partial fine-tune of the last four encoder blocks on a lab GPU. Full fine-tuning was not attempted, for the storage reasons given in Section 4.8.

### 4.1 Scope of the submitted study

The original project notes outlined a seven-class backchannel taxonomy (nod, shake, tilt, lean forward, lean back, eyebrow raise, neutral) and a multimodal predictor that would fuse text, audio, video, and FLAME/EMOCA sequences (see `docs/archive/02_research_methodology_and_roadmap.md`). That plan is **not** the submitted experiment.

The submitted study is narrower for two reasons that must be stated in the dissertation. First, reliable human labels were obtained only for a binary decision: **clear nod** versus **unclear**. Second, the lab account used for pose extraction (otter) has a home quota of about 25 GB. After a CPU PyTorch install, free space was about 6.5 GB, which is not enough to store RealTalk video shards together with a VideoMAE checkpoint. Official EMOCA/FLAME pickles were **streamed** from Hugging Face and were not written to disk as `emoca.tar.gz`. EMOCA was not trained.

The research questions that this Methods chapter is designed to answer are therefore:

1. **RQ1.** After the pose rule is frozen on 15 DEV windows, what clip-level precision, recall, and F1 does it obtain on 15 unseen TEST windows?
2. **RQ2.** Does a 1D CNN trained only on 80 automatic rule labels improve those TEST scores?
3. **RQ3.** At this sample size, do extra pose channels (full Euler xyz, first differences, expression coefficients) change TEST F1?

Two constrained VideoMAE variants (frozen head; partial fine-tune) **were** run and are reported as results (Section 4.8). **Full** VideoMAE fine-tuning, seven-class typing, and event-level F1 at temporal IoU 0.30 remain **future work**. They are not reported as results.

### 4.2 Dataset and listener convention

Columbia RealTalk comprises in-the-wild dyadic conversation videos at **25 frames per second**, with per-frame EMOCA/FLAME-style head parameters for two face tracks (Geng et al., 2023). In the official convention used here, **p0 is the LEFT participant and p1 is the RIGHT participant**. The listener to be scored is the person named in the annotation sheet, not “whoever is silent”.

Thirty source videos were selected. For each video a single watch window of about 60 s was defined (1,500 frames at 25 fps). The windows are listed in `data/gold/annotation_sheet.csv` (YouTube clock) and in `data/gold_annotations.csv` (frame indices). DEV and TEST source-video identifiers do not overlap.

Official pose archives are not redistributed in the code repository. Features used for modelling are compact NumPy clips (`rotation_xyz`, `expression`) written during a streaming pass over the Hugging Face EMOCA tar archive.

### 4.3 Human labels (gold)

**Definition used throughout this dissertation.** A **gold** (or **human-labelled**, **manually annotated**) label is the binary decision written by the annotator. It is not a detector output. **Pseudo-labels** are automatic 0/1 assignments from the frozen pose rule on unlabelled TRAIN windows. **Predictions** are the 0/1 outputs of the rule or the CNN on DEV or TEST. Predictions are never called gold.

The annotation protocol uses two classes only:

| Code | Name | Role in evaluation |
| --- | --- | --- |
| `1` | Clear nod | The only gold **positive** |
| `0` | Unclear | Gold **negative** (no claim that the clip is motionless) |

A single annotator (the author) watched each 60 s window on the public RealTalk YouTube copies, attending only to the named side (LEFT or RIGHT). Class `1` was used when a nod was judged clearly visible in that window. Class `0` was used when the motion was absent, ambiguous, or not a nod. The annotator also marked a short clock interval for the gesture (mean duration 1.1 s). That short interval documents *where* the nod was seen; **the experimental unit remains the full ~60 s window**, classified as a whole.

The split is **15 DEV / 15 TEST**. DEV may be used to choose a rotation axis, an amplitude threshold, a CNN epoch, and a probability threshold. TEST is labelled but is **scored once**. TEST is not used to choose any of those quantities, and it is not used as training data.

Label counts:

| | DEV | TEST | All |
| --- | ---: | ---: | ---: |
| Windows | 15 | 15 | 30 |
| Clear nod (`1`) | 9 | 10 | 19 |
| Unclear (`0`) | 6 | 5 | 11 |
| LEFT / RIGHT (all 30) |  |  | 15 / 15 |

Two marked clock times fall outside the planned watch window and were **kept as recorded**: video `Ak2Bm8mfL3w` (marked 1:57–1:58; planned window 13:34–14:34) and video `Zrer1sqWzOQ` (marked 4:48–4:49; planned window 4:56–5:56). Pose was extracted from the planned frame range. This is a protocol defect and is discussed as a limitation; it is not corrected after looking at TEST scores.

There is no second annotator and therefore **no inter-annotator agreement**. Ethics: RealTalk licence terms were followed; the study does not attempt identity re-identification; only derived pose clips and public video identifiers are used in the write-up.

### 4.4 Pose features (EMOCA streamed, not trained)

For each gold window, per-frame EMOCA embeddings were read for the named person (`p0` or `p1`). The first three pose coefficients were treated as an axis-angle rotation and converted to Euler angles in degrees with

`Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)`.

The three resulting channels are stored as `rotation_xyz`. They are **not** assumed, without further evidence, to be anatomical pitch, yaw, and roll. The rule search is allowed to pick any one of the three axes on DEV. Expression coefficients (padded or truncated to 20 dimensions) are stored alongside pose but are used only in ablation D, which diverged and is not a reported result.

Missing Euler samples were linearly interpolated when at least some finite samples existed in the window. A window was discarded only if fewer than 5% of frames yielded a pose. All 30 gold windows produced usable pose clips.

Eighty additional unlabelled windows, drawn from other RealTalk conversations in the same archive (not the 30 gold video identifiers), form the weakly supervised **TRAIN** pool. For each of those videos the first available ~60 s of pose was stored, preferring `p0` and falling back to `p1` if `p0` was missing.

The Hugging Face object `emoca.tar.gz` was **not** downloaded as a file. Members were read from an HTTP stream, converted to compact `.npz` clips, and discarded. That choice is a storage constraint, not a claim that EMOCA was fine-tuned in this project.

### 4.5 Frozen pose rule

The scored rule is an **amplitude** detector on one Euler channel. It is implemented in `scripts/run_full_experiment.py` (`rule_score`). It is **not** the older 1–3 Hz band-pass cycle detector in `src/rules/nod.py`; that code was not the TEST baseline.

For a clip and a chosen axis \(a \in \{0,1,2\}\):

1. Take the finite samples of `rotation_xyz[:, a]`.
2. Smooth with a Savitzky–Golay filter (window length 11, polynomial order 2; the window is shortened on very short series so that it remains odd and at least 5).
3. Find turning points of the smoothed series.
4. For turning-point pairs whose separation is between **5 and 50 frames** (0.20–2.00 s at 25 fps), record the absolute difference in smoothed angle.
5. The clip **score** is the largest such amplitude, or the peak-to-peak range of the smoothed series if no pair qualifies.

A clip is predicted positive if \(\mathrm{score} \ge \tau\).

Axis \(a\) and threshold \(\tau\) were selected **on DEV only**. For each axis, candidate thresholds were the unique 10th-to-90th percentiles of the 15 DEV scores (17 quantiles). Each candidate was scored with clip-level precision, recall, and F1 against DEV gold. The pair with the highest F1, with balanced accuracy as a tie-break, was frozen:

- axis **x** (channel 0);
- \(\tau = 16.35^\circ\) (16.3538° in `results/rule_selected_config.json`);
- DEV F1 = 0.86 (9 TP, 3 FP, 3 TN, 0 FN on DEV).

**DEV F1 is a tuning diagnostic. It is not a result.** After this freeze, the same \((a,\tau)\) was applied to TEST **once** and to the 80 TRAIN windows (to make pseudo-labels). TEST was not used to try other axes or thresholds.

### 4.6 Pseudo-labels (TRAIN)

The frozen rule labelled the 80 unlabelled TRAIN clips: **70 predicted nod**, **10 predicted unclear**. Those 0/1 assignments are **pseudo-labels**. They inherit the rule’s bias toward the positive class. They are not human labels and are not gold.

No gold TEST clip is in TRAIN. No gold DEV clip is in TRAIN. The CNN therefore never sees human TEST labels during learning. It does see human DEV labels **after** each epoch, but only to choose an epoch and a probability threshold (Section 4.7). That is model selection, not TEST tuning.

### 4.7 1D CNN (pose-based temporal classifier)

**What the network is, and is not.** The classifier in this section is a **temporal model over pose sequences**. Its only input is the EMOCA `rotation_xyz` Euler series (plus first differences), resampled to a fixed length. It is **not** an RGB vision model: no video frame, face crop, or pixel tensor is read at any point in training or scoring. The pixel-based systems of this study are the two VideoMAE variants of Section 4.8.

**Architecture.** Each clip is resampled to a fixed length of **128** time steps. Feature set **C** (the reported model) concatenates Euler xyz with first differences, giving **6** input channels. Inputs are standardised with the mean and standard deviation of the TRAIN (pseudo) set, then the same statistics are applied to DEV and TEST.

The network is a three-layer 1D CNN in PyTorch:

- `Conv1d` 6→32, kernel 5, padding 2; batch norm; ReLU; dropout 0.2;
- `Conv1d` 32→64, kernel 5, padding 2; batch norm; ReLU; dropout 0.2;
- `Conv1d` 64→64, kernel 3, padding 1; ReLU;
- adaptive average pool over time to a 64-D vector;
- linear 64→1 logit.

**Training.** Loss is binary cross-entropy with logits. The positive-class weight is \(n_{\mathrm{neg}}/n_{\mathrm{pos}}\) on TRAIN (10/70), so the majority nod class is down-weighted. Optimiser: Adam, learning rate \(10^{-3}\). Batch size 16. Maximum 15 epochs. Seed 42. Device: **CPU** (`torch`). Early stopping: training stops if DEV F1 does not improve for four consecutive epochs. The weights with the best DEV F1 are restored.

**Threshold.** After each epoch, DEV probabilities are swept over 13 thresholds in \([0.20, 0.80]\). The threshold that maximises DEV F1 (balanced accuracy as tie-break) is stored with that epoch. For feature set C the selected epoch is **9**, with DEV F1 0.89 and probability threshold **0.45**. **Those DEV figures are not headlines.** TEST probabilities are then thresholded **once** at 0.45.

**Ablations (RQ3).** The same training recipe was repeated with four input designs. Only TEST F1 after DEV-based epoch selection is comparable; even then, \(n=15\) cannot separate close scores.

| Set | Channels | Dim. | Role |
| --- | --- | --- | --- |
| A | Euler \(x\) only | 1 | Single-axis pose |
| B | Euler \(xyz\) | 3 | Full rotation |
| C | \(xyz\) + first differences | 6 | **Reported model** |
| D | C + 20 expression coefficients | 26 | Attempted; **training diverged** (`loss = nan`, TEST F1 0). **Not a result.** |

Sets A–C are described in the Results chapter. Set D is mentioned only to record that expression was tried and failed numerically. It is not interpreted as evidence that expression is uninformative.

### 4.8 VideoMAE systems (RGB comparison)

A VideoMAE fine-tune on face crops was part of the original plan (Tong et al., 2022). It was initially **blocked by storage**: on otter, free disk after installing CPU PyTorch was about **6.5 GB** against a **25 GB** home quota, and RealTalk video shards plus a VideoMAE checkpoint do not fit that remainder. Two constrained variants were ultimately run under the same frozen protocol; **full** fine-tuning and the multimodal fusion of text, audio, and video remain future work.

**Input pipeline (shared by both variants).** For each of the 110 windows (80 TRAIN pseudo-labelled, 15 DEV, 15 TEST), 16 RGB frames were fetched from the source videos via HTTP range reads, cropped to the listener face (Haar detector; for **12 of 110 clips** no face was found and a centre-crop fallback was used), and resized to 224×224 with the `VideoMAEImageProcessorPil` preprocessing of the checkpoint. The backbone is `MCG-NJU/videomae-base` in both variants.

**Variant 1 — frozen head (CPU).** The encoder was kept **frozen**; token embeddings were mean-pooled to a single **768-D** vector per clip, and a small **MLP head** was trained on the 80 rule pseudo-labels. Epoch and threshold were selected on DEV (best epoch 10, DEV F1 0.90, threshold 0.40); TEST was scored once.

**Variant 2 — partial fine-tune (GPU).** The checkpoint was loaded as `VideoMAEForVideoClassification` with a single logit. Patch embeddings and the first 8 encoder blocks were frozen; the **last 4 encoder blocks plus `fc_norm` and the classifier** were trained (**28.4M of 86.2M parameters**). Optimiser AdamW with learning rates **1e-5 (backbone)** and **1e-4 (head)**; batch size 8; `BCEWithLogitsLoss` with `pos_weight = 0.143` (TRAIN 70 nod / 10 unclear); horizontal-flip augmentation on TRAIN only; automatic mixed precision on a single **RTX A4000 16 GB** (lab host otter95; torch 2.13.0+cu126, transformers 5.15.1, seed 42). Because the 25 GB home quota could not hold a CUDA PyTorch stack (~5–8 GB installed), the GPU environment lived on **`/scratch`**, local disk outside the quota. Preprocessing was verified **bit-identical** to the frozen pipeline (max abs diff 0.0). Early stopping on DEV F1 selected epoch **5** (DEV F1 0.857, threshold 0.45); TEST was scored exactly once.

For both variants the TRAIN/DEV/TEST splits are identical to the pose systems, and a leakage gate (pseudo clips disjoint from DEV and TEST by id and video) passed at startup. A scaling repeat of variant 2 used **200** frozen-rule pseudo-labels with the same DEV/TEST (Results §5.8); the n=80 artefacts were left untouched. The dissertation must not invent scores beyond the locked TEST values (frozen 0.57; fine-tuned n=80 **0.82**; scaling n=200 **0.63**; Results §5.6–5.8).

Training the 1D CNN on a GPU would not change the reported TEST counts: the model is small and was already trained to the early-stopping epoch on CPU.

### 4.9 Evaluation protocol and metrics

The experimental unit is one watch window. A prediction is a single 0/1 decision for that window.

From the 2×2 counts on a split,

\[
\mathrm{P} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \quad
\mathrm{R} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \quad
\mathrm{F1} = \frac{2\mathrm{PR}}{\mathrm{P}+\mathrm{R}},
\]

with the convention that a undefined ratio is 0 (no positive predictions). Headline metrics on TEST are **precision, recall, and F1**. Accuracy and balanced accuracy are reported as secondary numbers because the TEST set is imbalanced (10 nod, 5 unclear). Accuracy is not optimised and is not the title result.

**This protocol is clip-level.** An event-level F1 at temporal IoU 0.30 (one-to-one matching of time intervals) is implemented in `src/metrics.py` but was **not** the metric used to freeze the rule or to score TEST for these 30 windows. Do not write “F1 @ IoU 0.30” as the headline of this study.

TEST \(n=15\) is small. A change of one clip in the confusion matrix moves F1 by several points. Differences of a single false positive must be described as such, not as a statistically tested improvement.

### 4.10 Reproducibility and leakage controls

- Gold CSV: `data/gold_annotations.csv`. Splits: `data/splits/gold_dev.txt`, `gold_test.txt`.
- Frozen rule: `results/rule_selected_config.json`.
- TEST scores: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/videomae_frozen_head/metrics.json`, `results/videomae_finetuned/metrics.json`; master table `results/tables/main_results.md`; CIs `results/tables/bootstrap_ci.csv` (TEST-only predictions; for the fine-tuned model the canonical file is `results/videomae_finetuned/predictions_test.csv`, 15 rows).
- Seed 42; CPU PyTorch (pose systems, frozen head); GPU partial fine-tune on otter95 with CUDA PyTorch on `/scratch`; no TEST-based retraining.
- Repository tests include split-leakage checks (`tests/test_invariants.py`).
- Synthetic `pilot_*` clips, if present from earlier pipeline debugging, are **not** RealTalk results and are not tabulated.

---

**End of Methods paste.** Continue with Results (`results_chapter_draft.md`).
