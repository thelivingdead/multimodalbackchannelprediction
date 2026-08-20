# Discussion, limitations, and conclusion (paste-ready)

Paste after Results. Do not put DEV F1 in this chapter as a finding. Do not invent VideoMAE or ablation-D scores.

---

## 8. Discussion

### 8.1 What the TEST comparison shows

On the locked TEST set (\(n=15\); 10 human-labelled nods, 5 unclear) the frozen pose rule obtained precision 0.64, recall 0.70, and F1 **0.67** (TP 7, FP 4, TN 1, FN 3). The pose-based 1D CNN — a temporal classifier over EMOCA `rotation_xyz` sequences, not an RGB vision model — trained on 80 pseudo-labels obtained precision 0.70, recall 0.70, and F1 **0.70** (TP 7, FP 3, TN 2, FN 3).

Two facts follow immediately. First, **neither system is a solved detector**. Both miss 3 of 10 gold nods and both still raise false alarms on unclear windows. Second, the CNN does **not** transform the rule. Recall is identical (7/10). The F1 gain is exactly **one extra true negative** (false positives 4 → 3). Balanced accuracy moves from 0.45 to 0.55; accuracy from 0.53 to 0.60. Those secondary figures are consistent with one clip, not with a new capability.

With 15 TEST windows, a one-clip change is within ordinary sampling noise. The dissertation should therefore **not** claim that weak supervision “significantly outperforms” the rule, and it should **not** claim that the CNN failed. The defensible statement is that a small temporal network, trained only on automatic labels, **matched the rule’s recall and reduced false positives by one clip**.

DEV F1 was 0.86 for the rule and 0.89 for the CNN. Those numbers describe the set used to choose \(\tau\) and the epoch. They are expected to be optimistic. Reporting them as generalisation would be a leakage error.

### 8.2 Answers to the research questions

**RQ1 (frozen rule).** A peak-to-peak amplitude rule on Euler \(x\), with \(\tau=16.35^\circ\) frozen on DEV, detects most of the human-labelled nods on TEST (recall 0.70) but is not precise (precision 0.64). The DEV search favoured high recall on the tuning set (DEV recall 1.00). That bias is visible on TEST: four of five gold-negatives are still called nods by the rule.

**RQ2 (pseudo-label CNN).** Training on 70 automatic nods and 10 automatic negatives did not collapse. The CNN is usable, but it largely **reproduces the rule**. That is the expected behaviour of weak supervision when the labelling function is also the main baseline (Ratner et al., 2017): the student cannot invent a decision boundary that the teacher never used, except insofar as the 1D filters smooth or re-weight the same Euler traces.

**RQ3 (features).** Sets A (x only), B (xyz), and C (xyz + derivatives) all give TEST F1 of 0.70 at two-decimal reporting. Set B trades precision for recall (precision 0.62, recall 0.80) but does not change the headline. At \(n=15\) these variants are **not distinguishable**. Set D (expression concatenated) diverged (`loss = nan`) and yields F1 0 by arithmetic, not by a scientific comparison. Expression is therefore **untested**, not disproved.

### 8.3 Error pattern

Shared errors are more informative than the one-clip F1 gap.

Both systems are **false positive** on `gold_017` (`oQNpe8uwSUA`, unclear) and `gold_022` (`jn_3yDP58Ik`, unclear), and also on `gold_028` (`N-6L1u42cnw`, unclear). Both are **false negative** on `gold_018` (`Zrer1sqWzOQ`, labelled nod) and `gold_026` (`G6tLY8FiheE`, labelled nod).

The CNN differs from the rule on three TEST clips only:

- `gold_023` (`zS-xXIiLrWw`, unclear): rule false positive, CNN true negative (this is the extra correct rejection that moves F1);
- `gold_029` (`PDd6qEv0_7c`, nod): rule true positive, CNN miss;
- `gold_030` (`ktR3_bXoxaE`, nod): rule miss, CNN true positive.

`gold_024` (`PM3oaJMiDd4`, unclear) is a true negative for **both** systems. The CNN therefore **swaps** one false negative for another while picking up one true negative. That is not a systematic win on nods; it is a slight shift on negatives.

A protocol issue likely contributes to at least one shared miss. For `gold_018`, the annotator marked a nod at 4:48–4:49, which sits **outside** the planned watch window 4:56–5:56 from which pose was extracted. The gold label is still `1`. A clip-level detector that never sees the marked instant cannot be expected to recover it. This does not “explain away” all false negatives (`gold_026` is inside its window), but it shows that the gold protocol and the feature window are not perfectly aligned.

False positives on large Euler amplitude are consistent with the rule definition: any strong rotation on axis \(x\) counts, including non-nod motion (speech-related head movement, tracking jumps, posture shifts). The CNN, trained on the same scores as labels, inherits that confusion.

### 8.4 Relation to prior work

Lin et al. (2025) report strong tri-modal F1 for **binary backchannel** on MM-F2F. Those figures are **not comparable** to the present TEST F1. Their task is Keep/Turn/Backchannel on a different corpus with word-level labels and a much larger set. This study asks a different question—whether a **clear nod** is present in a 60 s RealTalk window—using 30 human labels and pose-only input. Citing MM-F2F 0.91 as if it were a baseline for this table would be misleading.

EMOCA (Danecek et al., 2022) and FLAME (Li et al., 2017) supply the pose parameterisation. This dissertation **uses** official EMOCA outputs; it does not retrain EMOCA and does not contribute a new 3D face method.

Weak supervision is used here in the simplest form: one labelling function, applied once, then distilled into a small network. It is closer to a frozen heuristic teacher than to a full Snorkel pipeline with many labelling functions and generative label aggregation (Ratner et al., 2017).

### 8.5 Scope reduction relative to the proposal

The submitted dissertation is deliberately narrower than the proposal-era plan, and the narrowing is reported openly rather than treated as failure. What was executed is a binary, pose-only study: thirty human-labelled RealTalk windows; a frozen EMOCA-pose amplitude rule (TEST precision 0.64, recall 0.70, F1 0.67; TP 7, FP 4, TN 1, FN 3 on \(n=15\)); and a pose-based temporal 1D CNN over EMOCA `rotation_xyz` sequences — not an RGB vision model — trained on eighty pseudo-labels issued by that rule (TEST precision 0.70, recall 0.70, F1 0.70; TP 7, FP 3, TN 2, FN 3). TEST was scored exactly once; the DEV F1 values (0.86 for the rule, 0.89 for the CNN) are tuning diagnostics, not results. One feature ablation (set D, expression coefficients) diverged during training and is omitted rather than interpreted. VideoMAE, multimodal fusion, and seven-class typing were not run, for the storage reasons given in Methods; they are planned future work, and no VideoMAE, fusion, or macro-F1 number appears anywhere in this dissertation.

The proposal-era documents describe seven backchannel types, VideoMAE, and multimodal fusion. The submitted evidence is a **binary pose study on 30 windows**. That reduction should be written as a deliberate, resource-aware narrowing, not as a hidden change of topic. The scientific contribution that can be defended is: a documented human protocol, a frozen rule with a one-shot TEST score, and a weakly supervised CNN that does not leak TEST into training. The contribution that **cannot** be defended is a multimodal 7-class system.

---

## 9. Limitations

The following limitations are part of the result, not an afterthought.

1. **Sample size.** TEST has 15 windows. One clip changes F1 by several points. No bootstrap interval or McNemar test is reported; none would be persuasive at this \(n\).

2. **Single annotator.** Labels are human-labelled by the author only. There is no Cohen’s \(\kappa\) or second pass. Class `0` (“unclear”) mixes true non-nods with difficult nods; that mixture inflates false-positive cost if some `0`s are actually nods.

3. **Clip-level metric.** Precision, recall, and F1 are computed on 30 binary window labels. Event F1 at IoU 0.30 was not the protocol. Temporal localisation of the nod inside the minute is not scored.

4. **Pseudo-label imbalance.** TRAIN is 70 nod / 10 unclear according to the rule. The CNN is trained on a teacher that already over-predicts nods. Class weighting (`pos_weight = 10/70`) mitigates but does not remove that bias.

5. **Pose convention.** Euler \(x\) was selected empirically. It is not independently verified as anatomical pitch on every clip. Camera angle and EMOCA tracking error are unquantified beyond a valid-frame ratio.

6. **Two off-window annotations.** `Ak2Bm8mfL3w` (DEV) and `Zrer1sqWzOQ` (TEST) have marked times outside the extracted window. Features and labels can disagree by construction.

7. **Expression ablation.** Set D diverged. No conclusion about facial expression is licensed.

8. **No pixel model.** VideoMAE was not trained, because otter had about 6.5 GB free after CPU PyTorch under a 25 GB quota, and video plus a VideoMAE checkpoint do not fit. EMOCA was streamed, not trained; `emoca.tar.gz` was never saved. These are missing experiments, not failed TEST runs. They belong in Future work.

9. **CPU-only learning.** The 1D CNN is small; TEST counts would not change on one GPU. GPU access was not the bottleneck; disk quota was.

10. **In-the-wild dyads.** RealTalk is unconstrained video. Lighting, occlusion, and who is the “listener” can all corrupt both pose and human judgement.

---

## 10. Conclusion

This dissertation asked whether a pose-based detector can recognise **clear listener nods** in Columbia RealTalk when only 30 windows are human-labelled and TEST is never used for tuning.

A Savitzky–Golay amplitude rule, frozen on DEV at axis \(x\) and \(16.35^\circ\), reached TEST precision 0.64, recall 0.70, and F1 **0.67**. A pose-based 1D CNN — a temporal classifier over `rotation_xyz`, with no RGB input — trained on 80 pseudo-labels from that same rule reached TEST precision 0.70, recall 0.70, and F1 **0.70**. The CNN matched the rule on nods (7/10) and rejected one additional unclear clip. That is a modest, one-clip difference on a 15-window test, not a demonstration that learned models supersede the heuristic.

The study’s defensible contributions are a documented 1/0 annotation protocol, a leakage-controlled split, streamed (not retrained) EMOCA pose, and two locked TEST confusion matrices. The study does not deliver seven-class backchannel typing, multimodal fusion, or VideoMAE. Those items remain **future work** for a machine with enough disk to hold video and a pretrained vision transformer.

A useful next experiment, if more than 48 hours and more than 6.5 GB were available, would be to enlarge the human TEST set, add a second annotator, score events at IoU 0.30, and only then consider a pixel model. Until those steps are taken, the supported headline remains: **on 15 held-out RealTalk windows, a frozen pose rule and a pseudo-label pose CNN both sit near F1 0.7**.

---

## 11. Future work (short; optional subsection)

- Increase gold \(n\) and measure inter-annotator agreement.
- Align the marked nod interval with the pose window; drop or relabel off-window cases *before* freezing a new TEST (do not relabel the present TEST after seeing scores).
- Report event-level F1 at IoU 0.30 on time intervals, in addition to clip-level F1.
- Retrain only if a new TEST is collected; do not retune on the present 15.
- VideoMAE or another video backbone **if and only if** the video shards can be reached without exhausting the 25 GB quota. The authorised next lab step is read-only: probe whether the Hugging Face video shards honour HTTP byte ranges and, only on a `206 Partial Content` answer, build a `video_id → shard` index so that 16-frame face crops can be streamed without saving multi-GB shards. The only feasible model variant on this quota is a frozen CPU VideoMAE feature extractor with a small trained head — no fine-tuning. Any VideoMAE F1 will be reported only if actually measured on this same held-out TEST set. Do not start that job on otter as it stands.
- Multi-class head gestures only after a pose rule for each class has a human-checked TEST score.

**Otter should stay idle until after submission.** No further training is required for this document.
