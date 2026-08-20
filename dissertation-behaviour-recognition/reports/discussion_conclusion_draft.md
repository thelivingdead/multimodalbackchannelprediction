# Discussion, limitations, and conclusion (paste-ready)

Paste after Results. Do not put DEV F1 in this chapter as a finding. Do not invent ablation-D scores. VideoMAE numbers are the locked TEST values only (frozen head 0.57; fine-tuned 0.82) — no others exist.

---

## 8. Discussion

### 8.1 What the TEST comparison shows

On the locked TEST set (\(n=15\); 10 human-labelled nods, 5 unclear) the frozen pose rule obtained precision 0.64, recall 0.70, and F1 **0.67** (TP 7, FP 4, TN 1, FN 3). The pose-based 1D CNN — a temporal classifier over EMOCA `rotation_xyz` sequences, not an RGB vision model — trained on 80 pseudo-labels obtained precision 0.70, recall 0.70, and F1 **0.70** (TP 7, FP 3, TN 2, FN 3).

Two facts follow immediately. First, **neither system is a solved detector**. Both miss 3 of 10 gold nods and both still raise false alarms on unclear windows. Second, the CNN does **not** transform the rule. Recall is identical (7/10). The F1 gain is exactly **one extra true negative** (false positives 4 → 3). Balanced accuracy moves from 0.45 to 0.55; accuracy from 0.53 to 0.60. Those secondary figures are consistent with one clip, not with a new capability.

With 15 TEST windows, a one-clip change is within ordinary sampling noise. The dissertation should therefore **not** claim that weak supervision “significantly outperforms” the rule, and it should **not** claim that the CNN failed. The defensible statement is that a small temporal network, trained only on automatic labels, **matched the rule’s recall and reduced false positives by one clip**.

DEV F1 was 0.86 for the rule and 0.89 for the CNN. Those numbers describe the set used to choose \(\tau\) and the epoch. They are expected to be optimistic. Reporting them as generalisation would be a leakage error.

Two RGB systems ran under the same protocol (Results §5.6–5.7). The frozen VideoMAE head scored TEST F1 **0.57** — below both pose systems — while the partially fine-tuned VideoMAE (last 4 encoder blocks, GPU) scored TEST F1 **0.82** (TP 9, FP 3, TN 2, FN 1), the highest point estimate of the five systems. The frozen-vs-fine-tuned pair, on identical inputs, splits, and pseudo-labels, indicates that **task adaptation of the video backbone, not the RGB input alone, drove the gain**. The same statistical caution applies with full force: at \(n=15\) the fine-tuned 95% CI [0.60, 0.96] overlaps the pose CNN's [0.40, 0.89], the rule's [0.35, 0.87], and the frozen head's [0.24, 0.75], so **no pairwise difference is statistically significant**. The dissertation should claim the highest point estimate, not a proven superiority.

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

The submitted dissertation is deliberately narrower than the proposal-era plan, and the narrowing is reported openly rather than treated as failure. What was executed is a binary study: thirty human-labelled RealTalk windows; a frozen EMOCA-pose amplitude rule (TEST precision 0.64, recall 0.70, F1 0.67; TP 7, FP 4, TN 1, FN 3 on \(n=15\)); and a pose-based temporal 1D CNN over EMOCA `rotation_xyz` sequences — not an RGB vision model — trained on eighty pseudo-labels issued by that rule (TEST precision 0.70, recall 0.70, F1 0.70; TP 7, FP 3, TN 2, FN 3). TEST was scored exactly once; the DEV F1 values (0.86 for the rule, 0.89 for the CNN) are tuning diagnostics, not results. One feature ablation (set D, expression coefficients) diverged during training and is omitted rather than interpreted. Two constrained VideoMAE systems were run despite the storage constraint — a frozen head (TEST F1 0.57) and a partial fine-tune on a lab GPU with CUDA PyTorch on `/scratch` (TEST F1 0.82) — but **full** fine-tuning, multimodal fusion, and seven-class typing were not run; they remain planned future work, and no fusion or macro-F1 number appears anywhere in this dissertation.

The proposal-era documents describe seven backchannel types, full VideoMAE modelling, and multimodal fusion. The submitted evidence is a **binary study on 30 windows**: two pose systems plus two constrained VideoMAE variants. That reduction should be written as a deliberate, resource-aware narrowing, not as a hidden change of topic. The scientific contribution that can be defended is: a documented human protocol, a frozen rule with a one-shot TEST score, a weakly supervised CNN that does not leak TEST into training, and a controlled frozen-vs-fine-tuned VideoMAE contrast under the same protocol. The contribution that **cannot** be defended is a multimodal 7-class system.

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

8. **Constrained pixel modelling.** Otter had about 6.5 GB free after CPU PyTorch under a 25 GB quota, and video plus a full VideoMAE training stack do not fit. What was run: a frozen VideoMAE head (TEST F1 0.57) and a partial fine-tune of the last 4 encoder blocks on an RTX A4000 with CUDA PyTorch installed on `/scratch`, outside the quota (TEST F1 0.82). Full fine-tuning of all 86.2M parameters and a larger pseudo-label pool remain future work. EMOCA was streamed, not trained; `emoca.tar.gz` was never saved.

9. **Small-sample RGB comparison.** The fine-tuned VideoMAE F1 of 0.82 is the highest point estimate, but at \(n=15\) its 95% CI [0.60, 0.96] overlaps every other system's; no pairwise difference is statistically significant. An earlier bootstrap over all 110 predictions was train-contaminated and was corrected to the 15-row TEST-only file (`results/videomae_finetuned/predictions_test.csv`) before reporting. For the pose CNN, TEST counts would not change on one GPU; the model is small.

10. **In-the-wild dyads.** RealTalk is unconstrained video. Lighting, occlusion, and who is the “listener” can all corrupt both pose and human judgement.

---

## 10. Conclusion

This dissertation asked whether a pose-based detector can recognise **clear listener nods** in Columbia RealTalk when only 30 windows are human-labelled and TEST is never used for tuning.

A Savitzky–Golay amplitude rule, frozen on DEV at axis \(x\) and \(16.35^\circ\), reached TEST precision 0.64, recall 0.70, and F1 **0.67**. A pose-based 1D CNN — a temporal classifier over `rotation_xyz`, with no RGB input — trained on 80 pseudo-labels from that same rule reached TEST precision 0.70, recall 0.70, and F1 **0.70**. The CNN matched the rule on nods (7/10) and rejected one additional unclear clip. That is a modest, one-clip difference on a 15-window test, not a demonstration that learned models supersede the heuristic.

The study’s defensible contributions are a documented 1/0 annotation protocol, a leakage-controlled split, streamed (not retrained) EMOCA pose, four locked TEST confusion matrices across pose and RGB systems, and a controlled frozen-vs-fine-tuned VideoMAE contrast. The study does not deliver seven-class backchannel typing, multimodal fusion, or full VideoMAE fine-tuning. Those items remain **future work**.

A useful next experiment would be to enlarge the human TEST set, add a second annotator, score events at IoU 0.30, and only then fully fine-tune a video backbone on a larger pseudo-label pool. Until those steps are taken, the supported headline remains: **on 15 held-out RealTalk windows, all systems sit between F1 0.57 and 0.82, with the fine-tuned VideoMAE highest at 0.82 — and every 95% interval overlapping, so the ordering is a point-estimate ranking, not a tested one**.

---

## 11. Future work (short; optional subsection)

- Increase gold \(n\) and measure inter-annotator agreement.
- Align the marked nod interval with the pose window; drop or relabel off-window cases *before* freezing a new TEST (do not relabel the present TEST after seeing scores).
- Report event-level F1 at IoU 0.30 on time intervals, in addition to clip-level F1.
- Retrain only if a new TEST is collected; do not retune on the present 15.
- **Full** VideoMAE fine-tuning (all 86.2M parameters) and a larger pseudo-label pool, building on the partial fine-tune reported here (frozen head 0.57 → partial fine-tune 0.82 on identical splits). Any further VideoMAE F1 will be reported only if actually measured on a held-out TEST set under the same protocol.
- Multi-class head gestures only after a pose rule for each class has a human-checked TEST score.

**Otter should stay idle until after submission.** No further training is required for this document.
