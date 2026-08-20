# Abstract, introduction, and literature (paste-ready)

Keep this shorter than Methods/Results if time is scarce. Do not import 7-class or VideoMAE **results**. You may describe them as the original plan.

Harvard-style citations: see `references_harvard.md`.

---

## Abstract

Listener head nods are a common visual backchannel in face-to-face talk, but public dyadic video corpora such as Columbia RealTalk (Geng et al., 2023) do not ship nod labels. This MSc dissertation studies **clip-level recognition of a clear nod** on RealTalk using official EMOCA/FLAME head-pose parameters. Thirty one-minute windows were **human-labelled** by a single annotator into two classes only: **1 = clear nod** (the only gold positive) and **0 = unclear**. The split is 15 development (DEV) and 15 test (TEST) windows, with no shared source videos. DEV may be used to choose a rule axis, a threshold, and a network epoch. TEST is scored **once** and is never used for those choices.

A Savitzky–Golay amplitude rule on one Euler channel of the EMOCA pose sequence was frozen on DEV (axis \(x\), threshold \(16.35^\circ\)). On TEST (\(n=15\)) the rule obtained precision 0.64, recall 0.70, and F1 **0.67** (7 true positives, 4 false positives, 1 true negative, 3 false negatives). The same frozen rule labelled 80 further unlabelled windows (70 predicted nod, 10 predicted unclear). A **pose-based temporal classifier** — a small 1D convolutional network over the EMOCA `rotation_xyz` sequence and its first differences, with no RGB input of any kind — trained on those **pseudo-labels** obtained TEST precision 0.70, recall 0.70, and F1 **0.70** (7 true positives, 3 false positives, 2 true negatives, 3 false negatives). The CNN matched the rule’s recall and reduced false positives by one clip. That difference is within the uncertainty of a 15-window test set. DEV scores (rule F1 0.86, CNN F1 0.89) were used only for tuning and are not reported as results.

EMOCA was **streamed**, not trained. A pixel model (VideoMAE) was **not** run: the lab account used for extraction has a 25 GB quota and had about 6.5 GB free after CPU PyTorch, which is insufficient for RealTalk video plus a VideoMAE checkpoint. VideoMAE is therefore reported as planned future work, not as a completed or failed result, and no VideoMAE score appears in this dissertation. Metrics are clip-level precision, recall, and F1, not event IoU. The dissertation reports a leakage-controlled pose baseline and a weakly supervised CNN, not a multimodal 7-class system.

**Keywords:** head-nod recognition; EMOCA; RealTalk; weak supervision; 1D CNN; clip-level F1.

---

## 1. Introduction

### 1.1 Motivation

Spoken dialogue is full-duplex. Listeners produce short feedback behaviours—**backchannels**—that signal attention or understanding without taking the floor (Yngve, 1970; Sacks, Schegloff and Jefferson, 1974). A head nod is one of the most familiar visual backchannels. Automatic recognition of nods is useful for conversational agents, for coding non-verbal behaviour, and as a building block for finer listener-state models.

Columbia RealTalk provides in-the-wild dyadic video at 25 fps together with per-frame EMOCA/FLAME head parameters (Geng et al., 2023; Danecek, Black and Bolkart, 2022; Li et al., 2017). It does **not** provide nod event labels. Any nod detector on this corpus must therefore either rely on heuristic pose rules, collect human labels, or both.

### 1.2 Problem statement

This dissertation addresses a binary decision on a short watch window:

> Given the listener’s EMOCA head-pose time series for about 60 s, does the window contain a **clear nod**?

The positive class is conservative: only nods the annotator judged clear. Ambiguous motion is labelled **unclear**, not nod. Evaluation is **clip-level**: one label and one prediction per window. The study does not claim to localise the nod in time (no event IoU headline) and does not classify shakes, tilts, leans, or eyebrow raises.

### 1.3 Original plan and submitted scope

Project notes written earlier in the MSc described a seven-class backchannel taxonomy and a multimodal architecture with VideoMAE and fusion, following work such as MM-F2F on coarse Keep/Turn/Backchannel prediction (Lin et al., 2025). That plan exceeded what could be labelled and stored in the time and disk available. Human labels were collected for **binary nod presence** on 30 windows. Pose was extracted by streaming official EMOCA pickles. The learned model is a pose-based temporal classifier — a 1D CNN over EMOCA `rotation_xyz` sequences, not an RGB vision model — trained on CPU. VideoMAE was not started, because about 6.5 GB remained on a 25 GB lab quota after PyTorch; it remains planned future work, not a completed or failed experiment.

The submitted thesis therefore evaluates two pose systems under a frozen TEST split. VideoMAE and seven-class typing are discussed as **future work**, not as experiments that failed on TEST.

### 1.4 Research questions

1. **RQ1.** What clip-level precision, recall, and F1 does a pose-amplitude rule achieve on 15 held-out RealTalk windows when its axis and threshold are frozen on 15 DEV windows?
2. **RQ2.** Does a 1D CNN trained on 80 automatic labels from that frozen rule improve TEST F1 without using TEST for tuning?
3. **RQ3.** Do additional pose channels (full Euler xyz, derivatives, expression) change TEST F1 at this sample size?

### 1.5 Contributions

1. A 30-window **human-labelled** gold set on RealTalk (15 DEV / 15 TEST; p0 = LEFT, p1 = RIGHT; 1 = clear nod, 0 = unclear), with TEST unused for training or threshold search.
2. A frozen Savitzky–Golay amplitude rule with a locked TEST score: F1 **0.67**.
3. A weakly supervised pose-based 1D CNN (temporal classifier over `rotation_xyz`; no RGB input) on 80 pseudo-labels with a locked TEST score: F1 **0.70**, differing from the rule by one false positive.
4. An explicit account of what was not run (VideoMAE, EMOCA training, event IoU, seven-class labelling) and why.

### 1.6 Thesis structure

Chapter 2 reviews backchannels, RealTalk, EMOCA/FLAME, weak supervision, and video transformers (the last as context for future work). Chapter 3 summarises the gold set. Chapter 4 describes the rule, pseudo-labels, and CNN. Chapter 5 reports TEST results. Chapter 6 discusses limitations and concludes.

*(Renumber to match your Word template.)*

---

## 2. Related work

Each subsection ends with a link to this dissertation, as required by the project evidence notes.

### 2.1 Backchannels and conversational action

Yngve (1970) described backchannels as short listener contributions that do not claim the turn. Sacks, Schegloff and Jefferson (1974) analysed turn-taking as a systematic organisation of speaker change. Recent computational work often predicts **coarse** actions. MM-F2F (Lin et al., 2025) models Keep, Turn-taking, and Backchannel from text, audio, and face video and reports strong binary backchannel F1 in **that** setting. Those numbers are not a baseline for clear-nod F1 on 15 RealTalk clips.

**Link.** This dissertation studies one visual backchannel type (the nod) with pose features, not the full Keep/Turn/BC taxonomy.

### 2.2 Columbia RealTalk

Geng et al. (2023) released RealTalk: hundreds of in-the-wild dyadic videos with audio, ASR, active-speaker information, and EMOCA tracks for two participants. A different corpus also named REALTALK (long-term text messaging) is **not** used here.

**Link.** All gold windows and streamed pickles come from Columbia RealTalk at 25 fps, with p0 = LEFT and p1 = RIGHT.

### 2.3 FLAME, DECA, and EMOCA

FLAME is a 3D morphable head model with pose and expression parameters (Li et al., 2017). DECA (Feng et al., 2021) and EMOCA (Danecek, Black and Bolkart, 2022) estimate such parameters from monocular video. RealTalk ships EMOCA-style tracks; this work **reads** them.

**Link.** Pose is converted from axis-angle to Euler xyz degrees. EMOCA is not fine-tuned. Expression coefficients were concatenated in one ablation that diverged and is not interpreted.

### 2.4 Head-nod detection from pose

Rule-based nod detectors typically look for oscillatory vertical head rotation in a limited duration band. This project contains an unimplemented band-pass cycle detector in the library code; the **scored** baseline is simpler: peak-to-peak amplitude after Savitzky–Golay smoothing (Savitzky and Golay, 1964) on a DEV-chosen Euler axis. Learned models can in principle use the full trace. Here the learned model is a 1D CNN on Euler sequences, not a video transformer.

**Link.** The dissertation compares the frozen amplitude rule with a CNN trained on the rule’s own labels.

### 2.5 Weak supervision

Snorkel (Ratner et al., 2017) formalises training-set creation from labelling functions when gold data are scarce. The present pipeline uses **one** labelling function (the frozen rule) and treats its outputs as TRAIN labels. There is no generative label model and no mix of independent heuristics. The method is therefore weak supervision in a minimal sense: automatic labels, human TEST.

**Link.** Eighty pseudo-labels (70/10) are the CNN’s only training supervision.

### 2.6 Video transformers (not used)

VideoMAE (Tong et al., 2022) pretrains a vision transformer by masking video patches and is a standard starting point for clip classification. It is discussed because it appeared in the original plan. **No VideoMAE run was performed** in this MSc, for disk reasons stated in Methods. Related work must not be padded with a fabricated score.

**Link.** VideoMAE is future work if storage allows; it is not a result.

### 2.7 Synthesis

Prior conversational work emphasises coarse backchannel prediction at scale (Lin et al., 2025). Prior vision work supplies the face parameterisation (Li et al., 2017; Danecek et al., 2022). What is missing for RealTalk is a **small, human-checked nod protocol with explicit leakage control**, one that does not leak TEST into threshold search. That is the gap this dissertation fills. It does not fill the gap of large-scale multimodal typing.

---

## 3. Data and annotation (short form)

If your template wants a separate Data chapter, paste the gold-count table and protocol from Methods §4.2–4.3 here instead of repeating them twice. Recommended figures:

- `figures/gold_visuals/label_counts.jpg` — 19 clear nod / 11 unclear;
- `figures/gold_visuals/labels_by_split.jpg` — 15/15 split;
- `figures/gold_visuals/labels_by_person.jpg` — LEFT/RIGHT;
- `figures/gold_visuals/clip_overview.jpg` — window overview.

State once: **gold = human label; pseudo-label = rule on TRAIN; prediction = model output.**

Mention the two off-window marked times (`Ak2Bm8mfL3w`, `Zrer1sqWzOQ`) in Data or Limitations, not in the abstract.
