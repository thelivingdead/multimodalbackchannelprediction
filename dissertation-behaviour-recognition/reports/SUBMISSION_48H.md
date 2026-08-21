# 48-hour submission plan

**SUPERSEDED — 21 August 2026.** VideoMAE **was run**. Ignore any line below that says it was not. Locked TEST scores: `results/tables/main_results.md` (rule 0.67, pose CNN 0.70, frozen VideoMAE 0.57, fine-tune n=80 **0.82**, n=200 ablation **0.63**). Paste from `results_chapter_draft.md` and `discussion_conclusion_draft.md`. This file is only an old writing timetable.

**Candidate:** Divya Bisht, MSc, University of Surrey  
**Originally written:** Wednesday 19 August 2026 (pose-only plan; now outdated)

Paste from the markdown files in this folder into your Word (or Overleaf) template. There is **no** existing `.docx`/`.tex` dissertation in the workspace.

---

## Locked TEST table (n = 15; never retune)

Copy this table exactly. Rounding is intentional (rule precision 0.636… → 0.64; rule F1 0.666… → 0.67).

| Method | Precision | Recall | F1 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen pose rule | 0.64 | 0.70 | **0.67** | 7 | 4 | 1 | 3 |
| 1D CNN (80 pseudo-labels) | 0.70 | 0.70 | **0.70** | 7 | 3 | 2 | 3 |

- Gold: 30 human-labelled clips, 15 DEV / 15 TEST, **1 = clear nod**, **0 = unclear**, p0 = LEFT, p1 = RIGHT, RealTalk 25 fps.
- Frozen rule: axis **x**, threshold **16.35°**. DEV F1 **0.86 is not a headline**.
- CNN TRAIN: 70 nod / 10 unclear **pseudo-labels** (automatic rule labels, not gold). DEV F1 **0.89 is not a headline**.
- Clip-level P/R/F1. **Not** event IoU.
- Ablation D diverged (`nan` / F1 0): **do not report as a result**.
- VideoMAE **was run** (see master table). Do not write “not run.”

JSON originals: `../results/rule_test_metrics.json`, `../results/classifier_test_metrics.json`.

---

## What to paste (file map)

| Dissertation part | Paste from |
| --- | --- |
| Abstract + Ch. 1 Intro + Ch. 2 Literature | `abstract_intro_lit_draft.md` |
| Ch. 3 Data (if separate) | Data section of that file, plus gold tables in Results |
| Ch. 4 Methods | `methods_chapter_draft.md` |
| Ch. 5 Results | `results_chapter_draft.md` |
| Ch. 6 Discussion + limitations + conclusion | `discussion_conclusion_draft.md` |
| Figure captions | `figure_captions.md` |
| References | `references_harvard.md` |
| Inventory of old notes | `WRITING_INVENTORY.md` |

Renumber headings to match the Surrey template you were given. Keep the **words**.

---

## What **not** to claim

- That the CNN “significantly” beats the rule (one FP on \(n=15\)).
- DEV F1 0.86 or 0.89 as the result.
- Any VideoMAE F1, accuracy, or confusion matrix.
- That EMOCA was trained (it was **streamed**).
- That ablation D shows expression is useless (it **diverged**).
- Event-level F1 at IoU 0.30 as the headline of **this** 30-window protocol.
- Seven-class backchannel results, LMF fusion, or MM-F2F’s 0.91 as *your* score.
- Synthetic `pilot_*` clips as RealTalk evidence.
- Predictions described as “gold”.
- That VideoMAE’s absence is a failed experiment rather than a 25 GB / ~6.5 GB free constraint.

**Terminology:** prefer **human-labelled** / **manually annotated**. Define **gold** once (the human 1/0 label). **Pseudo-labels** = automatic rule labels on TRAIN. **Predictions** ≠ gold.

---

## Figures to insert (Day 1)

From `dissertation-behaviour-recognition/figures/`:

| Must insert | File |
| --- | --- |
| Label counts | `gold_visuals/label_counts.jpg` |
| Split counts | `gold_visuals/labels_by_split.jpg` |
| LEFT/RIGHT | `gold_visuals/labels_by_person.jpg` |
| Rule TEST confusion | `rule_confusion_matrix.jpg` |
| CNN TEST confusion | `classifier_confusion_matrix.jpg` |
| TEST F1 bars | `model_comparison_f1.jpg` |
| Pseudo-label 70/10 | `pseudo_label_distribution.jpg` |
| DEV threshold curve (Methods) | `rule_dev_threshold_curve.jpg` |

Helpful: `example_positive_rotation.jpg`, `example_negative_rotation.jpg`, `gold_visuals/clip_overview.jpg`, `gold_visuals/pose_traces_extracted.jpg`.

Appendix only: `training_loss.jpg`, `dev_f1_by_epoch.jpg`.

Skip or heavily caption: `ablation_f1.jpg` (contains invalid D).

---

## Day 1 — Methods + Results + figures (no lab)

Assume ~12 working hours. Shift the clock to when you actually start; keep the **order**.

| Hours | Task | Done when |
| --- | --- | --- |
| 0:00–0:20 | Read this file and the locked table. Skim `WRITING_INVENTORY.md`. **Do not log into otter.** | You can recite TEST F1 0.67 / 0.70 |
| 0:20–1:00 | Open the official Surrey cover / declaration / abstract page. Title suggestion: *Weakly supervised head-nod recognition from EMOCA pose on Columbia RealTalk*. | Title page exists |
| 1:00–2:30 | Paste **Abstract** and **Chapter 1** from `abstract_intro_lit_draft.md`. Add your student ID, supervisor, word count later. | Intro ends with RQs and contributions |
| 2:30–4:00 | Paste **Chapter 2** literature. Add only references you have opened. | Each subsection ends with a link to this study |
| 4:00–5:00 | **Lunch / break.** Do not “quickly train” anything. | — |
| 5:00–8:00 | Paste **Methods** (`methods_chapter_draft.md`). Insert Methods figures (5–9 in `figure_captions.md`). Check: streamed EMOCA, CPU torch, axis x, 16.35°, 80 pseudo-labels, clip-level metric. | Methods is complete sentences, not bullets |
| 8:00–11:00 | Paste **Results** (`results_chapter_draft.md`). Insert Figures 10–12. Type the TEST table **by hand from this page**, then check against JSON. Insert gold count figures. | Table matches TP7/FP4/TN1/FN3 and TP7/FP3/TN2/FN3 |
| 11:00–12:00 | Captions, cross-references, file names removed from the prose (“see Figure 10”, not `rule_confusion_matrix.jpg`). Save a PDF snapshot. | Day-1 PDF exists |

If you run behind, **cut literature**, not the TEST table.

---

## Day 2 — Discussion + proofread + submit (no lab)

| Hours | Task | Done when |
| --- | --- | --- |
| 0:00–2:30 | Paste **Discussion, Limitations, Conclusion** from `discussion_conclusion_draft.md`. | RQ1–RQ3 answered with TEST numbers only |
| 2:30–3:30 | **References** from `references_harvard.md`. Match in-text citations. Remove any paper you did not look at. | No invented DOIs |
| 3:30–5:00 | Front matter: abstract word limit, acknowledgements, contents, list of figures/tables, ethics/licence sentence for RealTalk. | Contents page works |
| 5:00–7:00 | Proofread pass **A — numbers.** Search the PDF for `0.86`, `0.89`, `VideoMAE`, `IoU`, `nan`, `pilot`. DEV F1 may appear only as “tuning, not generalisation”. VideoMAE only as not run / future work. | Number audit complete |
| 7:00–9:00 | Proofread pass **B — English.** Complete sentences, Surrey register, no hype (“significant”, “state-of-the-art”, “proves”). Gold vs pseudo-label vs prediction. | One quiet read-aloud of Abstract + Conclusion |
| 9:00–10:00 | Format: margins, heading styles, figure resolution, page numbers, declaration signed. | Print-preview looks like a thesis |
| 10:00–11:00 | Export PDF. Run Turnitin / repository upload **according to the department email**, not this file. Keep a dated copy on the Mac and on a USB/cloud. | Submission receipt |
| 11:00+ | Stop. Do not log into otter “just to add VideoMAE”. | Submitted |

---

## Emergency cuts (if you have 8 hours, not 24)

Keep, in this order:

1. Abstract with the TEST table in words  
2. Methods (protocol + no TEST leakage)  
3. Results table + two confusion matrices  
4. Limitations list (n=15, one annotator, no VideoMAE, clip-level F1)  
5. Conclusion (F1 0.67 vs 0.70; one FP)

Drop first: long literature, ablation table, per-clip error table, appendix plots.

---

## Otter / GPU

**Stay idle.** The TEST scores are locked. More epochs cannot be reported without violating “TEST once”. VideoMAE does not fit the quota. This document is the submission path.

---

## Quick self-test before upload

- [ ] TEST table is the locked table above  
- [ ] DEV F1 is not in the abstract  
- [ ] Word “gold” is defined once as human labels  
- [ ] Pseudo-labels described as automatic  
- [ ] VideoMAE appears only as not run (25 GB / ~6.5 GB free)  
- [ ] Ablation D omitted or marked diverged  
- [ ] No event-IoU headline  
- [ ] Figures 10–12 show the same 2×2 as the table  
- [ ] Otter was not used in these 48 hours  
