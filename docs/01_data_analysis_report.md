# Data Analysis Report: Multimodal Fine-Grained Backchannel Prediction

**Project:** Predicting multimodal backchannel behaviours  
**Candidate framing:** MSc Dissertation | University of Surrey (CVSSP)  
**Primary experimental dataset:** Columbia RealTalk (Geng et al., 2023)  
**Key multimodal baseline literature:** MM-F2F (Lin et al., ACL 2025; arXiv:2505.12654)

---

## 1. Research problem

Natural spoken dialogue is full-duplex: listeners do not wait for an explicit end-of-turn signal. They produce short feedback behaviours—**backchannels**—such as nods, brief vocalisations (“mm-hmm”), eyebrow raises, or posture leans that signal attention, understanding, or agreement without taking the floor (Yngve, 1970; Sacks et al., 1974).

Most deployed voice agents still rely on voice-activity detection or push-to-talk. That produces unnatural pauses and misses listener state. Recent work predicts coarse conversational actions—*Keep*, *Turn-taking*, *Backchannel*—from text, audio, and video (Lin et al., 2025). That is necessary but not sufficient for socially fluent agents: a system that only knows “a backchannel is likely” still does not know **which** non-verbal behaviour to generate or interpret.

This dissertation focuses on **fine-grained multimodal backchannel typing**: predicting a small taxonomy of listener behaviours from linguistic, acoustic, visual, and parametric face (FLAME/EMOCA) signals.

### 1.1 Why fine-grained typing matters

| Coarse label (MM-F2F style) | Fine-grained need |
| --- | --- |
| Backchannel (binary) | Distinguish affirmation (nod) vs uncertainty (tilt/lean) vs emphasis (eyebrow raise) |
| Keep / Turn | Separate floor-holding from listener feedback |
| Single modality cues | Combine ASR context, prosody, and head dynamics |

Fine-grained labels support (i) more natural avatar/listener animation, (ii) better turn-management policies, and (iii) analysis of which modalities carry which cue types.

---

## 2. Dataset landscape and primary choice

### 2.1 Clarifying “REALTALK” naming

Two different resources share similar names:

| Resource | Reference | Modality | Relevance to this project |
| --- | --- | --- | --- |
| **Columbia RealTalk** | Geng et al., 2023; [realtalk.cs.columbia.edu](https://realtalk.cs.columbia.edu/) | Dyadic video, audio, ASR, EMOCA/FLAME, active speaker | **Primary experimental dataset** |
| **REALTALK (messaging)** | Lee et al., arXiv:2502.13270 | 21-day text messaging dialogues, EI/persona benchmarks | Long-term dialogue context only; **not** the multimodal BC source |

This report uses **Columbia RealTalk** for multimodal analysis. arXiv:2502.13270 is noted only to avoid citation confusion.

### 2.2 Columbia RealTalk (primary)

From project notes and published dataset descriptions:

- ~**692** in-the-wild dyadic conversation videos at **25 fps**
- Per-frame **EMOCA / FLAME-style** head and expression parameters
- **ASR transcripts**, raw **audio**, and **active-speaker** annotations
- Speaker/listener role structure is available; **backchannel-type labels are not shipped**
- Labels for this dissertation are **derived** from FLAME signals (with hand validation)

Implication for analysis: EDA must cover both shipped modalities and the quality of derived event labels.

### 2.3 MM-F2F (baseline literature / comparison)

Lin et al. (2025) release MM-F2F: ~**210 hours**, ~**169k utterances**, ~**1.5M** word frames, ~**51k** turn-taking and ~**22k** backchannel instances, with text + audio + face video and word-level Keep/Turn/BC labels. Their tri-modal model reports strong gains on binary BC (F1 ≈ 0.91 in their setting).

**Gap relative to this dissertation:** MM-F2F treats backchannel as a single class. It does not provide fine-grained nod/shake/tilt/lean/eyebrow typing from FLAME parameters.

### 2.4 Comparative summary of related corpora

Adapted from MM-F2F Table 1 and project notes (T/A/V = text/audio/video; Turn/BC = turn-taking / backchannel annotations):

| Dataset | T | A | V | Turn | BC | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Switchboard | ✓ | ✓ | — | ✓ | ✓ | Classic telephony; no video |
| FTAD | ✓ | — | — | ✓ | ✓ | Text-focused turn/BC actions |
| EgoCom | ✓ | ✓ | ✓ | ✓ | — | Multi-person / egocentric; limited F2F face cues |
| MM-F2F | ✓ | ✓ | ✓ | ✓ | ✓ (binary) | Strong tri-modal baseline; coarse BC |
| Columbia RealTalk | ✓ | ✓ | ✓ + FLAME | role/ASD | **derived 7-class** | Chosen for parametric head dynamics |

---

## 3. Prediction taxonomy (7 classes)

| Class ID | Name | Primary FLAME / signal cue (label generation) |
| --- | --- | --- |
| 0 | `nod` | Pitch oscillation in ~1–3 Hz band; ≥2 clean cycles |
| 1 | `shake` | Yaw (left–right) oscillatory head rotation |
| 2 | `tilt` | Roll / lateral tilt of neck rotation |
| 3 | `lean_forward` | Head translation toward camera / interlocutor |
| 4 | `lean_back` | Head translation away |
| 5 | `eyebrow_raise` | Expression / Action-Unit style brow signal |
| 6 | `neutral` | No event rule fires in the window |

This taxonomy follows the project’s head-nod / MMHead-style reasoning: pose and expression are annotated separately from parametric signals, then one unified classifier learns all classes.

---

## 4. Exploratory data analysis plan

Until RealTalk is mounted on local/server storage, quantitative tables below are **planned analyses** with expected patterns. The companion notebook [`notebooks/01_eda_skeleton.ipynb`](../notebooks/01_eda_skeleton.ipynb) implements the loaders and plots.

### 4.1 Corpus-level checks

1. **Video inventory:** count clips, duration histogram, fps consistency (expect 25 fps).
2. **Speaker coverage:** number of unique identities / dyads; active-speaker balance (left vs right face track).
3. **ASR coverage:** words per utterance, empty/failed ASR rate, language consistency (English).
4. **FLAME completeness:** missing frames, tracking failures, outlier pose magnitudes.

### 4.2 Conversational structure

Expected RealTalk / dyadic patterns (to verify empirically):

- Short listener responses and longer speaker turns
- Backchannel-like windows often co-occur with speaker continuation (Keep), not floor shifts
- Class imbalance: `neutral` and `nod` likely dominate; `lean_*` and `eyebrow_raise` rarer

### 4.3 Derived label statistics (critical)

For each video window (e.g. 0.5–1.0 s, hop 0.1–0.2 s):

| Analysis | What to report |
| --- | --- |
| Class histogram | Counts / percentages for 7 classes |
| Co-occurrence | How often multiple rules fire; priority policy |
| Duration | Event length distribution per class |
| Temporal IoU stability | Sensitivity to band-pass / peak thresholds |
| Role conditioning | Listener-only vs speaker-window false positives |

### 4.4 Modality alignment sanity checks

- Word timestamps vs audio energy
- FLAME frame index vs video frame
- Active-speaker mask vs who should exhibit listener BC

Misalignment is a primary source of training noise and must be quantified before model runs.

### 4.5 Expected imbalance (working hypothesis)

| Class | Relative frequency (hypothesis) | Modelling implication |
| --- | --- | --- |
| neutral | High | Downsample or class weights |
| nod | Medium–high | Strong visual/pose cue |
| shake / tilt | Low–medium | Need careful thresholds |
| lean_forward / lean_back | Low | Risk of under-recall |
| eyebrow_raise | Low | Benefit from expression channels |

---

## 5. Label derivation analysis

### 5.1 Head nod (reference method)

Following Chen et al. (ICCVW head-nod work) and project notes:

1. Extract FLAME pitch time series \(p(t)\).
2. Band-pass filter to the nod band (~**1–3 Hz**).
3. Detect peaks; require **≥ 2 clean cycles** in a span.
4. Flag that span as `nod`.

Libraries planned: `scipy.signal` (`butter`, `filtfilt`, `find_peaks`), `scipy.spatial.transform.Rotation` for axis-angle → pitch/yaw/roll.

### 5.2 Other pose classes

| Class | Signal | Rule sketch |
| --- | --- | --- |
| shake | yaw \(y(t)\) | Oscillatory peaks above amplitude threshold |
| tilt | roll \(r(t)\) | Sustained or oscillatory lateral deviation |
| lean_forward / lean_back | translation \(z(t)\) or depth proxy | Signed displacement over window |
| eyebrow_raise | expression coeffs / AU proxy | Threshold on brow-related dimensions |
| neutral | — | Default when no rule fires |

### 5.3 Conflict resolution

When multiple rules fire in the same window, apply a fixed priority (example):

`nod > shake > eyebrow_raise > tilt > lean_forward > lean_back > neutral`

Document the priority in the dissertation and ablate alternatives (multi-label vs single-label).

### 5.4 Validation protocol

1. Auto-label full corpus.
2. Sample **N ≈ 200–400** windows stratified by class.
3. Human double-check; target **precision ≥ 0.85** on non-neutral classes before freezing labels v1.
4. Report confusion between auto and human labels.

---

## 6. Links to modelling (preview)

EDA outputs feed the methodology (see `02_research_methodology_and_roadmap.md`):

- Class weights from the histogram
- Window length from event duration stats
- Modality dropout rates from missingness rates
- Baselines that ignore rare classes will look artificially strong on accuracy—**macro-F1** is mandatory

Literature anchors for multimodal fusion performance (MM-F2F, on their 3-class task):

| Training modalities | Approx. BC F1 (Lin et al., 2025) |
| --- | --- |
| Text | 0.707 |
| Audio | 0.805 |
| Video (face) | 0.513 |
| Text+Audio | 0.894 |
| Text+Audio+Video | **0.906** |

These numbers motivate tri-modal modelling but do **not** transfer directly to 7-class FLAME typing; they are baselines for experimental design, not claimed results of this dissertation.

---

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Auto-label noise | Inflated/deflated metrics | Hand validation set; noise-robust loss; report label agreement |
| Class imbalance | Accuracy dominated by neutral | Weighted CE, macro-F1, resampling |
| Tracking failures | Spurious pose events | Mask low-confidence FLAME frames |
| Storage / GPU | Slow iteration | Cache features; start with subset of videos |
| Privacy | Ethics review | Follow RealTalk licence; no identity re-identification |
| Confusable cues | Nod vs lean / tilt confusion | Per-class error analysis; qualitative video review |
| Naming confusion with arXiv:2502.13270 | Wrong citations | Always cite Columbia RealTalk + Geng et al. for data |

---

## 8. Deliverables from the analysis phase

1. This report (literature + analysis plan + risks).
2. Runnable EDA skeleton notebook once data path is set.
3. Label-rule reference implementation used by the demo API (`api/label_rules.py`).
4. Figures for dissertation Chapter “Data”: duration histograms, class bars, alignment heatmaps (to be filled after mount).

---

## 9. Immediate next steps after data mount

1. Set `REALTALK_ROOT` and run `notebooks/01_eda_skeleton.ipynb`.
2. Tune nod band-pass thresholds on a 10-video pilot.
3. Freeze **labels v1** + validation spreadsheet.
4. Proceed to methodology roadmap Week 3–4 (dataloader + splits).

---

## References (core)

- Geng et al. (2023). *RealTalk* — Columbia audiovisual dyadic conversations. https://realtalk.cs.columbia.edu/
- Lin, Zheng, Zeng, Shi (2025). Predicting Turn-Taking and Backchannel… ACL 2025. arXiv:2505.12654. https://github.com/Linyx1125/MM-F2F
- Lee et al. (2025). REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation. arXiv:2502.13270 *(distinct resource)*.
- Yngve (1970). On getting a word in edgewise.
- Sacks, Schegloff, Jefferson (1974). A simplest systematics for the organization of turn-taking for conversation.
- Ekstedt & Skantze (2022). Voice Activity Projection (VAP).
- Liu et al. (2018). Efficient low-rank multimodal fusion (LMF).
- Wu et al. (2024). MMHead — FLAME/EMOCA head-pose labelling pipeline. arXiv:2410.07757.

---

*Report status: analysis framework complete; corpus-specific numeric tables pending RealTalk mount.*
