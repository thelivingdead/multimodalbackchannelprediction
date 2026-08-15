# Multimodal Backchannel Dissertation 

Fine-grained **7-class** multimodal backchannel prediction artefacts for an MSc dissertation:

| Class | Meaning |
| --- | --- |
| `nod` | Affirmation / attention (pitch ~1–3 Hz) |
| `shake` | Negation (yaw oscillation) |
| `tilt` | Uncertainty / curiosity (roll) |
| `lean_forward` / `lean_back` | Engagement vs distancing |
| `eyebrow_raise` | Surprise / social accent |
| `neutral` | No event in the window |

**Primary data (experimental):** Columbia RealTalk (Geng et al., 2023) — [realtalk.cs.columbia.edu](https://realtalk.cs.columbia.edu/)  
**Key literature baseline:** MM-F2F (Lin et al., ACL 2025) — [arxiv:2505.12654](https://arxiv.org/abs/2505.12654)  
**Note:** [arxiv:2502.13270](https://arxiv.org/abs/2502.13270) is a *different* REALTALK (messaging corpus), not the audiovisual dataset used here.

Demo product name: **BackchannelAI**

---

## Repository layout

```
docs/
  01_data_analysis_report.md
  02_research_methodology_and_roadmap.md
notebooks/
  01_eda_skeleton.ipynb
scripts/
  visualise_flame_vs_frames.py
  README_VIS.md
  nod_pipeline/          10×1min nod pipeline (storage-safe)
api/
  main.py
  label_rules.py
  requirements.txt
web/
  (Vite + React + TypeScript app)
README.md
```

---

## 1. Read the reports

- Data analysis: [`docs/01_data_analysis_report.md`](docs/01_data_analysis_report.md)
- Methodology + roadmap: [`docs/02_research_methodology_and_roadmap.md`](docs/02_research_methodology_and_roadmap.md)

## 2. Run the EDA notebook

```bash
cd notebooks
export REALTALK_ROOT="/path/to/realtalk"   # optional; synthetic demo if unset
jupyter notebook 01_eda_skeleton.ipynb
```

Requires: `numpy`, `pandas`, `matplotlib`, `scipy`.

## 3. Start the prediction API

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- Interactive docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  
- JSON predict: `POST /predict` with `{"text":"Yeah, that makes sense","modalities":["text"]}`  
- Upload predict: `POST /predict/upload` (multipart text + optional audio/video)

The current predictor is **heuristic** (FLAME-style rules + lexical priors). `MODEL_CHECKPOINT` in `main.py` is the swap-in point for a trained PyTorch model later.

## 4. Start the website

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open http://127.0.0.1:5173 — the Vite dev server proxies `/api/*` to the FastAPI server on port 8000.

Pages:

- `/` — BackchannelAI overview  
- `/predict` — run predictions  
- `/docs` — links to dissertation artefacts  

---

## Quick smoke test (API)

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Yeah, that makes sense — go on.","modalities":["text"]}'
```

---

## 5. Frame ↔ FLAME visualisation (lab GPU / VS Code Remote)

```bash
pip install -r scripts/requirements.txt
python scripts/visualise_flame_vs_frames.py --demo --out outputs/viz_demo
```

See [`scripts/README_VIS.md`](scripts/README_VIS.md). Progress bullets for supervisors: [`docs/03_two_week_progress.md`](docs/03_two_week_progress.md).

## 6. Tiny-subset nod pipeline (10 × 1-minute clips)

Storage-safe path: do **not** download the full EMOCA archive. Demo first, then 10 real minutes.

```bash
pip install -r scripts/nod_pipeline/requirements.txt
bash scripts/nod_pipeline/run_all_demo.sh
```

Full instructions: [`scripts/nod_pipeline/README.md`](scripts/nod_pipeline/README.md).

## Dissertation mapping

| Deliverable | Use in thesis |
| --- | --- |
| Data report + EDA | Data chapter |
| Methodology & roadmap | Methods + project plan |
| Two-week progress | Supervisor update |
| Frame/FLAME viz script | Data QA figures |
| BackchannelAI site/API | Demo / appendix / qualitative inspection |
