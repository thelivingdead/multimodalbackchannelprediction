# 3 s audio-only nod (DEV)

DEV leave-one-clip-out MFCC logistic regression on the same 3 s windows as the nod visual experiments. TEST is not scored.

Run on Otter after you push the scripts:

```bash
bash scripts/run_windowed_audio_3s_otter.sh
```

This folder is filled by that run (`dataset_audit.csv`, `metrics.json`, `predictions.csv`, `figures/`, `tables/`).
