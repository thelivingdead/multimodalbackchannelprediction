# Methodology notes

- Two annotation classes only: `1` clear nod, `0` unclear.
- Gold positives = class 1 events only.
- Primary event metric: F1 at temporal IoU 0.30, one-to-one matching.
- Tune nod rule on PILOT/DEV only. GOLD TEST unused until 15 new videos exist.
- EMOCA pose conversion is LIKELY (rotvec[:3] → Euler xyz degrees) until `03_inspect_emoca.py` marks VERIFIED.
- Do not map unknown expression coefficients to eyebrow raise.
- Storage hard cap 24 GB; no full RealTalk / emoca.tar.gz.
