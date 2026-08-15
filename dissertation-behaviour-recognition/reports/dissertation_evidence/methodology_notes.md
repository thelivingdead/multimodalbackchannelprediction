# Methodology notes

- Two annotation classes only: `1` clear nod, `0` unclear.
- Gold positives = class 1 events only.
- Primary event metric: F1 at temporal IoU 0.30, one-to-one matching.
- Tune the nod rule on DEV only. Score GOLD TEST once; do not train on it.
- EMOCA pose conversion is treated as likely (`rotvec[:3]` → Euler xyz degrees) until `03_inspect_emoca.py` marks it verified.
- Do not map unknown expression coefficients to eyebrow raise.
- Pose archives are not stored in git. Use existing lab pose files; do not fetch the full RealTalk or EMOCA tarball.
