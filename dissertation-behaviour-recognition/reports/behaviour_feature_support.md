# Behaviour vs EMOCA/FLAME feature support

Status after code inspection, **before** a real RealTalk pickle is opened on the lab.
Re-run `python scripts/03_inspect_emoca.py` and update `reports/emoca_schema.json`.

| behaviour | available feature | source | meaning | verified? | rule feasible | limitations | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nod | pose[0:3] → pitch | EMOCA 6D pose (LIKELY) | vertical head rotation | LIKELY until pickle inspect | yes | amplitude/camera; talking motion | medium-high |
| head_shake | pose[0:3] → yaw | same | horizontal rotation | LIKELY | yes, unscored | same | medium |
| head_tilt | pose[0:3] → roll | same | lateral tilt | LIKELY | yes, unscored | resting pose confound | medium |
| eyebrow_raise | expression coeffs | unknown key | facial AU-like | NOT AVAILABLE | no | do not invent coeff meaning | none |
| lean_forward | translation / cam | unknown key | depth/approach | NOT AVAILABLE | no | do not use pitch as lean | none |
| lean_back | translation / cam | unknown key | retreat | NOT AVAILABLE | no | same | none |
| neutral | complement of events | derived | non-event time | implicit | n/a | not every non-event is a good negative | — |

Conversion used for pose: `scipy.spatial.transform.Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)` → pitch, yaw, roll.
