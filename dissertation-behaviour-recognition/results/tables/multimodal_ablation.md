# RGB / audio ablation (GOLD DEV only)

**DEV ONLY.** GOLD TEST was not scored. Nod TEST headline remains fine-tuned VideoMAE F1 **0.82** (locked RGB). Shake TEST headline remains pose rule F1 **0.70** (locked).

Pose + RGB in the locked tables are **visual representation experiments** (two encodings of the camera stream), not two sensory modalities. This table is the first **auditory** stream on the nod task, fused by concatenating a *frozen* VideoMAE vector with clip audio statistics. VideoMAE was not retrained.

| model | input | n | P | R | F1 | bal-acc | TP FP TN FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| always_nod | none | 15 | 0.600 | 1.000 | **0.750** | 0.500 | 9 6 0 0 |
| rgb_lr_dev_threshold | frozen VideoMAE 768-D | 15 | 0.750 | 1.000 | **0.857** | 0.750 | 9 3 3 0 |
| audio_lr_dev_threshold | MFCC/RMS/centroid 30-D | 15 | 0.615 | 0.889 | **0.727** | 0.528 | 8 5 1 1 |
| rgb_audio_lr_dev_threshold | concat 768-D + 30-D | 15 | 0.643 | 1.000 | **0.783** | 0.583 | 9 5 1 0 |

TRAIN n=80 (pseudo-labels). DEV n=15 (gold). Thresholds chosen on DEV (RGB 0.550, audio 0.300, fusion 0.200). Text/transcript models were not run (future work). This is **supervised prediction of the backchannel label associated with a conversational window**, not anticipatory forecasting.

