# Limitations

- Gold set is 30 clips (15 TEST). A one-clip change in the confusion matrix moves TEST F1 by a few points.
- Single annotator; no inter-annotator agreement.
- Pseudo-labels on 80 TRAIN clips were 70 nod / 10 unclear (frozen rule bias).
- Clip-level Precision / Recall / F1 is the reported metric. Event F1 @ IoU 0.30 was not computed for this protocol.
- EMOCA was streamed from the official RealTalk archive, not trained. `emoca.tar.gz` was never saved on otter (25 GB quota).
- Feature set D (expression) diverged (`loss = nan`); that ablation is omitted from headlines.
- Compute/quota: otter had ~6.5 GB free on a 25 GB home quota after CPU PyTorch, so video shards plus a full VideoMAE training stack do not fit. What **was** run: a frozen VideoMAE head (TEST F1 0.57) and a partial fine-tune of the last 4 encoder blocks on an RTX A4000 (otter95) with CUDA PyTorch installed on `/scratch` outside the quota (TEST F1 0.82). What remains future work: full fine-tuning of all 86.2M parameters and a larger pseudo-label pool.
- The fine-tuned VideoMAE F1 (0.82) is the highest point estimate, but at n=15 TEST its 95% CI [0.60, 0.96] overlaps every other system's; no pairwise difference is statistically significant. An earlier bootstrap over all 110 predictions was train-contaminated and was corrected to the 15-row TEST-only file.
- Synthetic `pilot_*` clips are not RealTalk results and are not reported.
