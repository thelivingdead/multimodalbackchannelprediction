# Limitations

- Gold set is 30 clips (15 TEST). A one-clip change in the confusion matrix moves TEST F1 by a few points.
- Single annotator; no inter-annotator agreement.
- Pseudo-labels on 80 TRAIN clips were 70 nod / 10 unclear (frozen rule bias).
- Clip-level Precision / Recall / F1 is the reported metric. Event F1 @ IoU 0.30 was not computed for this protocol.
- EMOCA was streamed from the official RealTalk archive, not trained. `emoca.tar.gz` was never saved on otter (25 GB quota).
- Feature set D (expression) diverged (`loss = nan`); that ablation is omitted from headlines.
- VideoMAE / pixel models were not run: otter had ~6.5 GB free after CPU PyTorch; video shards plus a VideoMAE checkpoint do not fit.
- Synthetic `pilot_*` clips are not RealTalk results and are not reported.
