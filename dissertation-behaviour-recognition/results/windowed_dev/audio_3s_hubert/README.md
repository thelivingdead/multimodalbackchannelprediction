# 3 s frozen HuBERT nod (DEV)

Same 435 DEV windows as `audio_3s/` (MFCC). Frozen `facebook/hubert-base-ls960`, mean-pooled 768-D, leave-one-clip-out logistic regression, threshold 0.5. Weights are not updated. TEST is not scored.

Run on Otter after the MFCC wav cache exists:

```bash
bash scripts/run_windowed_hubert_3s_otter.sh
```

Compare to MFCC DEV BA 0.534 [0.470, 0.601]. Do not fine-tune until this frozen run is read. Do not score TEST.
