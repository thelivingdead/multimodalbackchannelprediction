# HuBERT DEV experiment

Frozen HuBERT representations extracted from approximately sixty second mixed conversation audio windows, trained using the existing pose derived pseudo labelled training set and evaluated exploratorily on fifteen gold development clips.

- Model: `facebook/hubert-base-ls960` (frozen; not fine-tuned)
- Audio: mixed conversation audio (RealTalk container soundtrack), resampled to 16 kHz
- Windows: existing nod TRAIN pseudo-labelled clips (`results/pseudo_labels.csv`) and GOLD DEV `gold_001`–`gold_015`
- Embeddings: mean pool over HuBERT frames in 10 s chunks, then length-weighted mean over chunks (768-D)
- Classifier: StandardScaler and PCA fitted on TRAIN only, then LogisticRegression (`class_weight=balanced`, seed 42)
- Evaluation split: GOLD DEV only. GOLD TEST (`gold_016`–`gold_030`) is not used
- Fusion (if RGB probabilities exist): `p_fusion = 0.5 * p_rgb + 0.5 * p_hubert` at threshold 0.5. No weight search
- Outputs: this directory

Primary HuBERT numbers use threshold 0.5. Any DEV-selected threshold is stored as `EXPLORATORY_DEV_SELECTED_THRESHOLD` and does not replace the 0.5 result.
