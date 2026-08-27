# Audio alignment check (Step A)

**Verdict: FAIL.** 1 / 5 selected nod windows had extractable source audio. GOLD TEST was not used. Steps 2–9 were **not** run. No F1 invented.

Approved dissertation title (unchanged): **Predicting Backchannel Events from Multimodal Conversational Signals**.

Audio, when present, is the RealTalk **container soundtrack** (mixed conversation; stereo MP3 48 kHz on `gold_012`). It is **not** speaker audio and **not** listener-only.

Pass rule: at least 3 clips must pass. This run stopped after one clip (20-minute Step 1 budget). Hugging Face Range reads work outside the Cursor sandbox proxy (HTTP 206).

## Clips

| sample_id | video_id | split | person | label | t0_s | duration_s | src_sr | wav_sr | wav_dur_s | rms | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gold_012 | 6RDkdbgzeAI | DEV | p0 | 0 | 368.000 | 60.000 | 48000 | 48000 | 60.000 | 0.038 | True |
| gold_013 | WrWFSBLjWZU | DEV | p0 | 1 | 42.000 | 60.000 |  |  |  |  | False |
| gold_010 | jg6y3LABwTs | DEV | p1 | 0 | 88.000 | 60.000 |  |  |  |  | False |
| gold_014 | f6aNo5Mod9I | DEV | p0 | 1 | 53.000 | 60.000 |  |  |  |  | False |
| gold_009 | GJtqigeWHV8 | DEV | p1 | 1 | 124.000 | 60.000 |  |  |  |  | False |

## Blocker

Step A FAIL: 1/5 clips extracted (need 3) within the 20-minute wall-clock stop. gold_012 PASS (WAV 60.000 s, 48000 Hz, 2880000 samples, RMS 0.03762281). Remaining ids not fetched. Do not run Steps 2–9. See `reports/audio_alignment_audit.md`.

## Next step

Re-run `scripts/audio_alignment_check.py` (Mac with `all`/no-proxy network, or otter) until **Verdict: PASS**. Then DEV-only `train_audio_baseline_dev.py` / `train_av_fusion_dev.py`. They refuse GOLD TEST. Nod TEST headline remains RGB fine-tune **0.82**.
