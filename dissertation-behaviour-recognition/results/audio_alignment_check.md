# Audio alignment check (Step A)

**Verdict: PASS.** 5 / 5 selected nod windows passed source-video / timestamp / audible-audio checks. GOLD TEST was not used.

Approved dissertation title (unchanged): **Predicting Backchannel Events from Multimodal Conversational Signals**.

Audio is the RealTalk **container soundtrack** for the watch window (both participants), not a separated listener channel. Pose and RGB are visual encodings of the same camera; this check only asks whether that camera file has usable, time-aligned audio.

Pass rule: at least 3 clips must pass, each with a video stream, an audio stream, fps ≈ 25, non-empty audible WAV whose duration matches the 1500-frame / 25 fps pose window (±0.5 s), and matching LEFT/RIGHT listener vs `who_to_watch`.

## Clips

| sample_id | video_id | split | person | label | t0_s | duration_s | src_sr | wav_sr | wav_dur_s | rms | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gold_012 | 6RDkdbgzeAI | DEV | p0 | 0 | 368.000 | 60.000 | 48000 | 48000 | 60.000 | 0.038 | True |
| gold_013 | WrWFSBLjWZU | DEV | p0 | 1 | 42.000 | 60.000 | 48000 | 48000 | 60.000 | 0.037 | True |
| gold_010 | jg6y3LABwTs | DEV | p1 | 0 | 88.000 | 60.000 | 48000 | 48000 | 60.000 | 0.108 | True |
| gold_014 | f6aNo5Mod9I | DEV | p0 | 1 | 53.000 | 60.000 | 48000 | 48000 | 60.000 | 0.050 | True |
| gold_009 | GJtqigeWHV8 | DEV | p1 | 1 | 124.000 | 60.000 | 48000 | 48000 | 60.000 | 0.051 | True |

## gold_012 (`6RDkdbgzeAI`)

- origin: `gold` split **DEV** (TEST clips are refused)
- person: `p0` → LEFT | sheet: LEFT — watch the person on the LEFT. Ignore the speaker on the right.
- pose frames: 9200–10699 inclusive (1500 frames @ 25 fps → 60.0 s) from t0=368.0 s
- watch clock: 6:08–7:08 | marked nod: 6:33–6:34
- youtube: https://www.youtube.com/watch?v=6RDkdbgzeAI&t=368
- source: huggingface_range
- probe fps=25.0 sr=48000 duration=1333.76 has_video=True has_audio=True
- WAV: `/user/HS400/db01550/multimodalbackchannelprediction/dissertation-behaviour-recognition/data/audio_alignment_check/gold_012_6RDkdbgzeAI.wav` duration=60.0 sr=48000 peak=0.41029083728790283 rms=0.03762396052479744
- checks: **PASS** []

## gold_013 (`WrWFSBLjWZU`)

- origin: `gold` split **DEV** (TEST clips are refused)
- person: `p0` → LEFT | sheet: LEFT — watch the person on the LEFT. Ignore the speaker on the right.
- pose frames: 1050–2549 inclusive (1500 frames @ 25 fps → 60.0 s) from t0=42.0 s
- watch clock: 0:42–1:42 | marked nod: 0:59–1:00
- youtube: https://www.youtube.com/watch?v=WrWFSBLjWZU&t=41
- source: huggingface_range
- probe fps=25.0 sr=48000 duration=1286.76 has_video=True has_audio=True
- WAV: `/user/HS400/db01550/multimodalbackchannelprediction/dissertation-behaviour-recognition/data/audio_alignment_check/gold_013_WrWFSBLjWZU.wav` duration=60.0 sr=48000 peak=0.9933469891548157 rms=0.0366768017411232
- checks: **PASS** []

## gold_010 (`jg6y3LABwTs`)

- origin: `gold` split **DEV** (TEST clips are refused)
- person: `p1` → RIGHT | sheet: RIGHT — watch the person on the RIGHT. Ignore the speaker on the left.
- pose frames: 2200–3699 inclusive (1500 frames @ 25 fps → 60.0 s) from t0=88.0 s
- watch clock: 1:28–2:28 | marked nod: 2:07–2:08
- youtube: https://www.youtube.com/watch?v=jg6y3LABwTs&t=88
- source: huggingface_range
- probe fps=25.0 sr=48000 duration=1405.92 has_video=True has_audio=True
- WAV: `/user/HS400/db01550/multimodalbackchannelprediction/dissertation-behaviour-recognition/data/audio_alignment_check/gold_010_jg6y3LABwTs.wav` duration=60.0 sr=48000 peak=1.000030517578125 rms=0.10801517963409424
- checks: **PASS** []

## gold_014 (`f6aNo5Mod9I`)

- origin: `gold` split **DEV** (TEST clips are refused)
- person: `p0` → LEFT | sheet: LEFT — watch the person on the LEFT. Ignore the speaker on the right.
- pose frames: 1325–2824 inclusive (1500 frames @ 25 fps → 60.0 s) from t0=53.0 s
- watch clock: 0:53–1:53 | marked nod: 1:51–1:52
- youtube: https://www.youtube.com/watch?v=f6aNo5Mod9I&t=52
- source: huggingface_range
- probe fps=25.0 sr=48000 duration=1257.41 has_video=True has_audio=True
- WAV: `/user/HS400/db01550/multimodalbackchannelprediction/dissertation-behaviour-recognition/data/audio_alignment_check/gold_014_f6aNo5Mod9I.wav` duration=60.0 sr=48000 peak=0.8323923349380493 rms=0.05014578625559807
- checks: **PASS** []

## gold_009 (`GJtqigeWHV8`)

- origin: `gold` split **DEV** (TEST clips are refused)
- person: `p1` → RIGHT | sheet: RIGHT — watch the person on the RIGHT. Ignore the speaker on the left.
- pose frames: 3100–4599 inclusive (1500 frames @ 25 fps → 60.0 s) from t0=124.0 s
- watch clock: 2:04–3:04 | marked nod: 2:24–2:25
- youtube: https://www.youtube.com/watch?v=GJtqigeWHV8&t=123
- source: huggingface_range
- probe fps=25.0 sr=48000 duration=2769.67 has_video=True has_audio=True
- WAV: `/user/HS400/db01550/multimodalbackchannelprediction/dissertation-behaviour-recognition/data/audio_alignment_check/gold_009_GJtqigeWHV8.wav` duration=60.0 sr=48000 peak=0.9692373275756836 rms=0.05112987011671066
- checks: **PASS** []

## Next step

Step B (`scripts/train_audio_baseline_dev.py`) and Step C (`scripts/train_av_fusion_dev.py`) must **not** run unless this verdict is PASS. They are DEV-only and refuse GOLD TEST.
