# Audio alignment audit (Step 1)

**Verdict: FAIL.** Protocol requires 3–5 existing nod windows with extractable, time-aligned source audio. This run completed **1 / 5** selected DEV windows. GOLD TEST was not used. Steps 2–9 (prosody, HuBERT, RGB matching, fusion, ablation, figures) were **not run**. No F1 invented.

Approved title (unchanged): **Predicting Backchannel Events from Multimodal Conversational Signals**.

The task remains **supervised prediction of the backchannel label associated with a conversational window**, not forecasting.

## Stop rule

Step 1 wall-clock budget is **20 minutes**. Hugging Face Range reads of RealTalk members work from this Mac when the local HTTP proxy is not used (HTTP 206). One 452 MB member (`6RDkdbgzeAI`) was downloaded and one 60.000 s WAV was extracted. Clips 2–5 were not fetched before the budget was exhausted. **Training stops here.**

## Audio source

**Mixed conversation audio** (RealTalk **container soundtrack**). Do **not** call this speaker audio.

Evidence from `gold_012` / `6RDkdbgzeAI` ffmpeg probe of the Range-fetched member:

- Video: MPEG-4, 1280×720, **25 fps**, duration **00:22:13.76**
- Audio: **MP3, 48000 Hz, stereo**, 128 kb/s (both participants in one mix)
- Container metadata title/comment match YouTube id `6RDkdbgzeAI`
- Extracted watch-window WAV was downmixed to **mono** PCM s16le (ffmpeg `-ac 1`); that is a mixdown of the stereo conversation track, not a listener-only or speaker-only channel

There is no separate speaker channel in this RealTalk member.

## Selected windows (DEV / TRAIN only; TEST refused)

Default ids from `src/audio_io.py`: `gold_012`, `gold_013`, `gold_010`, `gold_014`, `gold_009`. All are GOLD **DEV**.

| sample_id | video_id | listener | speaker (audio) | start_s | end_s | duration_s | sample_rate | num_samples | RMS | extracted |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gold_012 | 6RDkdbgzeAI | LEFT p0 | mixed conversation (stereo MP3 48 kHz) | 368.0 | 428.0 | 60.000 | 48000 | 2880000 | 0.03762281 | yes |
| gold_013 | WrWFSBLjWZU | LEFT p0 | not extracted | 42.0 | 102.0 | 60.0 | — | — | — | no |
| gold_010 | jg6y3LABwTs | RIGHT p1 | not extracted | 88.0 | 148.0 | 60.0 | — | — | — | no |
| gold_014 | f6aNo5Mod9I | LEFT p0 | not extracted | 53.0 | 113.0 | 60.0 | — | — | — | no |
| gold_009 | GJtqigeWHV8 | RIGHT p1 | not extracted | 124.0 | 184.0 | 60.0 | — | — | — | no |

Pose windows are 1500 frames @ 25 fps = 60.0 s (`features/gold/*.npz`). Sheet `who_to_watch` matches `p0`/`p1`. YouTube `watch_from`/`watch_until` match pose `t0`/`t1` to 1 s for these ids (committed gold artefacts; no TEST listening).

## gold_012 verification (the only extracted clip)

- **Correct video:** Range member from `videos_01.tar` at index offset 8125140992, size 474089718 bytes. ffmpeg metadata `comment` is `https://www.youtube.com/watch?v=6RDkdbgzeAI`. Probe starts with RIFF/AVI, 25.00 fps, 1280×720.
- **Timestamps:** pose frames 9200–10699 inclusive → t0=368.0 s, t1=428.0 s. Sheet watch 6:08–7:08. ffmpeg `-ss 368 -t 60`.
- **Duration:** WAV **00:01:00.00** (60.000 s). File size 5 760 258 bytes = 44-byte header + 2 880 000 × 2 bytes.
- **Sample count:** **2 880 000** at **48 000 Hz** (ffmpeg volumedetect `n_samples: 2880000`).
- **Non-empty / audible:** peak **0.41027832**, RMS **0.03762281**; volumedetect mean_volume **−28.5 dB**, max_volume **−7.7 dB**. Not silence.
- **Channel identity:** source **stereo conversation mix**, not speaker-only, not listener-only.
- **WAV path (gitignored):** `data/audio_alignment_check/gold_012_6RDkdbgzeAI.wav`

The temporary AVI was deleted after probing to free disk (~452 MB). Re-fetch with the same Range if the file is needed again.

## What blocked a PASS

1. Need **≥ 3** passing clips. Only one WAV exists.
2. Remaining members are large (WrWFSBLjWZU 777 MB, jg6y3LABwTs 1034 MB, f6aNo5Mod9I 813 MB, GJtqigeWHV8 1491 MB). Sequential Range download at ~2.7 MB/s is several minutes per file; the 20-minute stop fired after clip 1.
3. No local `REALTALK_VIDEO_DIR` of `{video_id}.mp4`. No `HF_TOKEN`. Earlier proxied `requests` calls hit **ProxyError 403**; **curl without that proxy returns HTTP 206** and valid AVI bytes.
4. Otter SSH (`otterdiv` → otter48) works, but **`/scratch/db01550/venv` and `~/multimodalbackchannelprediction/.venv` are missing**. Frozen VideoMAE embeddings **are** on otter (`data/features/videomae/`, 110 npz, 3.1 MB). They were **not** copied or used, because fusion is forbidden until alignment PASS.

## Otter commands (do not invent F1; DEV only)

`/scratch/db01550` does not exist on otter48 (checked 27 Aug 2026). System Python is `/usr/bin/python3`. Create a venv, then:

```bash
ssh otterdiv
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
python3 -m venv "$HOME/av_venv"
source "$HOME/av_venv/bin/activate"
python -m pip install librosa imageio-ffmpeg requests numpy scipy scikit-learn
python scripts/audio_alignment_check.py
# only if Verdict: PASS — still GOLD DEV only, never --score-test
python scripts/train_audio_baseline_dev.py
python scripts/train_av_fusion_dev.py
```

Optional: `export REALTALK_VIDEO_DIR=` to a folder of `{video_id}.mp4` if members are cached. `/scratch` has ~165 GB free on otter48 but no `db01550` directory yet.

Refuse `--score-test`. Seed 42. Locked nod TEST remains Fine-tuned VideoMAE last-4, 80 TRAIN, **F1=0.82**. Shake TEST remains locked.

## Planned N (from committed CSVs; not audio-matched)

These counts are **not** a trained matched set. Alignment FAIL means no audio/fusion matrix was built.

- Intended TRAIN (`results/pseudo_labels.csv`): **N=80**, n_pos=70, n_neg=10
- Intended GOLD DEV (`data/gold_annotations.csv`): **N=15**, n_pos=9, n_neg=6 (`gold_001`–`gold_015`)

## Metrics

**not run** (prosody, HuBERT, VideoMAE-on-this-split fusion, 50/50 fusion).
