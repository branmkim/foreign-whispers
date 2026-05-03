# Foreign Whispers

Brandon Kim, bmk7319@nyu.edu

Pipeline: download a YouTube video, transcribe (here: YouTube captions), translate with argos, synthesize Spanish with Chatterbox, time-align and stitch a dubbed MP4. Original README and architecture are in [README_old.md](./README_old.md).

[Link to sample output (Google Drive)](https://drive.google.com/file/d/12xoGLLQV-Fk56nBoRY_jZ_dLratk_Nia/view?usp=sharing)

In the Google Drive video, captions will not show by default. Turn them on with the [CC] icon in the bottom-right.

## Environment

I did not have access to an NVIDIA GPU for this assignment, so I ran the Chatterbox TTS API on a RunPod instance instead. The TTS client reads **`CHATTERBOX_API_URL`** (see [`api/src/services/tts_engine.py`](api/src/services/tts_engine.py) and the `api` service env in [`docker-compose.yml`](docker-compose.yml)). I hit connection and rate-limit issues while wiring the Mac-hosted API to RunPod; the final run matches what you would expect from a local GPU TTS container.

I did not run the Whisper STT container, I instead used YouTube captions for transcription. The frontend and API containers ran on my Mac without issue.

## Transcription changes

YouTube captions use overlapping rolling windows (e.g. segment 1 is 0:00–0:05 while segment 2 is 0:03–0:07). Downstream stages concatenate segment audio in wall-clock order, so overlapping intervals double-count time and break sync. [`foreign_whispers/segment_merge.py`](foreign_whispers/segment_merge.py) tiles each line into non-overlapping `[start, end)` windows before translate/TTS; see [`api/src/routers/transcribe.py`](api/src/routers/transcribe.py) and tests in [`tests/test_segment_merge.py`](tests/test_segment_merge.py).

The dubbed audio track also applies a small speech start offset when YouTube’s first caption start differs from the first transcript segment (`_compute_speech_offset` in [`api/src/services/tts_engine.py`](api/src/services/tts_engine.py)), so TTS lines up with when speech actually begins in the source video.

## Alignment

Task 1 (duration predictor) is in [`notebooks/alignment_integration/tts_data_collection.ipynb`](notebooks/alignment_integration/tts_data_collection.ipynb). On the Hormuz clip I collected 98 ground-truth Spanish TTS segment durations, fit a linear model from Spanish character and word counts to duration, and copied the coefficients into `_estimate_duration` in [`foreign_whispers/alignment.py`](foreign_whispers/alignment.py). Further work could add features (e.g. vowel clusters as a syllable proxy).