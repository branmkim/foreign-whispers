"""TTS → STT round-trip helpers for intelligibility (WER vs reference text).

Uses the same HTTP surfaces as the API stack: Chatterbox ``/v1/audio/speech``
and Speaches-style ``/v1/audio/transcriptions`` (see ``api/src/inference/whisper_remote.py``).
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import tempfile
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _tokenize(s: str) -> list[str]:
    return re.findall(r"\S+", s.strip().lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level error rate: edit distance on token lists / len(reference tokens)."""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    r_n, h_n = len(ref), len(hyp)
    dp = [[0] * (h_n + 1) for _ in range(r_n + 1)]
    for i in range(r_n + 1):
        dp[i][0] = i
    for j in range(h_n + 1):
        dp[0][j] = j
    for i in range(1, r_n + 1):
        for j in range(1, h_n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[r_n][h_n] / max(r_n, 1)


def _remote_transcribe_verbose_json(audio_path: str, api_url: str) -> dict[str, Any]:
    url = f"{api_url.rstrip('/')}/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (pathlib.Path(audio_path).name, f, "audio/wav")},
            data={"response_format": "verbose_json", "language": "es"},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def tts_stt_roundtrip_scores(
    reference_text: str,
    wav_path: str,
    *,
    whisper_model: Any | None = None,
    whisper_api_url: str | None = None,
) -> dict[str, Any]:
    """Run STT on *wav_path* and score how well the transcript matches *reference_text*.

    When *whisper_model* is provided, calls ``whisper_model.transcribe(wav_path)``
    and expects a dict with a ``text`` field (used in tests).

    Otherwise POSTs the WAV to the remote OpenAI-compatible Whisper service
    (``FW_WHISPER_API_URL``, default ``http://localhost:8000``).
    """
    if whisper_model is not None:
        raw = whisper_model.transcribe(wav_path)
    else:
        base = whisper_api_url or os.environ.get("FW_WHISPER_API_URL", "http://localhost:8000")
        raw = _remote_transcribe_verbose_json(wav_path, base)

    hyp = (raw.get("text") or "").strip()
    wer = word_error_rate(reference_text, hyp)
    return {
        "word_error_rate": wer,
        "intelligibility_score": max(0.0, 1.0 - wer),
        "hypothesis_text": hyp,
    }


def _chatterbox_tts_to_wav(spanish_text: str, output_wav: str, *, chatterbox_url: str) -> None:
    base = chatterbox_url.rstrip("/")
    resp = requests.post(
        f"{base}/v1/audio/speech",
        json={"input": spanish_text, "response_format": "wav"},
        timeout=(5, 120),
    )
    resp.raise_for_status()
    pathlib.Path(output_wav).write_bytes(resp.content)


def segment_tts_stt_intelligibility(
    reference_es: str,
    *,
    chatterbox_url: str | None = None,
    whisper_api_url: str | None = None,
    whisper_model: Any | None = None,
) -> dict[str, Any]:
    """TTS *reference_es* (Spanish) to a temp WAV, STT it, return WER vs *reference_es*."""
    t = (reference_es or "").strip()
    if not t:
        return {
            "word_error_rate": 0.0,
            "intelligibility_score": 1.0,
            "hypothesis_text": "",
        }

    base_tts = (
        chatterbox_url
        or os.environ.get("CHATTERBOX_API_URL")
        or os.environ.get("FW_CHATTERBOX_API_URL")
        or "http://localhost:8020"
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _chatterbox_tts_to_wav(t, tmp_path, chatterbox_url=base_tts)
        return tts_stt_roundtrip_scores(
            t,
            tmp_path,
            whisper_model=whisper_model,
            whisper_api_url=whisper_api_url,
        )
    except OSError as exc:
        logger.warning("roundtrip temp wav failed: %s", exc)
        raise
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
