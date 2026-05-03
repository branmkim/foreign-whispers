# tests/test_roundtrip.py
from unittest.mock import MagicMock

import pytest

from foreign_whispers.roundtrip import tts_stt_roundtrip_scores, word_error_rate


def test_word_error_rate_perfect():
    assert word_error_rate("hola mundo", "Hola mundo") == 0.0


def test_word_error_rate_one_substitution():
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)


def test_word_error_rate_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hello") == 1.0


def test_tts_stt_roundtrip_scores_with_injected_model():
    """No ``import whisper`` / load_model when *whisper_model* is injected."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "hola mundo"}

    out = tts_stt_roundtrip_scores(
        "hola mundo",
        "/fake/path.wav",
        whisper_model=fake_model,
    )

    assert out["word_error_rate"] == 0.0
    assert out["intelligibility_score"] == 1.0
    assert out["hypothesis_text"] == "hola mundo"
    fake_model.transcribe.assert_called_once()
