"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import contextlib
import functools
import inspect
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _trusted_torch_checkpoint_load() -> None:
    """Use legacy ``torch.load`` semantics while loading official HF/pyannote weights.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``. Checkpoints for
    ``pyannote/speaker-diarization-3.1`` are pickled with types that fail that
    path. This project only enables the permissive loader **around** loading and
    running that **trusted** Hugging Face pipeline—not for arbitrary files.
    """
    import torch

    real_load = torch.load

    @functools.wraps(real_load)
    def load_trusted(*args, **kwargs):
        # Libraries may pass weights_only=True explicitly; setdefault would not override it.
        params = inspect.signature(real_load).parameters
        if "weights_only" in params:
            kwargs["weights_only"] = False
        return real_load(*args, **kwargs)

    torch.load = load_trusted
    try:
        # If anything still calls torch.load(..., weights_only=True), allowlist types
        # the official pyannote checkpoint uses (PyTorch error message suggests this).
        safe = getattr(torch.serialization, "safe_globals", None)
        if safe is not None:
            try:
                from torch.torch_version import TorchVersion
            except ImportError:
                yield
            else:
                with safe([TorchVersion]):
                    yield
        else:
            yield
    finally:
        torch.load = real_load


def _patch_torchaudio_for_pyannote() -> None:
    """pyannote.audio 3.4 expects torchaudio APIs removed in torchaudio 2.2+.

    Without this, ``from pyannote.audio import Pipeline`` raises ``AttributeError``
    (e.g. ``AudioMetaData``, ``list_audio_backends``), which breaks the FastAPI
    diarize route with a non-JSON 500 body.
    """
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class _AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        setattr(torchaudio, "AudioMetaData", _AudioMetaData)

    if not hasattr(torchaudio, "list_audio_backends"):
        setattr(torchaudio, "list_audio_backends", lambda: ["soundfile", "ffmpeg"])

    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _info(path, *args, **kwargs):
            info = sf.info(path)
            bits = info.subtype_info
            bits_per_sample = 0
            if bits:
                import re

                m = re.search(r"(\d+)", bits)
                if m:
                    bits_per_sample = int(m.group(1))
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=bits_per_sample,
                encoding=info.subtype or "UNKNOWN",
            )

        setattr(torchaudio, "info", _info)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        _patch_torchaudio_for_pyannote()
        from pyannote.audio import Pipeline
    except (ImportError, TypeError, AttributeError) as exc:
        logger.warning(
            "pyannote.audio unavailable (%s) — returning empty diarization.",
            exc,
        )
        return []

    try:
        with _trusted_torch_checkpoint_load():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            diarization = pipeline(audio_path)
            return [
                {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    # ---- YOUR CODE HERE ----
    new_segments = []

    for segment in segments:
        speakers = {}  # {speaker: duration}
        for diarization_interval in diarization:
            # check for any partial or full overlap
            seg_start = segment["start"]
            seg_end = segment["end"]
            dia_start = diarization_interval["start_s"]
            dia_end = diarization_interval["end_s"]

            # There is overlap if segment starts before diarization ends and segment ends after diarization starts
            if seg_start < dia_end and seg_end > dia_start:
                duration = min(seg_end, dia_end) - max(seg_start, dia_start)
                speakers[diarization_interval["speaker"]] = speakers.get(diarization_interval["speaker"], 0) + duration

        if speakers:
            speaker = max(speakers, key=speakers.get)
            new_segments.append({**segment, "speaker": speaker})
        else:
            new_segments.append({**segment, "speaker": "SPEAKER_00"})

    return new_segments
    # ---- END YOUR CODE ----