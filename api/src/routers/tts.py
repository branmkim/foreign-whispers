"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from foreign_whispers.voice_resolution import resolve_speaker_wav

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _speaker_voice_map_for_transcript(
    trans_path: pathlib.Path,
    *,
    target_language: str,
    explicit_speaker_wav: str | None,
) -> dict[str, str] | None:
    """One resolved reference WAV per diarized speaker id (or a single explicit path)."""
    if not trans_path.is_file():
        return None
    data = json.loads(trans_path.read_text(encoding="utf-8"))
    segments = [s for s in data.get("segments", []) if isinstance(s, dict)]
    if not segments:
        return None
    unique = sorted({(s.get("speaker") or "SPEAKER_00") for s in segments})
    if explicit_speaker_wav is not None:
        return {spk: explicit_speaker_wav for spk in unique}
    return {
        spk: resolve_speaker_wav(settings.speakers_dir, target_language, spk)
        for spk in unique
    }


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(
        None,
        description="Reference voice WAV relative to pipeline_data/speakers (e.g. es/default.wav)",
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *speaker_wav* forces one reference voice for all segments; if omitted, voices are resolved per speaker.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")
    trans_path = pathlib.Path(source_path)
    speaker_voice_map = _speaker_voice_map_for_transcript(
        trans_path,
        target_language="es",
        explicit_speaker_wav=speaker_wav,
    )

    print("speaker_voice_map", speaker_voice_map)

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_voice_map=speaker_voice_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
