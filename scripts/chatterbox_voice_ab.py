#!/usr/bin/env python3
"""A/B two reference WAVs against the same Chatterbox /upload line (no FW pipeline).

Usage:
  uv run python scripts/chatterbox_voice_ab.py \\
    --voice-a pipeline_data/speakers/es/SPEAKER_00.wav \\
    --voice-b pipeline_data/speakers/es/SPEAKER_02.wav

  CHATTERBOX_API_URL=http://localhost:8020 uv run python scripts/chatterbox_voice_ab.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    p = argparse.ArgumentParser(description="Chatterbox voice A/B (same text, two reference WAVs).")
    p.add_argument(
        "--base-url",
        default=os.environ.get("CHATTERBOX_API_URL", "http://localhost:8020"),
        help="Chatterbox base URL (default: env CHATTERBOX_API_URL or http://localhost:8020)",
    )
    p.add_argument(
        "--text",
        default="Hola, esta es una prueba corta de la voz.",
        help="Spanish line to synthesize for both references",
    )
    p.add_argument("--voice-a", type=Path, required=True, help="Reference WAV (e.g. male)")
    p.add_argument("--voice-b", type=Path, required=True, help="Reference WAV (e.g. female)")
    p.add_argument("--out-a", type=Path, default=Path("chatterbox_ab_a.wav"))
    p.add_argument("--out-b", type=Path, default=Path("chatterbox_ab_b.wav"))
    p.add_argument(
        "--exaggeration",
        type=float,
        default=None,
        help="Optional: passed as form field if set (travisvn chatterbox-tts-api)",
    )
    args = p.parse_args()

    for label, path in ("A", args.voice_a), ("B", args.voice_b):
        if not path.is_file():
            print(f"error: voice-{label.lower()} not found: {path}", file=sys.stderr)
            return 1

    timeout = (5.0, 120.0)

    def post(voice: Path, out: Path) -> None:
        data = {"input": args.text, "response_format": "wav"}
        if args.exaggeration is not None:
            data["exaggeration"] = str(args.exaggeration)
        base = args.base_url.rstrip("/")
        url = f"{base}/v1/audio/speech/upload"
        with voice.open("rb") as f:
            resp = requests.post(
                url,
                data=data,
                files={"voice_file": (voice.name, f, "audio/wav")},
                timeout=timeout,
            )
        resp.raise_for_status()
        out.write_bytes(resp.content)
        print(f"wrote {out.resolve()} ({len(resp.content)} bytes)  ref={voice}")

    print(f"POST {args.base_url.rstrip('/')}/v1/audio/speech/upload")
    print(f"text: {args.text!r}")
    post(args.voice_a, args.out_a)
    post(args.voice_b, args.out_b)
    print("Listen to the two outputs back-to-back; if they still sound the same, the limit is Chatterbox, not the dubbing pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
