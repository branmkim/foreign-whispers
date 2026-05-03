"""Tests for exclusive caption / transcript timeline (rolling YouTube captions)."""

import json
from pathlib import Path

from foreign_whispers.segment_merge import exclusive_segment_timeline, transcript_with_exclusive_timeline


def test_youtube_style_rolling_stays_many_segments_not_one():
    """Overlapping rolling lines must not collapse to a single merged blob."""
    segs = [
        {"id": 0, "start": 2.32, "end": 6.12, "text": "60 Minutes overtime."},
        {"id": 1, "start": 6.48, "end": 10.24, "text": "What's the worst case scenario that"},
        {"id": 2, "start": 8.0, "end": 12.4, "text": "you're worried about is that it is"},
    ]
    out = exclusive_segment_timeline(segs)
    assert len(out) == 3
    assert out[0]["start"] == 2.32 and abs(out[0]["end"] - 6.12) < 0.02
    assert out[0]["text"] == "60 Minutes overtime."
    assert out[1]["start"] == 6.48 and abs(out[1]["end"] - 8.0) < 0.02
    assert out[2]["start"] == 8.0 and abs(out[2]["end"] - 12.4) < 0.02


def test_non_overlapping_unchanged_end():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 2.0, "end": 4.0, "text": "two"},
    ]
    out = exclusive_segment_timeline(segs)
    assert len(out) == 2
    assert out[0]["end"] == 2.0
    assert out[1]["end"] == 4.0


def test_last_segment_keeps_original_end():
    segs = [
        {"start": 0.0, "end": 2.5, "text": "a"},
        {"start": 1.0, "end": 3.0, "text": "b"},
    ]
    out = exclusive_segment_timeline(segs)
    assert out[-1]["end"] == 3.0


def test_transcript_rebuilds_full_text():
    trans = {
        "language": "en",
        "text": "ignore",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 1.0, "end": 3.0, "text": "world"},
        ],
    }
    out = transcript_with_exclusive_timeline(trans)
    assert out["text"] == "hello world"
    assert len(out["segments"]) == 2


def test_strait_of_hormuz_youtube_caption_file_many_segments():
    """Regression: real YouTube caption file must not become a single segment."""
    path = Path(
        "pipeline_data/api/youtube_captions/"
        "Strait of Hormuz disruption threatens to shake global economy.txt"
    )
    if not path.is_file():
        return
    segments = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        seg = json.loads(line)
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        duration = seg.get("duration", 0)
        if not text or duration <= 0:
            continue
        segments.append({"id": i, "start": start, "end": start + duration, "text": text})
    out = exclusive_segment_timeline(segments)
    assert len(out) >= 50
    assert len(out) == len(segments)
    assert out[-1]["end"] == segments[-1]["end"]
