"""Non-overlapping timeline for transcript segments (rolling captions safe).

YouTube auto captions use **rolling windows**: each line overlaps the next in
wall-clock time. **Transitive overlap-merge** would collapse the entire clip
into one segment.

Instead we **partition** time: each segment keeps its own ``text``, but its
``end`` becomes ``min(original_end, next_segment.start)`` so windows tile
``[start_i, start_{i+1})``-style without duplicating wall-clock in downstream TTS
concatenation.

Whisper segments with the same overlap pattern are handled the same way.
"""

from __future__ import annotations

import math


def exclusive_segment_timeline(
    segments: list[dict],
    *,
    tolerance_s: float = 1e-3,
) -> list[dict]:
    """Return segments with non-overlapping ``[start, end)`` times, one text line each.

    Segments are sorted by ``start`` then ``end``. For each row ``i``,
    ``end`` is set to ``min(original_end_i, start_{i+1})`` when a successor
    exists; the last row keeps ``original_end``. Rows that collapse to
    zero or negative duration (after ``tolerance_s``) are dropped.

    Preserves ``speaker`` (and other keys except ``id`` / ``start`` / ``end``)
    from each source row when present.
    """
    tol = tolerance_s
    rows: list[dict] = []
    for s in segments:
        try:
            start = float(s["start"])
            end = float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        rows.append({"start": start, "end": end, "text": text, "_raw": dict(s)})

    if not rows:
        return []

    rows.sort(key=lambda r: (r["start"], r["end"]))

    out: list[dict] = []
    n = len(rows)
    for i, r in enumerate(rows):
        s = r["start"]
        raw_end = r["end"]
        if i + 1 < n:
            next_start = rows[i + 1]["start"]
            e = min(raw_end, next_start)
        else:
            e = raw_end
        if e <= s + tol:
            continue
        seg: dict = {"id": len(out), "start": s, "end": e, "text": r["text"]}
        raw = r["_raw"]
        if raw.get("speaker"):
            seg["speaker"] = raw["speaker"]
        out.append(seg)

    return out


def transcript_with_exclusive_timeline(transcript: dict, *, tolerance_s: float = 1e-3) -> dict:
    """Return a copy of *transcript* with exclusive windows and ``text`` rebuilt from segments."""
    fixed = exclusive_segment_timeline(
        transcript.get("segments") or [],
        tolerance_s=tolerance_s,
    )
    full = " ".join(s.get("text", "").strip() for s in fixed)
    return {**transcript, "segments": fixed, "text": full.strip()}
