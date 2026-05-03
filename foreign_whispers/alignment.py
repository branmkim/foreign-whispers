"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.
- ``global_align_dp`` — DP over cumulative drift; may skip an eligible
  ``GAP_SHIFT`` (``REQUEST_SHORTER`` + penalty) to reduce drift for later segments.

No external dependencies — stdlib only.
"""
import dataclasses
import math
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


_SYLLABLE_RATE = 4.5  # syllables per second for Romance languages


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds using a syllable-rate heuristic."""
    n_chars = len(text)
    n_words = len(text.split())

    # these values are from the linear regression model in notebooks/alignment_integration/tts_data_collection.ipynb
    N_CHARS_COEFF = 0.06094574
    N_WORDS_COEFF = 0.06799571
    INTERCEPT = 0.3466867127865596

    return (N_CHARS_COEFF * n_chars) + (N_WORDS_COEFF * n_words) + INTERCEPT


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def _available_gap_after(silence_regions: list[dict], end_s: float) -> float:
    """Seconds of the first silence region that starts near *end_s* (VAD timeline)."""
    for r in silence_regions:
        if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
            return float(r["end_s"] - r["start_s"])
    return 0.0


def _align_one(
    m: SegmentMetrics,
    cumulative_drift: float,
    action: AlignAction,
    gap_shift: float,
    stretch: float,
) -> AlignedSegment:
    sched_start = m.source_start + cumulative_drift
    sched_end   = sched_start + m.source_duration_s + gap_shift
    return AlignedSegment(
        index           = m.index,
        original_start  = m.source_start,
        original_end    = m.source_end,
        scheduled_start = sched_start,
        scheduled_end   = sched_end,
        text            = m.translated_text,
        action          = action,
        gap_shift_s     = gap_shift,
        stretch_factor  = stretch,
    )


def _greedy_step(
    m: SegmentMetrics,
    gap_avail: float,
    max_stretch: float,
) -> tuple[AlignAction, float, float]:
    """One segment's greedy (``decide_action`` + gap/mild mapping). Returns (action, gap_shift, stretch)."""
    action = decide_action(m, available_gap_s=gap_avail)
    gap_shift, stretch = 0.0, 1.0
    if action == AlignAction.GAP_SHIFT:
        gap_shift = m.overflow_s
    elif action == AlignAction.MILD_STRETCH:
        stretch = min(m.predicted_stretch, max_stretch)
    return action, gap_shift, stretch


def _dp_step_candidates(
    m: SegmentMetrics,
    gap_avail: float,
    max_stretch: float,
    penalty_skip_gap: float,
    weight_drift: float,
    weight_mild: float,
) -> list[tuple[AlignAction, float, float, float]]:
    """Return (action, gap_shift, stretch, marginal_cost) options for one segment."""
    action, gap_shift, stretch = _greedy_step(m, gap_avail, max_stretch)
    cost = weight_drift * gap_shift + (weight_mild * (stretch - 1.0) if stretch > 1.0 else 0.0)
    out: list[tuple[AlignAction, float, float, float]] = [(action, gap_shift, stretch, cost)]

    # Optional: skip borrowing silence even though policy would gap-shift — keeps drift for later segments.
    if action == AlignAction.GAP_SHIFT:
        out.append(
            (AlignAction.REQUEST_SHORTER, 0.0, 1.0, float(penalty_skip_gap)),
        )
    return out


def _dp_better_transition(
    new_cost: float,
    new_prev_tick: int,
    new_j: int,
    old_cost: float,
    old_prev_tick: int | None,
    old_j: int | None,
    *,
    cost_eps: float,
) -> bool:
    """Prefer strictly lower cost; on cost tie prefer greedy-like (lower choice index, then lower prior drift)."""
    if new_cost < old_cost - cost_eps:
        return True
    if new_cost > old_cost + cost_eps:
        return False
    if old_prev_tick is None:
        return True
    if new_j != old_j:
        return new_j < old_j
    return new_prev_tick < old_prev_tick


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    *,
    grid_s:           float = 0.05,
    penalty_skip_gap: float = 3.0,
    weight_drift:     float = 1.0,
    weight_mild:      float = 0.05,
) -> list[AlignedSegment]:
    """Global alignment via dynamic programming over cumulative drift.

    At each segment the greedy policy may choose ``GAP_SHIFT``, which adds
    ``overflow_s`` to cumulative drift for all following segments.  This DP
    also considers **not** taking that gap (emitting ``REQUEST_SHORTER`` with
    zero ``gap_shift``) when ``GAP_SHIFT`` would have been valid, trading a
    fixed penalty for lower drift downstream.

    Drift is tracked on a fixed time grid (``grid_s``) using **integer ticks**
    so state keys and back-pointers stay stable; ``_align_one`` receives drift
    in seconds (tick × ``grid_s``), matching the quantized DP semantics.
    """
    if not metrics:
        return []

    if grid_s <= 0.0:
        raise ValueError("grid_s must be positive")

    max_drift = sum(m.overflow_s for m in metrics) + 1.0
    max_tick = max(0, int(math.ceil(max_drift / grid_s)))
    cost_eps = 1e-9

    def tick_to_drift(tick: int) -> float:
        return tick * grid_s

    def snap_combined_to_tick(combined_s: float) -> int:
        capped = min(max(combined_s, 0.0), max_drift)
        t = int(round(capped / grid_s))
        return max(0, min(t, max_tick))

    # tick after processing prefix -> min total cost
    cur: dict[int, float] = {0: 0.0}
    # (segment_index, tick_after) -> (prev_tick, choice_index)
    back_prev: dict[tuple[int, int], tuple[int, int]] = {}
    choice_lists: list[list[tuple[AlignAction, float, float, float]]] = []

    for i, m in enumerate(metrics):
        gap_avail = _available_gap_after(silence_regions, m.source_end)
        choices = _dp_step_candidates(
            m, gap_avail, max_stretch, penalty_skip_gap, weight_drift, weight_mild,
        )
        choice_lists.append(choices)
        nxt_cost: dict[int, float] = {}
        nxt_meta: dict[int, tuple[int, int]] = {}
        for tick_prev, cost_prev in cur.items():
            if tick_to_drift(tick_prev) > max_drift + grid_s:
                continue
            drift_prev_s = tick_to_drift(tick_prev)
            for j, (act, g, st, step_cost) in enumerate(choices):
                new_tick = snap_combined_to_tick(drift_prev_s + g)
                c_new = cost_prev + step_cost
                old_c = nxt_cost.get(new_tick, float("inf"))
                old_meta = nxt_meta.get(new_tick)
                if old_meta is None:
                    old_prev, old_j = None, None
                else:
                    old_prev, old_j = old_meta
                if _dp_better_transition(
                    c_new, tick_prev, j, old_c, old_prev, old_j, cost_eps=cost_eps,
                ):
                    nxt_cost[new_tick] = c_new
                    nxt_meta[new_tick] = (tick_prev, j)
                    back_prev[(i, new_tick)] = (tick_prev, j)

        cur = nxt_cost

    if not cur:
        return global_align(metrics, silence_regions, max_stretch)

    best_tick = min(cur, key=lambda t: (cur[t], t))

    aligned_rev: list[AlignedSegment] = []
    tick_after = best_tick
    for i in range(len(metrics) - 1, -1, -1):
        key = (i, tick_after)
        if key not in back_prev:
            return global_align(metrics, silence_regions, max_stretch)
        tick_prev, j = back_prev[key]
        act, g, st, _ = choice_lists[i][j]
        d_prev = tick_to_drift(tick_prev)
        aligned_rev.append(_align_one(metrics[i], d_prev, act, g, st))
        tick_after = tick_prev

    aligned_rev.reverse()
    return aligned_rev


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - See ``global_align_dp`` for a small DP that can skip eligible gap-shifts
      to reduce downstream drift.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        gap_avail = _available_gap_after(silence_regions, m.source_end)
        action, gap_shift, stretch = _greedy_step(m, gap_avail, max_stretch)

        aligned.append(_align_one(m, cumulative_drift, action, gap_shift, stretch))
        cumulative_drift += gap_shift

    return aligned
