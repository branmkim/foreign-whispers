"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
"""
import statistics as _stats

import numpy as np

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)
from foreign_whispers.reranking import _translate_with_marian
from foreign_whispers.roundtrip import segment_tts_stt_intelligibility

# Marian checkpoint for back-translation (ES → EN), same stack as ``reranking._translate_with_marian``.
_MARIAN_ES_EN = "Helsinki-NLP/opus-mt-es-en"


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned_segments: list[AlignedSegment],
    align_report: dict,
    *,
    include_intelligibility_tts_stt: bool = False,
    chatterbox_url: str | None = None,
    whisper_api_url: str | None = None,
) -> dict:
    clip_timing = clip_evaluation_report(metrics, aligned_segments)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError:
        model = None

    back_translations_en: list[dict] = []
    intelligibility_rows: list[dict] = []
    for m in metrics:
        t = (m.translated_text or "").strip()
        if not t:
            back_translations_en.append(
                {
                    "index": m.index,
                    "cosine_similarity": None,
                    "original_text": m.source_text,
                    "translated_text": m.translated_text,
                    "back_translation": "",
                }
            )
            if include_intelligibility_tts_stt:
                intelligibility_rows.append(
                    {
                        "index": m.index,
                        "word_error_rate": 0.0,
                        "intelligibility_score": 1.0,
                        "hypothesis_text": "",
                    }
                )
            continue

        # argos is very slow for me because i do not have access to a GPU
        # so i am using the local MarianMT model instead
        back_translation = _translate_with_marian(t, _MARIAN_ES_EN)

        cosine_similarity: float | None
        if model is not None:
            original_emb = model.encode(
                m.source_text, convert_to_numpy=True, normalize_embeddings=True
            )
            back_translation_emb = model.encode(
                back_translation, convert_to_numpy=True, normalize_embeddings=True
            )
            cosine_similarity = float(np.dot(original_emb, back_translation_emb))
        else:
            cosine_similarity = None

        back_translations_en.append(
            {
                "index": m.index,
                "cosine_similarity": cosine_similarity,
                "original_text": m.source_text,
                "translated_text": m.translated_text,
                "back_translation": back_translation,
            }
        )

        if include_intelligibility_tts_stt:
            intel = segment_tts_stt_intelligibility(
                t,
                chatterbox_url=chatterbox_url,
                whisper_api_url=whisper_api_url,
            )
            intelligibility_rows.append(
                {
                    "index": m.index,
                    "word_error_rate": intel["word_error_rate"],
                    "intelligibility_score": intel["intelligibility_score"],
                    "hypothesis_text": intel["hypothesis_text"],
                }
            )

    cosines = [
        b["cosine_similarity"]
        for b in back_translations_en
        if b.get("cosine_similarity") is not None
    ]
    overall_semantic_similarity = float(np.mean(cosines)) if cosines else 0.0

    out: dict = {
        "clip_evaluation_report": clip_timing,
        "semantic_back_translations_en": back_translations_en,
        "overall_semantic_similarity": overall_semantic_similarity,
    }
    if include_intelligibility_tts_stt:
        wers = [r["word_error_rate"] for r in intelligibility_rows]
        out["intelligibility_tts_stt"] = intelligibility_rows
        out["overall_mean_word_error_rate"] = float(np.mean(wers)) if wers else 0.0
        out["overall_mean_intelligibility_score"] = (
            float(np.mean([r["intelligibility_score"] for r in intelligibility_rows]))
            if intelligibility_rows
            else 0.0
        )
    return out