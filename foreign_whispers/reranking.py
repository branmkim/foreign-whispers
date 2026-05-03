"""Deterministic failure analysis and duration-aware translation re-ranking.

``analyze_failures`` uses simple thresholds from ``SegmentMetrics``.
``get_shorter_translations`` produces shorter Spanish candidates via MarianMT,
then optional Ollama / Gemini fallbacks — see its docstring.
"""

import dataclasses
import json
import logging
import os
import re
import urllib.error
import urllib.request
import numpy as np

import foreign_whispers.alignment as alignment

logger = logging.getLogger(__name__)
_MARIAN_CACHE: dict[str, tuple[object, object]] = {}


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
    semantic_lambda: float = 0.5,
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Strategy (in order): **MarianMT** (``Helsinki-NLP/opus-mt-en-es``) on
    *source_text*; if candidates are missing or still too long vs a
    ~15 chars/s budget, **Ollama** (translate / shorten prompts); if still too
    long, **Gemini API** (same). Candidates are sorted by proximity to the
    target character budget.

    Args:
        source_text: Original source-language segment text.
        baseline_es: Baseline target-language translation (e.g. from Argos).
        target_duration_s: Time budget in seconds for this segment.
        context_prev: Preceding segment text (reserved for future coherence).
        context_next: Following segment text (reserved for future coherence).
        semantic_lambda: Reserved for duration+semantic scoring.

    Returns:
        ``TranslationCandidate`` list (may be empty if all backends fail).
    """
    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline) — "
        "returning list of TranslationCandidates",
        target_duration_s,
        len(baseline_es),
    )

    # 1. use MarianMT local model to translate English to Spanish
    # 2. if no candidates are below target characters (+10 leeway), use local LLM
        # make sure ollama is running with `ollama serve`
        #  - TranslateGemma translate English -> Spanish
        #  - TranslateGemma shorten baseline Spanish
    # 3. if still no sufficient candidates, use Gemini API
        #  - Gemini translate English -> Spanish
        #  - Gemini shorten baseline Spanish
    

    CHARS_PER_SECOND = 15
    target_chars = int(target_duration_s * CHARS_PER_SECOND)

    candidates = []

    # add original as candidate
    # candidates.append(TranslationCandidate(
    #     text=baseline_es,
    #     char_count=len(baseline_es),
    #     brevity_rationale="Original",
    # ))

    # add MarianMT candidates
    marian_model_names = [
        "Helsinki-NLP/opus-mt-en-es",
    ]

    for model_name in marian_model_names:
        try:
            translation = _translate_with_marian(source_text, model_name)
            if translation:
                candidates.append(TranslationCandidate(
                    text=translation,
                    char_count=len(translation),
                    brevity_rationale=f"Translated with {model_name}",
                ))
        except Exception as e:
            logger.error(f"Error running MarianMT model {model_name}: {e}")


    # sort by proximity to target characters
    candidates.sort(key=lambda c: abs(c.char_count - target_chars))


    # If local MT candidates are missing or still too long, try local LLM
    if (not candidates) or (candidates[0].char_count > target_chars + 10):
        local_eng = _generate_with_local_llm(
            prompt=(
                f"Translate the following English text to Spanish in under {target_chars} characters. "
                "Preserve meaning and output only translation text.\n\n"
                f"{source_text}"
            )
        )
        if local_eng:
            candidates.append(TranslationCandidate(
                text=local_eng,
                char_count=len(local_eng),
                brevity_rationale="English translation with local LLM (Ollama)",
            ))

        local_es = _generate_with_local_llm(
            prompt=(
                f"Shorten this Spanish text to under {target_chars} characters. "
                "Preserve meaning and output only shortened text.\n\n"
                f"{baseline_es}"
            )
        )
        if local_es:
            candidates.append(TranslationCandidate(
                text=local_es,
                char_count=len(local_es),
                brevity_rationale="Spanish shortened with local LLM (Ollama)",
            ))

        candidates.sort(key=lambda c: abs(c.char_count - target_chars))

    # If local fallbacks are still missing/too long, use Gemini API
    if (not candidates) or (candidates[0].char_count > target_chars + 10):
        try:
            gemini_eng_translation = _generate_with_gemini_api(
                prompt=(
                    f"Translate the following English text to Spanish in under {target_chars} characters. "
                    "Preserve meaning and output only translation text.\n\n"
                    f"{source_text}"
                )
            )
            if gemini_eng_translation:
                candidates.append(TranslationCandidate(
                    text=gemini_eng_translation,
                    char_count=len(gemini_eng_translation),
                    brevity_rationale=f"English translation with Gemini",
                ))

            gemini_es_shortened = _generate_with_gemini_api(
                prompt=(
                    f"Shorten this Spanish text to under {target_chars} characters. "
                    "Preserve meaning and output only shortened text.\n\n"
                    f"{baseline_es}"
                )
            )
            if gemini_es_shortened:
                candidates.append(TranslationCandidate(
                    text=gemini_es_shortened,
                    char_count=len(gemini_es_shortened),
                    brevity_rationale=f"Spanish shortened with Gemini",
                ))
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")


        # sort by proximity to target characters
        candidates.sort(key=lambda c: abs(c.char_count - target_chars))

    return candidates


def _translate_with_marian(source_text: str, model_name: str) -> str:
    from transformers import MarianMTModel, MarianTokenizer

    if model_name in _MARIAN_CACHE:
        tokenizer, model = _MARIAN_CACHE[model_name]
    else:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _MARIAN_CACHE[model_name] = (tokenizer, model)
    tok = tokenizer(source_text, return_tensors="pt").input_ids
    output = model.generate(tok)[0]
    return tokenizer.decode(output, skip_special_tokens=True)


def _sanitize_candidate_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _generate_with_local_llm(prompt: str) -> str:
    """Call a local Ollama model. Requires ``pip/uv`` package ``ollama`` and ``ollama serve`` (or OLLAMA_HOST).

    Returns empty string if the client is missing, the server is unreachable, or the model is absent.
    """
    try:
        from ollama import chat
    except ImportError:
        logger.warning(
            "ollama package not installed; skipping local LLM rerank. "
            "Add it with: uv add ollama"
        )
        return ""
    try:
        response = chat(
            model="translategemma",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning(
            "Ollama chat failed (%s). Is the server running? "
            "From Docker use e.g. OLLAMA_HOST=http://host.docker.internal:11434",
            exc,
        )
        return ""

    result = response.message.content if response and response.message else ""
    return _sanitize_candidate_text(result)

def _generate_with_gemini_api(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )

    return _sanitize_candidate_text(response.text)