# tests/test_evaluation.py
from foreign_whispers.alignment import compute_segment_metrics, global_align
from foreign_whispers.evaluation import clip_evaluation_report, dubbing_scorecard


def _make_transcripts(src_dur=3.0, tgt_chars=30):
    en = {"segments": [{"start": 0.0, "end": src_dur, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": src_dur, "text": "x" * tgt_chars}]}
    return en, es


def test_report_keys():
    en, es = _make_transcripts()
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert set(report.keys()) == {
        "mean_abs_duration_error_s",
        "pct_severe_stretch",
        "n_gap_shifts",
        "n_translation_retries",
        "total_cumulative_drift_s",
    }


def test_report_no_issues_for_easy_segment():
    en, es = _make_transcripts(src_dur=3.0, tgt_chars=15)  # 1s predicted, 3s budget
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_gap_shifts"] == 0
    assert report["n_translation_retries"] == 0
    assert report["total_cumulative_drift_s"] == 0.0


def test_report_counts_retries_for_hard_segment():
    # 1s budget, 9 syllables (ba*9) → ~2.0s predicted → REQUEST_SHORTER
    en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 9}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_translation_retries"] == 1


def test_report_empty_inputs():
    report = clip_evaluation_report([], [])
    assert report["mean_abs_duration_error_s"] == 0.0
    assert report["n_gap_shifts"] == 0


def test_scorecard_timing_keys_and_range():
    en, es = _make_transcripts()
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    card = dubbing_scorecard(metrics, aligned, report)
    assert "timing" in card and "timing_detail" in card
    assert 0.0 <= card["timing"] <= 1.0
    td = card["timing_detail"]
    for k in (
        "duration_error_score",
        "severe_stretch_score",
        "drift_score",
        "mean_abs_duration_error_s",
        "pct_severe_stretch",
        "total_cumulative_drift_s",
    ):
        assert k in td
    assert 0.0 <= td["duration_error_score"] <= 1.0


def test_scorecard_stale_align_report_does_not_override_timing():
    """Recomputed clip report wins over contradictory align_report."""
    en, es = _make_transcripts()
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    real = clip_evaluation_report(metrics, aligned)
    fake = {**real, "mean_abs_duration_error_s": 99.0, "pct_severe_stretch": 100.0}
    card = dubbing_scorecard(metrics, aligned, fake)
    assert card["timing_detail"]["mean_abs_duration_error_s"] == real["mean_abs_duration_error_s"]
    assert card["timing_detail"]["pct_severe_stretch"] == real["pct_severe_stretch"]


def test_scorecard_empty_clip():
    card = dubbing_scorecard([], [], {})
    assert card["timing"] == 1.0
    assert card["timing_detail"]["duration_error_score"] == 1.0
