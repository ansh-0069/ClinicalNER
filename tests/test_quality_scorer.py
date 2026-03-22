"""Tests for DataQualityScorer (DQP-style note scoring)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.quality_scorer import DataQualityScorer, QualityResult


def test_scorer_production_ready_grade():
    s = DataQualityScorer()
    text = "Patient note " + "word " * 100
    entities = [
        {"label": "MRN"},
        {"label": "DATE"},
        {"label": "DOB"},
        {"label": "PHONE"},
    ]
    r = s.score(1, text, entities, is_valid=True, avg_confidence=0.95)
    assert isinstance(r, QualityResult)
    assert r.grade == "A"
    assert r.score >= 90
    d = r.to_dict()
    assert d["note_id"] == 1
    assert "breakdown" in d


def test_scorer_failed_validation():
    s = DataQualityScorer()
    r = s.score(2, "x" * 400, [{"label": "MRN"}], is_valid=False, avg_confidence=0.9)
    assert r.grade == "D"
    assert "validation" in r.label.lower() or "failed" in r.label.lower()


def test_scorer_short_text_flags():
    s = DataQualityScorer()
    r = s.score(3, "short", [{"label": "MRN"}], is_valid=True, avg_confidence=0.5)
    assert any("short" in f.lower() or "minimum" in f.lower() for f in r.flags) or r.score < 90


def test_scorer_low_confidence_flag():
    s = DataQualityScorer()
    r = s.score(
        4,
        "x" * 200,
        [{"label": "MRN"}, {"label": "DATE"}],
        is_valid=True,
        avg_confidence=0.5,
    )
    assert r.breakdown["ner_confidence"] < 20


def test_scorer_marginal_grade():
    s = DataQualityScorer()
    r = s.score(5, "x" * 85, [{"label": "MRN"}], is_valid=True, avg_confidence=0.65)
    assert r.grade in ("B", "C", "D", "A")


def test_scorer_single_entity_type_flag():
    s = DataQualityScorer()
    r = s.score(6, "x" * 200, [{"label": "MRN"}] * 3, is_valid=True, avg_confidence=0.95)
    assert any("one phi" in f.lower() or "one phi type" in f.lower() for f in r.flags)
