"""
test_anomaly_detector.py
─────────────────────────
Tests for AnomalyDetector and the /api/anomaly-scan Flask route.
Run: pytest tests/test_anomaly_detector.py -v
"""

import sys
sys.path.insert(0, ".")

import pytest
import numpy as np
from src.pipeline.anomaly_detector import AnomalyDetector, AnomalyResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_note(phi_count=3, text_len=300, labels=None, note_id=1):
    labels = labels or ["DATE", "PHONE", "MRN"]
    entities = [{"label": l} for l in labels[:phi_count]]
    return {"id": note_id, "text": "x" * text_len, "entities": entities}

def normal_corpus(n=50):
    return [make_note(phi_count=3, text_len=300, note_id=i) for i in range(n)]


@pytest.fixture
def fitted_detector():
    d = AnomalyDetector(contamination=0.05)
    d.fit(normal_corpus(50))
    return d


# ── Fit & predict ─────────────────────────────────────────────────────────────

def test_fit_requires_min_10_notes():
    d = AnomalyDetector()
    with pytest.raises(ValueError, match="at least 10"):
        d.fit([make_note() for _ in range(5)])

def test_fit_sets_is_fitted():
    d = AnomalyDetector()
    d.fit(normal_corpus(20))
    assert d.is_fitted is True

def test_predict_requires_fit():
    d = AnomalyDetector()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        d.predict([make_note()])

def test_predict_returns_one_result_per_note(fitted_detector):
    notes   = [make_note(note_id=i) for i in range(5)]
    results = fitted_detector.predict(notes)
    assert len(results) == 5

def test_predict_result_is_anomaly_result_type(fitted_detector):
    results = fitted_detector.predict([make_note()])
    assert isinstance(results[0], AnomalyResult)

def test_anomaly_score_non_negative(fitted_detector):
    results = fitted_detector.predict(normal_corpus(10))
    assert all(r.anomaly_score >= 0.0 for r in results)

def test_fit_predict_convenience(fitted_detector):
    d = AnomalyDetector()
    results = d.fit_predict(normal_corpus(20))
    assert len(results) == 20

def test_note_id_preserved(fitted_detector):
    notes   = [make_note(note_id=42)]
    results = fitted_detector.predict(notes)
    assert results[0].note_id == 42


# ── Anomaly detection quality ─────────────────────────────────────────────────

def test_extreme_outlier_flagged_as_anomaly():
    """Note with 50 entities in 20 chars should be flagged."""
    d = AnomalyDetector(contamination=0.1)
    corpus  = normal_corpus(50)
    extreme = make_note(phi_count=50, text_len=20, note_id=99)
    d.fit(corpus + [extreme])
    results = d.predict([extreme])
    assert results[0].is_anomaly is True

def test_normal_notes_mostly_not_anomalies(fitted_detector):
    notes   = normal_corpus(40)
    results = fitted_detector.predict(notes)
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    assert anomaly_count <= 5     # contamination=0.05 → ≤5% on normal data


# ── Flag generation ───────────────────────────────────────────────────────────

def test_empty_phi_generates_flag(fitted_detector):
    note    = {"id": 1, "text": "A normal clinical note with no PHI.", "entities": []}
    results = fitted_detector.predict([note])
    assert any("No PHI detected" in f for f in results[0].flags)

def test_short_note_generates_flag(fitted_detector):
    note    = {"id": 1, "text": "Short.", "entities": []}
    results = fitted_detector.predict([note])
    assert any("short" in f.lower() for f in results[0].flags)

def test_high_density_generates_flag(fitted_detector):
    entities = [{"label": "DATE"} for _ in range(20)]
    note     = {"id": 1, "text": "x" * 200, "entities": entities}
    results  = fitted_detector.predict([note])
    assert any("high PHI density" in f for f in results[0].flags)

def test_flags_is_list(fitted_detector):
    results = fitted_detector.predict([make_note()])
    assert isinstance(results[0].flags, list)


# ── Summary stats ─────────────────────────────────────────────────────────────

def test_summary_has_required_keys(fitted_detector):
    results = fitted_detector.predict(normal_corpus(10))
    summary = fitted_detector.summary(results)
    for key in ["total_notes", "anomalies_found", "anomaly_rate",
                "avg_score", "top_flags"]:
        assert key in summary

def test_summary_total_matches_input(fitted_detector):
    results = fitted_detector.predict(normal_corpus(15))
    summary = fitted_detector.summary(results)
    assert summary["total_notes"] == 15

def test_anomaly_rate_in_range(fitted_detector):
    results = fitted_detector.predict(normal_corpus(20))
    summary = fitted_detector.summary(results)
    assert 0.0 <= summary["anomaly_rate"] <= 1.0

def test_to_dict_has_required_keys(fitted_detector):
    results = fitted_detector.predict([make_note()])
    d = results[0].to_dict()
    for key in ["note_id", "anomaly_score", "is_anomaly", "flags"]:
        assert key in d


# ── Flask route ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from src.api.app import create_app
    app = create_app(db_path=":memory:")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_anomaly_scan_route_exists(client):
    notes = [make_note(note_id=i) for i in range(15)]
    res   = client.post("/api/anomaly-scan", json={"notes": notes})
    assert res.status_code == 200

def test_anomaly_scan_requires_10_notes(client):
    notes = [make_note(note_id=i) for i in range(5)]
    res   = client.post("/api/anomaly-scan", json={"notes": notes})
    assert res.status_code == 400

def test_anomaly_scan_returns_results_list(client):
    notes = [make_note(note_id=i) for i in range(15)]
    res   = client.post("/api/anomaly-scan", json={"notes": notes})
    data  = res.get_json()
    assert "results" in data
    assert len(data["results"]) == 15

def test_anomaly_scan_returns_summary_keys(client):
    notes = [make_note(note_id=i) for i in range(15)]
    res   = client.post("/api/anomaly-scan", json={"notes": notes})
    data  = res.get_json()
    for key in ["total_notes", "anomalies_found", "anomaly_rate"]:
        assert key in data
