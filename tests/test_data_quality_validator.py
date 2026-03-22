"""Tests for DataQualityValidator (DQP compliance)."""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.data_quality_validator import (
    DataQualityValidator,
    DQPReport,
    QualityCheckResult,
)


def _good_entities():
    return [
        {"label": "PERSON"},
        {"label": "DATE"},
        {"label": "PHONE"},
        {"label": "MRN"},
        {"label": "AGE"},
        {"label": "LOCATION"},
    ]


def test_validate_note_returns_report(tmp_path):
    db = tmp_path / "dq.db"
    v = DataQualityValidator(db_path=str(db), strict_mode=False)
    orig = "Patient seen. Phone 555-1212. MRN123. " * 4
    entities = _good_entities()
    proc = "[PERSON] [DATE] [PHONE] [MRN] [AGE] [LOCATION] " + "filler. " * 20
    report = v.validate_note(1, orig, proc, entities)
    assert isinstance(report, DQPReport)
    assert report.note_id == 1
    assert len(report.checks) == 5
    assert report.generated_at


def test_validate_note_strict_mode(tmp_path):
    db = tmp_path / "dq2.db"
    v = DataQualityValidator(db_path=str(db), strict_mode=True)
    report = v.validate_note(
        2,
        "Original text here.",
        "Masked [DATE] ok.",
        [{"label": "DATE"}, {"label": "MRN"}],
    )
    assert isinstance(report.passed, bool)


def test_validate_batch(tmp_path):
    db = tmp_path / "dq3.db"
    v = DataQualityValidator(db_path=str(db))
    df = pd.DataFrame(
        [
            {
                "note_id": 1,
                "original_text": "A" * 100,
                "masked_text": "B" * 100,
                "entities_json": json.dumps([{"label": "MRN"}, {"label": "DATE"}]),
            }
        ]
    )
    out = v.validate_batch(df)
    assert len(out) == 1
    assert "quality_score" in out.columns


def test_detect_anomalies(tmp_path):
    db = tmp_path / "dq4.db"
    v = DataQualityValidator(db_path=str(db))
    rows = []
    for i in range(15):
        rows.append(
            {
                "original_text": "x" * (50 + i * 10),
                "masked_text": "y" * (40 + i * 5),
                "entity_count": i,
            }
        )
    df = pd.DataFrame(rows)
    out = v.detect_anomalies(df, contamination=0.1)
    assert "anomaly_score" in out.columns
    assert "is_anomaly" in out.columns


def test_generate_quality_summary(tmp_path):
    db = tmp_path / "dq5.db"
    v = DataQualityValidator(db_path=str(db))
    v.validate_note(9, "orig", "proc", [{"label": "MRN"}])
    summary = v.generate_quality_summary()
    assert "summary" in summary
    assert "by_check" in summary


def test_validate_note_excessive_text_loss(tmp_path):
    db = tmp_path / "dq6.db"
    v = DataQualityValidator(db_path=str(db))
    long_o = "Sentence one. Sentence two. " * 20
    short_p = "x"
    entities = [{"label": "MRN"}, {"label": "DATE"}, {"label": "PHONE"}, {"label": "PERSON"}, {"label": "AGE"}, {"label": "LOCATION"}]
    report = v.validate_note(42, long_o, short_p, entities)
    assert any(c.check_name == "completeness" for c in report.checks)


def test_quality_check_result_dataclass():
    q = QualityCheckResult(
        check_name="t",
        passed=True,
        score=1.0,
        issues=[],
        severity="minor",
        timestamp="2024-01-01",
    )
    assert q.check_name == "t"
