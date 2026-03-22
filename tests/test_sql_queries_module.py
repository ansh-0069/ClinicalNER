"""Import and exercise sql_queries catalog (Associate CP SQL proficiency)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import sql_queries as sq


def test_query_catalog_keys():
    assert "study_summary" in sq.QUERY_CATALOG
    assert "high_risk_notes" in sq.QUERY_CATALOG
    assert len(sq.QUERY_CATALOG) >= 8


def test_template_functions_contain_sql():
    q = sq.get_notes_by_date_range("2024-01-01", "2024-12-31")
    assert "processed_notes" in q
    assert "2024-01-01" in q

    q2 = sq.get_quality_report_for_specialty("Cardiology")
    assert "quality_checks" in q2
    assert "Cardiology" in q2


def test_constant_queries_are_non_empty():
    assert "clinical_notes" in sq.STUDY_SUMMARY
    assert "audit_log" in sq.AUDIT_TRAIL_SUMMARY
    assert "phi" in sq.PHI_BY_SPECIALTY.lower()
