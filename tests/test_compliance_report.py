"""
test_compliance_report.py
─────────────────────────
Tests for HIPAAComplianceReport and ComplianceReport.
Drop into: tests/test_compliance_report.py
"""

import pytest
from unittest.mock import MagicMock
import pandas as pd
import json

from src.pipeline.compliance_report import (
    HIPAAComplianceReport,
    ComplianceReport,
    HIPAA_IDENTIFIERS,
    PHI_LABEL_TO_HIPAA,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_audit():
    audit = MagicMock()
    audit.total_events.return_value = 382
    audit.get_log.return_value = pd.DataFrame({
        "timestamp":   ["2026-03-18T23:55:04+00:00", "2026-03-20T04:06:49+00:00"],
        "event_type":  ["NER_COMPLETED", "DATA_CLEANED_POST"],
        "note_id":     [1, 1],
        "description": ["NER complete", "Cleaning complete"],
    })
    audit.get_summary.return_value = pd.DataFrame({
        "event_type": ["NER_COMPLETED", "DATA_CLEANED_POST"],
        "count":      [55, 255],
        "first_seen": ["2026-03-18T23:55:14+00:00", "2026-03-18T23:55:14+00:00"],
        "last_seen":  ["2026-03-20T04:06:49+00:00", "2026-03-20T04:06:49+00:00"],
    })
    audit.get_flagged_notes.return_value = pd.DataFrame()
    return audit

@pytest.fixture
def mock_loader_empty():
    loader = MagicMock()
    loader.sql_query.side_effect = Exception("no such table")
    return loader

@pytest.fixture
def mock_loader_with_data():
    loader = MagicMock()
    def sql_query(sql):
        if "COUNT" in sql:
            return pd.DataFrame({"n": [50]})
        if "entity_types_json" in sql:
            return pd.DataFrame({
                "entity_types_json": [
                    json.dumps({"DATE": 3, "PHONE": 2, "MRN": 1}),
                    json.dumps({"PERSON": 2, "HOSPITAL": 1}),
                ]
            })
        return pd.DataFrame()
    loader.sql_query.side_effect = sql_query
    return loader

@pytest.fixture
def reporter_empty(mock_loader_empty, mock_audit):
    return HIPAAComplianceReport(
        loader=mock_loader_empty,
        audit=mock_audit,
    )

@pytest.fixture
def reporter_with_data(mock_loader_with_data, mock_audit):
    return HIPAAComplianceReport(
        loader=mock_loader_with_data,
        audit=mock_audit,
    )


# ── HIPAA constants tests ─────────────────────────────────────────────────────

class TestHIPAAConstants:

    def test_18_identifiers_defined(self):
        assert len(HIPAA_IDENTIFIERS) == 18

    def test_names_in_identifiers(self):
        assert "Names" in HIPAA_IDENTIFIERS

    def test_dates_in_identifiers(self):
        assert "Dates (except year)" in HIPAA_IDENTIFIERS

    def test_phi_label_map_covers_pipeline_labels(self):
        expected = {"PERSON", "DATE", "DOB", "PHONE", "MRN", "HOSPITAL", "LOCATION", "AGE"}
        assert expected.issubset(set(PHI_LABEL_TO_HIPAA.keys()))


# ── ComplianceReport dataclass tests ─────────────────────────────────────────

class TestComplianceReport:

    def make_report(self, **kwargs):
        defaults = dict(
            generated_at="2026-03-20T00:00:00+00:00",
            report_period_start="2026-03-18T00:00:00+00:00",
            report_period_end="2026-03-20T00:00:00+00:00",
            total_notes_processed=50,
            total_phi_detected=200,
            total_phi_masked=200,
            residual_phi_count=0,
            audit_event_count=382,
            phi_coverage={"DATE": 100, "PHONE": 50},
            hipaa_coverage=[],
            audit_summary=[],
            residual_findings=[],
            compliance_status="COMPLIANT",
            attestation={},
            recommendations=[],
        )
        defaults.update(kwargs)
        return ComplianceReport(**defaults)

    def test_to_dict_returns_dict(self):
        report = self.make_report()
        assert isinstance(report.to_dict(), dict)

    def test_to_dict_has_all_keys(self):
        report = self.make_report()
        d = report.to_dict()
        for key in ["generated_at", "compliance_status", "total_phi_detected",
                    "audit_event_count", "hipaa_coverage", "attestation",
                    "recommendations", "residual_phi_count"]:
            assert key in d

    def test_compliant_status(self):
        report = self.make_report(residual_phi_count=0,
                                   compliance_status="COMPLIANT")
        assert report.compliance_status == "COMPLIANT"

    def test_review_required_status(self):
        report = self.make_report(residual_phi_count=3,
                                   compliance_status="REVIEW_REQUIRED")
        assert report.compliance_status == "REVIEW_REQUIRED"


# ── HIPAAComplianceReport tests ───────────────────────────────────────────────

class TestHIPAAComplianceReport:

    def test_generate_returns_compliance_report(self, reporter_empty):
        report = reporter_empty.generate()
        assert isinstance(report, ComplianceReport)

    def test_generate_has_generated_at(self, reporter_empty):
        report = reporter_empty.generate()
        assert "T" in report.generated_at

    def test_audit_event_count_populated(self, reporter_empty):
        report = reporter_empty.generate()
        assert report.audit_event_count == 382

    def test_hipaa_coverage_has_18_entries(self, reporter_empty):
        report = reporter_empty.generate()
        assert len(report.hipaa_coverage) == 18

    def test_hipaa_coverage_has_status_field(self, reporter_empty):
        report = reporter_empty.generate()
        for entry in report.hipaa_coverage:
            assert "identifier" in entry
            assert "status" in entry

    def test_hipaa_status_values_valid(self, reporter_empty):
        report = reporter_empty.generate()
        valid_statuses = {"DETECTED_AND_MASKED", "NOT_APPLICABLE", "NOT_DETECTED"}
        for entry in report.hipaa_coverage:
            assert entry["status"] in valid_statuses

    def test_compliance_compliant_when_no_residual(self, reporter_empty):
        report = reporter_empty.generate()
        assert report.compliance_status == "COMPLIANT"

    def test_audit_summary_populated(self, reporter_empty):
        report = reporter_empty.generate()
        assert isinstance(report.audit_summary, list)
        assert len(report.audit_summary) > 0

    def test_attestation_has_hipaa_key(self, reporter_empty):
        report = reporter_empty.generate()
        assert "hipaa_safe_harbor" in report.attestation

    def test_attestation_has_gcp_key(self, reporter_empty):
        report = reporter_empty.generate()
        assert "ich_e6_gcp" in report.attestation

    def test_attestation_hipaa_standard(self, reporter_empty):
        report = reporter_empty.generate()
        assert "164.514" in report.attestation["hipaa_safe_harbor"]["standard"]

    def test_attestation_gcp_standard(self, reporter_empty):
        report = reporter_empty.generate()
        assert "ICH E6" in report.attestation["ich_e6_gcp"]["standard"]

    def test_recommendations_not_empty(self, reporter_empty):
        report = reporter_empty.generate()
        assert len(report.recommendations) > 0

    def test_recommendations_are_strings(self, reporter_empty):
        report = reporter_empty.generate()
        assert all(isinstance(r, str) for r in report.recommendations)

    def test_with_data_phi_coverage_populated(self, reporter_with_data):
        report = reporter_with_data.generate()
        assert len(report.phi_coverage) > 0

    def test_with_data_detected_phi_in_hipaa(self, reporter_with_data):
        report = reporter_with_data.generate()
        detected = [e for e in report.hipaa_coverage
                    if e["status"] == "DETECTED_AND_MASKED"]
        assert len(detected) > 0

    def test_with_data_notes_processed(self, reporter_with_data):
        report = reporter_with_data.generate()
        assert report.total_notes_processed == 50

    def test_with_data_total_phi_detected(self, reporter_with_data):
        report = reporter_with_data.generate()
        assert report.total_phi_detected > 0

    def test_residual_findings_is_list(self, reporter_empty):
        report = reporter_empty.generate()
        assert isinstance(report.residual_findings, list)

    def test_report_period_start_populated(self, reporter_empty):
        report = reporter_empty.generate()
        assert report.report_period_start != "N/A"

    def test_to_dict_serialisable(self, reporter_empty):
        report = reporter_empty.generate()
        import json
        dumped = json.dumps(report.to_dict())
        assert len(dumped) > 0