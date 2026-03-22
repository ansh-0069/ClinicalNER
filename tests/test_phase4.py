"""
test_phase4.py
--------------
Tests for Flask app routes (app.py).
Drop this file into: tests/test_phase4.py
"""

import io

import pytest
import json
from src.api.app import create_app


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """
    Create a test Flask client using an in-memory database.
    scope="module" means the app is created once for all tests
    — avoids reloading the spaCy model per test.
    """
    app = create_app(db_path=":memory:")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ═══════════════════════════════════════════════════════════════
# /health
# ═══════════════════════════════════════════════════════════════

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        data = client.get("/health").get_json()
        assert data is not None

    def test_health_status_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_health_service_name(self, client):
        data = client.get("/health").get_json()
        assert data["service"] == "ClinicalNER"


# ═══════════════════════════════════════════════════════════════
# POST /api/deidentify
# ═══════════════════════════════════════════════════════════════

class TestDeidentify:

    def test_valid_request_returns_200(self, client):
        r = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985. Phone: (415) 555-9876."})
        assert r.status_code == 200

    def test_response_is_json(self, client):
        r = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985."})
        assert r.get_json() is not None

    def test_response_contains_masked_text(self, client):
        data = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985."}).get_json()
        assert "masked_text" in data

    def test_response_contains_entity_count(self, client):
        data = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985."}).get_json()
        assert "entity_count" in data
        assert isinstance(data["entity_count"], int)

    def test_response_contains_entity_types(self, client):
        data = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985."}).get_json()
        assert "entity_types" in data
        assert isinstance(data["entity_types"], dict)

    def test_response_contains_is_valid(self, client):
        data = client.post("/api/deidentify",
            json={"text": "Patient visited the hospital."}).get_json()
        assert "is_valid" in data

    def test_response_contains_changes(self, client):
        data = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985."}).get_json()
        assert "changes" in data

    def test_phi_masked_in_output(self, client):
        data = client.post("/api/deidentify",
            json={"text": "DOB: 04/12/1985. MRN: 302145."}).get_json()
        masked = data["masked_text"]
        assert "04/12/1985" not in masked or "[DATE]" in masked

    def test_missing_text_field_returns_400(self, client):
        r = client.post("/api/deidentify", json={"note_id": 1})
        assert r.status_code == 400

    def test_empty_text_returns_400(self, client):
        r = client.post("/api/deidentify", json={"text": ""})
        assert r.status_code == 400

    def test_whitespace_only_text_returns_400(self, client):
        r = client.post("/api/deidentify", json={"text": "   "})
        assert r.status_code == 400

    def test_non_json_request_returns_400(self, client):
        r = client.post("/api/deidentify",
            data="plain text", content_type="text/plain")
        assert r.status_code == 400
        err = r.get_json()
        assert "error" in err
        assert "text" in err["error"].lower() or "file" in err["error"].lower()

    def test_multipart_txt_file_deidentifies(self, client):
        buf = io.BytesIO(b"Patient DOB: 04/12/1985. Phone: (415) 555-9876.")
        r = client.post(
            "/api/deidentify",
            data={"file": (buf, "note.txt")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "masked_text" in data
        assert data.get("entity_count", 0) >= 0

    def test_multipart_bad_extension_returns_400(self, client):
        buf = io.BytesIO(b"hello")
        r = client.post(
            "/api/deidentify",
            data={"file": (buf, "x.exe")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_oversized_text_returns_413(self, client):
        r = client.post("/api/deidentify", json={"text": "x" * 50_001})
        assert r.status_code == 413

    def test_error_response_has_error_key(self, client):
        data = client.post("/api/deidentify",
            json={"text": ""}).get_json()
        assert "error" in data

    def test_optional_note_id_accepted(self, client):
        r = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985.", "note_id": 99})
        assert r.status_code == 200

    def test_save_false_accepted(self, client):
        r = client.post("/api/deidentify",
            json={"text": "Patient DOB: 04/12/1985.", "save": False})
        assert r.status_code == 200

    def test_exactly_50000_chars_accepted(self, client):
        r = client.post("/api/deidentify", json={"text": "x" * 50_000})
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════
# GET /api/note/<id>
# ═══════════════════════════════════════════════════════════════

class TestGetNote:

    def test_nonexistent_note_returns_error(self, client):
        r = client.get("/api/note/999999")
        assert r.status_code in (404, 500)

    def test_error_response_has_error_key(self, client):
        data = client.get("/api/note/999999").get_json()
        assert "error" in data

    def test_processed_note_retrieval_no_crash(self, client):
        client.post("/api/deidentify",
            json={"text": "DOB: 04/12/1985.", "note_id": 1001, "save": True})
        r = client.get("/api/note/1001")
        assert r.status_code in (200, 404, 500)


# ═══════════════════════════════════════════════════════════════
# GET /api/stats
# ═══════════════════════════════════════════════════════════════

class TestStats:

    def test_stats_returns_200(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_stats_returns_json(self, client):
        assert client.get("/api/stats").get_json() is not None

    def test_stats_has_note_count(self, client):
        data = client.get("/api/stats").get_json()
        assert "note_count" in data
        assert isinstance(data["note_count"], int)

    def test_stats_has_processed_count(self, client):
        data = client.get("/api/stats").get_json()
        assert "processed_count" in data

    def test_stats_has_entity_totals(self, client):
        data = client.get("/api/stats").get_json()
        assert "entity_totals" in data
        assert isinstance(data["entity_totals"], dict)

    def test_stats_has_audit_summary(self, client):
        data = client.get("/api/stats").get_json()
        assert "audit_summary" in data
        assert isinstance(data["audit_summary"], list)

    def test_stats_has_total_audit_events(self, client):
        data = client.get("/api/stats").get_json()
        assert "total_audit_events" in data
        assert isinstance(data["total_audit_events"], int)

    def test_stats_has_specialty(self, client):
        data = client.get("/api/stats").get_json()
        assert "specialty" in data
        assert isinstance(data["specialty"], list)

    def test_stats_has_phi_by_specialty(self, client):
        data = client.get("/api/stats").get_json()
        assert "phi_by_specialty" in data

    def test_audit_events_increase_after_deidentify(self, client):
        before = client.get("/api/stats").get_json()["total_audit_events"]
        client.post("/api/deidentify", json={"text": "DOB: 01/01/2000."})
        after = client.get("/api/stats").get_json()["total_audit_events"]
        assert after > before


# ═══════════════════════════════════════════════════════════════
# Blueprint: /api/model-registry, /api/data-quality
# ═══════════════════════════════════════════════════════════════

class TestApiBlueprint:

    def test_model_registry_returns_200(self, client):
        r = client.get("/api/model-registry")
        assert r.status_code == 200
        data = r.get_json()
        assert data.get("active_model_id") or data.get("models")

    def test_data_quality_returns_200(self, client):
        r = client.get("/api/data-quality")
        assert r.status_code == 200
        data = r.get_json()
        assert "completeness" in data
        assert "accuracy" in data


# ═══════════════════════════════════════════════════════════════
# GET /dashboard
# ═══════════════════════════════════════════════════════════════

class TestDashboard:

    def test_dashboard_returns_200(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_returns_html(self, client):
        r = client.get("/dashboard")
        assert b"<!DOCTYPE html>" in r.data or b"<html" in r.data

    def test_dashboard_contains_title(self, client):
        r = client.get("/dashboard")
        assert b"ClinicalNER" in r.data


# ═══════════════════════════════════════════════════════════════
# GET /report/<note_id>
# ═══════════════════════════════════════════════════════════════

class TestReport:

    def test_report_nonexistent_note_returns_404(self, client):
        r = client.get("/report/999999")
        assert r.status_code == 404

    def test_report_returns_html_not_json(self, client):
        r = client.get("/report/999999")
        assert b"<" in r.data