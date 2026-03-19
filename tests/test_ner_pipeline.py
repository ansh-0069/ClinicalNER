"""
test_ner_pipeline.py
────────────────────
Unit tests for NERPipeline.

Run with:  pytest tests/test_ner_pipeline.py -v

Key decision: tests use SYNTHETIC notes — never real patient data.
Each test targets one specific behaviour to make failures easy to diagnose.
"""

import sys
sys.path.insert(0, ".")

import pytest
from src.pipeline.ner_pipeline import NERPipeline, PHIEntity

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """NERPipeline in regex-only mode (no spaCy needed for CI)."""
    return NERPipeline(db_path=":memory:", use_spacy=False)


# ── PHIEntity dataclass ───────────────────────────────────────────────────────

def test_phi_entity_repr():
    e = PHIEntity(label="PERSON", text="James Smith", start=8, end=19)
    assert "PERSON" in repr(e)
    assert "James Smith" in repr(e)


# ── Date detection ────────────────────────────────────────────────────────────

def test_detects_date_slash_format(pipeline):
    note = "Patient was admitted on 04/12/2022 for observation."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "DATE" in labels

def test_detects_date_iso_format(pipeline):
    note = "Discharge date: 2023-07-15."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "DATE" in labels

def test_detects_date_verbal_format(pipeline):
    note = "Seen on January 5, 2022 at the clinic."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "DATE" in labels


# ── Phone detection ───────────────────────────────────────────────────────────

def test_detects_phone_with_parentheses(pipeline):
    note = "Call the patient at (415) 555-1234 for follow-up."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "PHONE" in labels

def test_detects_phone_dot_separator(pipeline):
    note = "Contact: 415.555.9876"
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "PHONE" in labels


# ── Age detection ─────────────────────────────────────────────────────────────

def test_detects_age_year_old(pipeline):
    note = "This is a 58-year-old male with hypertension."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "AGE" in labels

def test_detects_age_colon_format(pipeline):
    note = "Age: 72. No known allergies."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "AGE" in labels


# ── MRN detection ─────────────────────────────────────────────────────────────

def test_detects_mrn(pipeline):
    note = "Patient MRN902341 was referred by Dr. Chen."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "MRN" in labels


# ── Hospital detection ────────────────────────────────────────────────────────

def test_detects_hospital_name(pipeline):
    note = "Patient was treated at St. Mary's Hospital last week."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "HOSPITAL" in labels

def test_detects_medical_center(pipeline):
    note = "Referred to City General Medical Center for surgery."
    result = pipeline.process_note(note, save_to_db=False)
    labels = [e["label"] for e in result["entities"]]
    assert "HOSPITAL" in labels


# ── Masking ───────────────────────────────────────────────────────────────────

def test_masking_replaces_phi(pipeline):
    note = "Patient DOB: 03/15/1980. Phone: (212) 555-7890."
    result = pipeline.process_note(note, save_to_db=False)
    assert "03/15/1980" not in result["masked_text"]
    assert "(212) 555-7890" not in result["masked_text"]

def test_masking_uses_bracket_style_by_default(pipeline):
    note = "Admitted on 01/01/2023."
    result = pipeline.process_note(note, save_to_db=False)
    assert "[DATE]" in result["masked_text"]

def test_masking_tag_style():
    pipe = NERPipeline(db_path=":memory:", use_spacy=False, mask_style="tag")
    note = "Admitted on 01/01/2023."
    result = pipe.process_note(note, save_to_db=False)
    assert "<DATE/>" in result["masked_text"]

def test_original_text_preserved(pipeline):
    note = "DOB: 05/20/1975. MRN12345678."
    result = pipeline.process_note(note, save_to_db=False)
    assert result["original_text"] == note


# ── Overlap resolution ────────────────────────────────────────────────────────

def test_overlapping_spans_resolved(pipeline):
    """Two overlapping spans should produce exactly one entity."""
    note = "Seen at St. Mary's Hospital on 06/10/2021."
    result = pipeline.process_note(note, save_to_db=False)
    starts = [e["start"] for e in result["entities"]]
    assert len(starts) == len(set(starts)), "Overlapping entity starts detected"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_string(pipeline):
    result = pipeline.process_note("", save_to_db=False)
    assert result["entity_count"] == 0
    assert result["masked_text"] == ""

def test_no_phi_note(pipeline):
    note = "The patient's cardiovascular exam was unremarkable. No acute distress."
    result = pipeline.process_note(note, save_to_db=False)
    # Should not crash; PHI count may or may not be 0 depending on patterns
    assert isinstance(result["entity_count"], int)

def test_result_structure(pipeline):
    note = "Call (312) 555-0000 after 04/01/2024."
    result = pipeline.process_note(note, save_to_db=False)
    assert "masked_text"   in result
    assert "entities"      in result
    assert "entity_count"  in result
    assert "entity_types"  in result
    assert "original_text" in result


# ── Batch processing ──────────────────────────────────────────────────────────

def test_batch_processing(pipeline):
    notes = [
        {"note_id": 1, "transcription": "Patient DOB: 01/01/1990, MRN100001."},
        {"note_id": 2, "transcription": "Age: 45. Phone: (713) 555-2345."},
        {"note_id": 3, "transcription": "No PHI in this note at all."},
    ]
    results = pipeline.process_batch(notes)
    assert len(results) == 3
    assert results[0]["entity_count"] > 0
    assert results[1]["entity_count"] > 0


# ── Evaluation stats ─────────────────────────────────────────────────────────

def test_evaluate_returns_stats(pipeline):
    notes = [
        {"note_id": i, "transcription": f"Patient age 5{i}-year-old. MRN10000{i}."}
        for i in range(5)
    ]
    results = pipeline.process_batch(notes)
    stats = pipeline.evaluate(results)
    assert "total_notes"      in stats
    assert "notes_with_phi"   in stats
    assert "entity_breakdown" in stats
    assert stats["total_notes"] == 5
