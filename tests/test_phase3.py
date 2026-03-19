"""
test_phase3.py
--------------
Tests for DataCleaner and AuditLogger.
Drop this file into: tests/test_phase3.py
"""

import pytest
import json
from src.pipeline.data_cleaner import DataCleaner, CleaningResult
from src.pipeline.audit_logger import AuditLogger, AuditEntry, EventType


# ═══════════════════════════════════════════════════════════════
# DataCleaner — pre-NER tests
# ═══════════════════════════════════════════════════════════════

class TestDataCleanerPreNer:

    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_returns_cleaning_result(self):
        result = self.cleaner.clean_pre_ner("Patient John Smith.")
        assert isinstance(result, CleaningResult)

    def test_original_text_preserved(self):
        text = "Patient DOB: 04/12/1985"
        result = self.cleaner.clean_pre_ner(text)
        assert result.original_text == text

    def test_empty_string_returns_safely(self):
        result = self.cleaner.clean_pre_ner("")
        assert result.cleaned_text == ""
        assert result.change_count == 0

    def test_fixes_encoding_artefacts(self):
        text = "Patient\x92s history"
        result = self.cleaner.clean_pre_ner(text)
        assert "\x92" not in result.cleaned_text
        assert "'" in result.cleaned_text
        assert result.encoding_fixed is True

    def test_fixes_non_breaking_space(self):
        text = "Patient\xa0Name:\xa0John"
        result = self.cleaner.clean_pre_ner(text)
        assert "\xa0" not in result.cleaned_text

    def test_removes_dictation_fillers(self):
        text = "Patient um has uh a history of hmm diabetes"
        result = self.cleaner.clean_pre_ner(text)
        assert "um" not in result.cleaned_text
        assert "uh" not in result.cleaned_text
        assert "hmm" not in result.cleaned_text

    def test_filler_removal_logged_in_changes(self):
        text = "Patient um has uh diabetes"
        result = self.cleaner.clean_pre_ner(text)
        change_types = [c[0] for c in result.changes]
        assert "fillers" in change_types

    def test_normalises_crlf_line_endings(self):
        text = "Line one\r\nLine two\r\nLine three"
        result = self.cleaner.clean_pre_ner(text)
        assert "\r\n" not in result.cleaned_text
        assert "\n" in result.cleaned_text

    def test_collapses_multiple_spaces(self):
        text = "Patient    has     diabetes"
        result = self.cleaner.clean_pre_ner(text)
        assert "  " not in result.cleaned_text
        assert result.whitespace_fixed is True

    def test_normalises_section_headers(self):
        text = "ASSESSMENT:   diabetes type 2\nPLAN:   insulin"
        result = self.cleaner.clean_pre_ner(text)
        assert "ASSESSMENT:" in result.cleaned_text
        assert "PLAN:" in result.cleaned_text

    def test_section_header_change_logged(self):
        text = "ASSESSMENT:  diabetes"
        result = self.cleaner.clean_pre_ner(text)
        change_types = [c[0] for c in result.changes]
        assert "headers" in change_types

    def test_strips_leading_trailing_whitespace(self):
        text = "   Patient has diabetes.   "
        result = self.cleaner.clean_pre_ner(text)
        assert result.cleaned_text == result.cleaned_text.strip()

    def test_clean_text_no_changes(self):
        text = "Patient has diabetes and hypertension."
        result = self.cleaner.clean_pre_ner(text)
        assert result.change_count == 0

    def test_change_count_property(self):
        text = "Patient\x92s um history"
        result = self.cleaner.clean_pre_ner(text)
        assert result.change_count >= 2

    def test_summary_returns_string(self):
        result = self.cleaner.clean_pre_ner("Patient um has diabetes")
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Changes" in summary


# ═══════════════════════════════════════════════════════════════
# DataCleaner — post-NER tests
# ═══════════════════════════════════════════════════════════════

class TestDataCleanerPostNer:

    def setup_method(self):
        self.cleaner = DataCleaner()
        self.strict_cleaner = DataCleaner(strict_mode=True)

    def test_returns_cleaning_result(self):
        result = self.cleaner.clean_post_ner("[PERSON] visited [HOSPITAL]")
        assert isinstance(result, CleaningResult)

    def test_empty_string_returns_safely(self):
        result = self.cleaner.clean_post_ner("")
        assert result.cleaned_text == ""

    def test_normalises_mask_tokens_with_spaces(self):
        text = "Patient [ DATE ] visited [  HOSPITAL ]"
        result = self.cleaner.clean_post_ner(text)
        assert "[DATE]" in result.cleaned_text
        assert "[HOSPITAL]" in result.cleaned_text
        assert "[ DATE ]" not in result.cleaned_text

    def test_normalises_lowercase_mask_tokens(self):
        text = "[person] visited [hospital]"
        result = self.cleaner.clean_post_ner(text)
        assert "[PERSON]" in result.cleaned_text
        assert "[HOSPITAL]" in result.cleaned_text

    def test_detects_residual_phone(self):
        text = "[PERSON] called (415) 555-1234 yesterday"
        result = self.cleaner.clean_post_ner(text)
        assert len(result.residual_phi) > 0
        assert any("phone" in r for r in result.residual_phi)

    def test_detects_residual_date(self):
        text = "[PERSON] was admitted on 04/12/2024"
        result = self.cleaner.clean_post_ner(text)
        assert any("date" in r for r in result.residual_phi)

    def test_detects_residual_mrn(self):
        text = "MRN 302145 was not masked properly"
        result = self.cleaner.clean_post_ner(text)
        assert any("mrn" in r for r in result.residual_phi)

    def test_clean_masked_text_no_residual(self):
        text = "[PERSON] was admitted to [HOSPITAL] on [DATE]"
        result = self.cleaner.clean_post_ner(text)
        assert result.residual_phi == []
        assert result.is_valid is True

    def test_strict_mode_marks_invalid(self):
        text = "[PERSON] called (415) 555-1234"
        result = self.strict_cleaner.clean_post_ner(text)
        assert result.is_valid is False

    def test_non_strict_mode_stays_valid_with_residual(self):
        text = "[PERSON] called (415) 555-1234"
        result = self.cleaner.clean_post_ner(text)
        assert result.is_valid is True
        assert len(result.residual_phi) > 0

    def test_removes_double_spaces_from_masking(self):
        text = "[PERSON]  visited  [HOSPITAL]"
        result = self.cleaner.clean_post_ner(text)
        assert "  " not in result.cleaned_text


# ═══════════════════════════════════════════════════════════════
# DataCleaner — batch tests
# ═══════════════════════════════════════════════════════════════

class TestDataCleanerBatch:

    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_batch_returns_results_and_stats(self):
        records = [
            {"note_id": 1, "masked_text": "[PERSON] visited [HOSPITAL]"},
            {"note_id": 2, "masked_text": "[PERSON] called on [DATE]"},
        ]
        results, stats = self.cleaner.clean_batch(records)
        assert len(results) == 2
        assert isinstance(stats, dict)

    def test_batch_stats_keys_present(self):
        records = [{"note_id": 1, "masked_text": "[PERSON] visited [HOSPITAL]"}]
        _, stats = self.cleaner.clean_batch(records)
        for key in ["total", "with_changes", "invalid_count", "clean_rate"]:
            assert key in stats

    def test_batch_total_matches_input(self):
        records = [{"masked_text": f"Note {i}"} for i in range(5)]
        results, stats = self.cleaner.clean_batch(records)
        assert stats["total"] == 5

    def test_batch_pre_ner_pass(self):
        records = [{"masked_text": "Patient um has diabetes"}]
        results, stats = self.cleaner.clean_batch(
            records, text_col="masked_text", pass_type="pre"
        )
        assert stats["total"] == 1


# ═══════════════════════════════════════════════════════════════
# AuditEntry tests
# ═══════════════════════════════════════════════════════════════

class TestAuditEntry:

    def test_create_returns_frozen_dataclass(self):
        entry = AuditEntry.create(EventType.NER_COMPLETED, "5 entities found")
        assert isinstance(entry, AuditEntry)

    def test_frozen_raises_on_mutation(self):
        entry = AuditEntry.create(EventType.NER_COMPLETED, "test")
        with pytest.raises(Exception):
            entry.description = "modified"

    def test_metadata_serialised_as_json(self):
        entry = AuditEntry.create(
            EventType.NER_COMPLETED, "test",
            metadata={"count": 5}
        )
        parsed = json.loads(entry.metadata)
        assert parsed["count"] == 5

    def test_empty_metadata_defaults_to_empty_dict(self):
        entry = AuditEntry.create(EventType.NER_COMPLETED, "test")
        assert entry.metadata == "{}"

    def test_timestamp_is_iso_string(self):
        entry = AuditEntry.create(EventType.NER_COMPLETED, "test")
        assert "T" in entry.timestamp


# ═══════════════════════════════════════════════════════════════
# AuditLogger tests
# ═══════════════════════════════════════════════════════════════

class TestAuditLogger:

    def setup_method(self):
        self.audit = AuditLogger(db_path=":memory:")

    def test_log_returns_row_id(self):
        row_id = self.audit.log(EventType.NER_COMPLETED, "test event")
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_row_ids_increment(self):
        id1 = self.audit.log(EventType.NER_COMPLETED, "first")
        id2 = self.audit.log(EventType.NER_COMPLETED, "second")
        assert id2 > id1

    def test_total_events_counts_correctly(self):
        self.audit.log(EventType.NER_COMPLETED, "event 1")
        self.audit.log(EventType.DATA_CLEANED_PRE, "event 2")
        self.audit.log(EventType.API_REQUEST, "event 3")
        assert self.audit.total_events() == 3

    def test_empty_db_total_events_zero(self):
        fresh = AuditLogger(db_path=":memory:")
        assert fresh.total_events() == 0

    def test_get_log_returns_dataframe(self):
        self.audit.log(EventType.NER_COMPLETED, "test")
        df = self.audit.get_log()
        assert hasattr(df, "columns")

    def test_get_log_filter_by_note_id(self):
        self.audit.log(EventType.NER_COMPLETED, "note 1", note_id=1)
        self.audit.log(EventType.NER_COMPLETED, "note 2", note_id=2)
        self.audit.log(EventType.NER_COMPLETED, "note 1 again", note_id=1)
        df = self.audit.get_log(note_id=1)
        assert len(df) == 2

    def test_get_log_filter_by_event_type(self):
        self.audit.log(EventType.NER_COMPLETED, "ner event")
        self.audit.log(EventType.API_REQUEST, "api event")
        df = self.audit.get_log(event_type=EventType.NER_COMPLETED)
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "NER_COMPLETED"

    def test_get_summary_returns_dataframe(self):
        self.audit.log(EventType.NER_COMPLETED, "event")
        df = self.audit.get_summary()
        assert "event_type" in df.columns
        assert "count" in df.columns

    def test_log_ner_result_convenience(self):
        ner_result = {
            "note_id": 42,
            "entity_count": 5,
            "entity_types": {"DATE": 2, "PHONE": 1},
        }
        row_id = self.audit.log_ner_result(ner_result)
        assert row_id >= 1
        df = self.audit.get_log(note_id=42)
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "NER_COMPLETED"

    def test_log_cleaning_result_clean_note(self):
        cleaner = DataCleaner()
        clean_result = cleaner.clean_post_ner("[PERSON] visited [HOSPITAL]")
        row_id = self.audit.log_cleaning_result(clean_result, note_id=10)
        assert row_id >= 1

    def test_log_cleaning_result_residual_phi_event(self):
        cleaner = DataCleaner()
        clean_result = cleaner.clean_post_ner("[PERSON] called (415) 555-1234")
        self.audit.log_cleaning_result(clean_result, note_id=99)
        df = self.audit.get_log(event_type=EventType.RESIDUAL_PHI_FOUND)
        assert len(df) >= 1

    def test_get_flagged_notes_returns_dataframe(self):
        df = self.audit.get_flagged_notes()
        assert hasattr(df, "columns")

    def test_get_pipeline_history_returns_dataframe(self):
        df = self.audit.get_pipeline_history()
        assert hasattr(df, "columns")

    def test_log_batch_pipeline(self):
        ner_results = [
            {"note_id": 1, "entity_count": 3, "entity_types": {}},
            {"note_id": 2, "entity_count": 5, "entity_types": {}},
        ]
        clean_stats = {"total": 2, "invalid_count": 0}
        self.audit.log_batch_pipeline(
            ner_results, clean_stats, pipeline_id="run_001"
        )
        df = self.audit.get_log(event_type=EventType.PIPELINE_COMPLETE)
        assert len(df) == 1

    def test_event_type_extends_str(self):
        assert EventType.NER_COMPLETED == "NER_COMPLETED"
        assert EventType.RESIDUAL_PHI_FOUND == "RESIDUAL_PHI_FOUND"