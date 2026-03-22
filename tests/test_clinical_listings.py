"""Tests for ClinicalReportGenerator."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reports.clinical_listings import ClinicalReportGenerator


def _minimal_schema(conn):
    conn.execute(
        """
        CREATE TABLE clinical_notes (
            note_id INTEGER PRIMARY KEY,
            medical_specialty TEXT,
            description TEXT,
            transcription TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE processed_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            masked_text TEXT,
            entity_count INTEGER,
            entity_types_json TEXT,
            processed_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE audit_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            description TEXT,
            note_id INTEGER,
            user_id TEXT,
            timestamp TEXT,
            metadata TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO clinical_notes VALUES (1,'Cardiology','d','transcript')"
    )
    conn.execute(
        """INSERT INTO processed_notes
        (note_id, masked_text, entity_count, entity_types_json, processed_at)
        VALUES (1,'[MRN]',3,'{"MRN":1}','2024-06-01T12:00:00')"""
    )
    conn.execute(
        """INSERT INTO audit_log
        (event_type, description, note_id, user_id, timestamp, metadata)
        VALUES ('TEST','evt',1,'u1','2024-06-01','{}')"""
    )


def test_processing_summary_and_export(tmp_path):
    db = tmp_path / "cl.db"
    outdir = tmp_path / "reports"
    with sqlite3.connect(db) as conn:
        _minimal_schema(conn)

    gen = ClinicalReportGenerator(db_path=str(db), output_dir=str(outdir))
    df = gen.generate_processing_summary()
    assert not df.empty

    sas_path = gen.export_to_sas("processed_notes")
    assert sas_path.exists()


def test_audit_listing(tmp_path):
    db = tmp_path / "cl2.db"
    outdir = tmp_path / "r2"
    with sqlite3.connect(db) as conn:
        _minimal_schema(conn)

    gen = ClinicalReportGenerator(db_path=str(db), output_dir=str(outdir))
    df = gen.generate_audit_listing()
    assert len(df) >= 1


def test_phi_summary(tmp_path):
    db = tmp_path / "cl3.db"
    outdir = tmp_path / "r3"
    with sqlite3.connect(db) as conn:
        _minimal_schema(conn)

    gen = ClinicalReportGenerator(db_path=str(db), output_dir=str(outdir))
    df = gen.generate_phi_summary_report()
    assert isinstance(df, pd.DataFrame)


def test_dm_listing_empty_without_subject_dm(tmp_path):
    db = tmp_path / "cl4.db"
    with sqlite3.connect(db) as conn:
        _minimal_schema(conn)
    gen = ClinicalReportGenerator(db_path=str(db), output_dir=str(tmp_path / "r4"))
    df = gen.generate_dm_free_text_listing()
    assert df.empty
