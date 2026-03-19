"""
test_coverage_boost.py
──────────────────────
Targeted tests to push coverage above 80% threshold.
Covers DataLoader and ClinicalEDA which were previously untested.
"""

import sys, os
sys.path.insert(0, ".")

import pytest
import pandas as pd
from pathlib import Path
from src.utils.data_loader import DataLoader
from src.utils.eda import ClinicalEDA


# ── DataLoader ────────────────────────────────────────────────────────────────

@pytest.fixture
def loader(tmp_path):
    return DataLoader(
        raw_dir=str(tmp_path / "raw"),
        db_path=str(tmp_path / "test.db"),
    )

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "note_id":           [1, 2, 3],
        "medical_specialty": ["Cardiology", "Neurology", "Surgery"],
        "description":       ["desc1", "desc2", "desc3"],
        "transcription":     [
            "Patient DOB: 04/12/1985. Phone: (415) 555-9876.",
            "Admitted on 01/01/2023. MRN302145.",
            "Follow-up at Memorial Medical Center on 06/10/2022.",
        ],
        "has_phi": [True, True, True],
    })

def test_loader_init(loader):
    assert loader.db_path.parent.exists()

def test_generate_synthetic_dataset(loader):
    df = loader.generate_synthetic_dataset(n_records=20)
    assert len(df) == 20
    assert "transcription" in df.columns
    assert "medical_specialty" in df.columns

def test_save_and_load_from_db(loader, sample_df):
    loader.save_to_db(sample_df)
    loaded = loader.load_from_db()
    assert len(loaded) == 3

def test_sql_query(loader, sample_df):
    loader.save_to_db(sample_df)
    result = loader.sql_query(
        "SELECT medical_specialty, COUNT(*) as n "
        "FROM clinical_notes GROUP BY medical_specialty"
    )
    assert len(result) == 3

def test_clean_column_names(loader):
    df = pd.DataFrame({"Medical Specialty": ["X"], "Transcription": ["Y"], "Description": ["Z"]})
    cleaned = loader._clean_column_names(df)
    assert "medical_specialty" in cleaned.columns

def test_basic_clean_removes_empty(loader):
    df = pd.DataFrame({
        "transcription": ["valid text", "", "   ", None],
        "medical_specialty": ["A", "B", "C", "D"],
        "description": ["d1", "d2", "d3", "d4"],
    })
    cleaned = loader._basic_clean(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["transcription"] == "valid text"

def test_save_to_db_returns_count(loader, sample_df):
    count = loader.save_to_db(sample_df)
    assert count == 3

def test_load_from_db_with_limit(loader, sample_df):
    loader.save_to_db(sample_df)
    loaded = loader.load_from_db(limit=2)
    assert len(loaded) == 2

def test_load_mtsamples_raises_if_missing(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_mtsamples()


# ── ClinicalEDA ───────────────────────────────────────────────────────────────

@pytest.fixture
def eda(sample_df, tmp_path):
    return ClinicalEDA(sample_df, output_dir=str(tmp_path / "eda_out"))

def test_eda_basic_stats(eda):
    stats = eda.basic_stats()
    assert stats["total_notes"] == 3
    assert "mean_note_length" in stats
    assert "unique_specialties" in stats

def test_eda_specialty_distribution(eda):
    path = eda.plot_specialty_distribution()
    assert Path(path).exists()
    assert path.endswith(".png")

def test_eda_note_length_distribution(eda):
    path = eda.plot_note_length_distribution()
    assert Path(path).exists()

def test_eda_phi_pattern_frequency(eda):
    path = eda.plot_phi_pattern_frequency()
    assert Path(path).exists()

def test_eda_missing_data(eda):
    path = eda.plot_missing_data()
    assert Path(path).exists()

def test_eda_top_clinical_words(eda):
    path = eda.plot_top_clinical_words()
    assert Path(path).exists()

def test_eda_run_full_eda(eda):
    summary = eda.run_full_eda()
    assert "basic_stats"      in summary
    assert "specialty_dist"   in summary
    assert "note_length_dist" in summary
    assert "phi_pattern_freq" in summary

def test_eda_precompute_adds_columns(sample_df, tmp_path):
    eda = ClinicalEDA(sample_df, output_dir=str(tmp_path))
    assert "note_length" in eda.df.columns
    assert "word_count"  in eda.df.columns
    assert "sent_count"  in eda.df.columns
