"""
run_structured_demo.py
──────────────────────
Loads SDTM-style DM demo (``data/dm_subject_demo.csv``) into SQLite table
``subject_dm`` and emits a combined DM + unstructured-note listing CSV.

Prerequisites: ``python run_phase1.py`` (clinical_notes present).
Optional: run NER/backfill so ``processed_notes`` is populated for the listing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "clinicalner.db"
CSV_PATH = ROOT / "data" / "dm_subject_demo.csv"


def main() -> None:
    if not DB_PATH.exists():
        print("Database not found. Run: python run_phase1.py")
        sys.exit(1)
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("subject_dm", conn, if_exists="replace", index=False)

    print(f"Loaded {len(df)} rows into subject_dm ({DB_PATH})")

    from src.reports.clinical_listings import ClinicalReportGenerator

    gen = ClinicalReportGenerator(db_path=str(DB_PATH), output_dir="data/reports")
    listing = gen.generate_dm_free_text_listing()
    print(f"DM + note listing: {len(listing)} rows → data/reports/dm_note_listing_*.csv")


if __name__ == "__main__":
    main()
