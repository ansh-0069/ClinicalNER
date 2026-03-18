"""
run_phase1.py
─────────────
Phase 1 runner — executes the full setup + EDA pipeline.
Run this from the project root:

    python run_phase1.py [--real]

Flags:
    (no flag)  → uses synthetic dataset (works immediately, no Kaggle needed)
    --real     → loads data/raw/mtsamples.csv (download from Kaggle first)
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Allow imports from src/
sys.path.insert(0, ".")

from src.utils.data_loader import DataLoader
from src.utils.eda import ClinicalEDA


def main(use_real_data: bool = False) -> None:
    print("\n" + "="*60)
    print("  ClinicalNER — Phase 1: Data Ingestion & EDA")
    print("="*60 + "\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    loader = DataLoader(raw_dir="data/raw", db_path="data/clinicalner.db")

    if use_real_data:
        print("► Loading MTSamples CSV from data/raw/mtsamples.csv ...")
        df = loader.load_mtsamples()
    else:
        print("► Generating synthetic dataset (500 clinical notes) ...")
        print("  ℹ  Swap to MTSamples later: python run_phase1.py --real")
        df = loader.generate_synthetic_dataset(n_records=500)

    print(f"  ✓ {len(df)} notes loaded\n")

    # ── Step 2: Save to SQLite ────────────────────────────────────────────────
    print("► Saving to SQLite database ...")
    count = loader.save_to_db(df, table="clinical_notes")
    print(f"  ✓ {count} rows saved to data/clinicalner.db\n")

    # ── Step 3: Demonstrate SQL queries ──────────────────────────────────────
    print("► Running SQL analysis queries ...")

    specialty_counts = loader.sql_query(
        "SELECT medical_specialty, COUNT(*) as note_count "
        "FROM clinical_notes "
        "GROUP BY medical_specialty "
        "ORDER BY note_count DESC "
        "LIMIT 5"
    )
    print("\n  Top 5 specialties:")
    print(specialty_counts.to_string(index=False))

    avg_lengths = loader.sql_query(
        "SELECT medical_specialty, "
        "ROUND(AVG(LENGTH(transcription)), 0) as avg_chars "
        "FROM clinical_notes "
        "GROUP BY medical_specialty "
        "ORDER BY avg_chars DESC "
        "LIMIT 5"
    )
    print("\n  Top 5 specialties by avg note length:")
    print(avg_lengths.to_string(index=False))

    # ── Step 4: EDA ───────────────────────────────────────────────────────────
    print("\n► Running full EDA (5 charts) ...")
    eda = ClinicalEDA(df, output_dir="data/eda_outputs")

    stats = eda.basic_stats()
    print("\n  Corpus Statistics:")
    for k, v in stats.items():
        print(f"    {k:<28}: {v}")

    summary = eda.run_full_eda()

    print("\n  Charts saved:")
    for key, path in summary.items():
        if isinstance(path, str) and path.endswith(".png"):
            print(f"    ✓ {path}")

    print("\n  Sample notes from corpus:")
    eda.print_sample_notes(n=2)

    print("\n" + "="*60)
    print("  Phase 1 Complete!")
    print("  ─────────────────────────────────────────────────────")
    print("  DB     : data/clinicalner.db")
    print("  Charts : data/eda_outputs/")
    print("  Next   : Phase 2 — NER Pipeline (spaCy)")
    print("="*60 + "\n")


if __name__ == "__main__":
    use_real = "--real" in sys.argv
    main(use_real_data=use_real)
