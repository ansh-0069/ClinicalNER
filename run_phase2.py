"""
run_phase2.py
─────────────
Phase 2 runner — NER pipeline on the full corpus.

Run from the project root:
    python run_phase2.py

Expects Phase 1 to have been run first (data/clinicalner.db must exist).
If not, generates a fresh synthetic dataset automatically.
"""

import sys
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.utils.data_loader import DataLoader
from src.pipeline.ner_pipeline import NERPipeline


def main() -> None:
    print("\n" + "="*60)
    print("  ClinicalNER — Phase 2: NER Pipeline")
    print("="*60 + "\n")

    # ── Step 1: Load notes from DB ────────────────────────────────────────────
    loader = DataLoader(db_path="data/clinicalner.db")

    try:
        df = loader.load_from_db(table="clinical_notes")
        print(f"► Loaded {len(df)} notes from database\n")
    except Exception:
        print("► No DB found — running Phase 1 first to generate data...")
        df = loader.generate_synthetic_dataset(500)
        loader.save_to_db(df)
        print(f"  ✓ Generated and saved {len(df)} synthetic notes\n")

    # ── Step 2: Demo on 3 individual notes ───────────────────────────────────
    print("► Demo: processing 3 individual notes")
    print("─" * 60)

    pipeline = NERPipeline(db_path="data/clinicalner.db", use_spacy=True)

    sample_notes = [
        "Patient James Smith, a 58-year-old male presented to "
        "St. Mary's Hospital on 04/12/2022. Phone: (415) 555-9876. MRN304512.",

        "PATIENT: Linda Johnson | DOB: 07/22/1978 | MRN: MRN890231\n"
        "Date of service: January 15, 2023 | Facility: City General Medical Center\n"
        "Contact: 212-555-0034\n"
        "Assessment: 45-year-old female with progressive shortness of breath.",

        "Operative report — Memorial Medical Center, 2022-09-10.\n"
        "Patient: Robert Garcia, Age 72, male. Surgeon: Dr. Thompson.\n"
        "Procedure: Laparoscopic cholecystectomy. MRN901872.\n"
        "Follow-up at Memorial Medical Center on 09/24/2022.",
    ]

    for i, note in enumerate(sample_notes, 1):
        result = pipeline.process_note(note, note_id=None, save_to_db=False)

        print(f"\nNote {i}:")
        print(f"  Original : {note[:120]}{'...' if len(note)>120 else ''}")
        print(f"  Masked   : {result['masked_text'][:120]}{'...' if len(result['masked_text'])>120 else ''}")
        print(f"  Entities ({result['entity_count']} found):")
        for ent in result["entities"]:
            print(f"    [{ent['label']:10}] '{ent['text']}' (src={ent['source']})")

    # ── Step 3: Batch process full corpus ─────────────────────────────────────
    print("\n" + "─"*60)
    print(f"► Batch processing {len(df)} notes ...")

    notes_list = df.to_dict("records")
    results = pipeline.process_batch(
        notes_list,
        text_col="transcription",
        id_col="note_id",
    )

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n► Evaluation summary:")
    stats = pipeline.evaluate(results)
    print(f"\n  {'Metric':<28} {'Value'}")
    print(f"  {'─'*44}")
    print(f"  {'Total notes':<28} {stats['total_notes']}")
    print(f"  {'Notes with PHI':<28} {stats['notes_with_phi']} ({stats['phi_rate']})")
    print(f"  {'Total entities found':<28} {stats['total_entities']}")
    print(f"  {'Avg entities per PHI note':<28} {stats['avg_per_phi_note']}")
    print(f"\n  {'Entity type breakdown':}")
    print(f"  {'─'*44}")
    for label, count in stats["entity_breakdown"].items():
        bar = "█" * min(int(count / max(stats["entity_breakdown"].values()) * 30), 30)
        print(f"  {label:<12} {count:>5}  {bar}")

    # ── Step 5: SQL query on processed results ────────────────────────────────
    print("\n► SQL analysis on processed_notes table:")
    try:
        sql_stats = loader.sql_query(
            "SELECT COUNT(*) as total_processed, "
            "AVG(entity_count) as avg_entities "
            "FROM processed_notes"
        )
        print(sql_stats.to_string(index=False))
    except Exception as e:
        logger.warning("SQL query on processed_notes failed: %s", e)

    print("\n" + "="*60)
    print("  Phase 2 Complete!")
    print("  ─────────────────────────────────────────────────────")
    print("  DB tables  : clinical_notes, processed_notes")
    print("  Run tests  : pytest tests/test_ner_pipeline.py -v")
    print("  Next       : Phase 3 — DataCleaner + AuditLogger")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()