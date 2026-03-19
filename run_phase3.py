"""
run_phase3.py
─────────────
Phase 3 runner — DataCleaner + AuditLogger on the full corpus.

Run from the project root:
    python run_phase3.py
"""

import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.utils.data_loader import DataLoader
from src.pipeline.ner_pipeline import NERPipeline
from src.pipeline.data_cleaner import DataCleaner
from src.pipeline.audit_logger import AuditLogger, EventType


def main() -> None:
    print("\n" + "="*60)
    print("  ClinicalNER — Phase 3: DataCleaner + AuditLogger")
    print("="*60 + "\n")

    loader  = DataLoader(db_path="data/clinicalner.db")
    cleaner = DataCleaner(strict_mode=False)
    audit   = AuditLogger(db_path="data/clinicalner.db")
    pipeline = NERPipeline(db_path="data/clinicalner.db", use_spacy=True)

    # ── Log pipeline start ────────────────────────────────────────────────────
    audit.log(EventType.PIPELINE_START, "Phase 3 pipeline started")

    # ── Step 1: Load raw notes, apply pre-NER cleaning ────────────────────────
    print("► Step 1: Pre-NER cleaning on raw corpus")
    df_raw = loader.load_from_db("clinical_notes")
    notes  = df_raw.to_dict("records")

    pre_results, pre_stats = cleaner.clean_batch(
        notes, text_col="transcription", id_col="note_id", pass_type="pre"
    )

    print(f"  ✓ {pre_stats['total']} notes cleaned")
    print(f"  ✓ {pre_stats['with_changes']} had changes applied")
    print(f"  ✓ {pre_stats['encoding_fixed']} encoding fixes")
    print(f"  ✓ {pre_stats['ws_fixed']} whitespace fixes")

    # Log a few pre-cleaning events
    for i, (note, result) in enumerate(zip(notes[:10], pre_results[:10])):
        if result.change_count > 0:
            audit.log(
                EventType.DATA_CLEANED_PRE,
                f"Pre-clean: {result.change_count} change(s)",
                note_id=note.get("note_id"),
                metadata={"changes": result.changes},
            )

    # ── Step 2: Run NER on cleaned text ───────────────────────────────────────
    print("\n► Step 2: NER on cleaned text (sample of 50 notes)")

    cleaned_notes = []
    for note, pre_result in zip(notes[:50], pre_results[:50]):
        cleaned_notes.append({
            **note,
            "transcription": pre_result.cleaned_text
        })

    ner_results = pipeline.process_batch(
        cleaned_notes, text_col="transcription", id_col="note_id"
    )
    print(f"  ✓ {len(ner_results)} notes processed by NER")

    # Log NER results
    for r in ner_results[:10]:
        audit.log_ner_result(r)

    # ── Step 3: Post-NER cleaning + validation ────────────────────────────────
    print("\n► Step 3: Post-NER cleaning + residual PHI scan")

    masked_records = [
        {"note_id": r["note_id"], "masked_text": r["masked_text"]}
        for r in ner_results
    ]
    post_results, post_stats = cleaner.clean_batch(
        masked_records, text_col="masked_text", id_col="note_id", pass_type="post"
    )

    print(f"  ✓ {post_stats['total']} masked notes validated")
    print(f"  ✓ {post_stats['clean_rate']} clean rate")
    print(f"  ⚠  {post_stats['residual_phi']} notes with possible residual PHI")

    # Log post-cleaning and residual PHI events
    for note, result in zip(masked_records, post_results):
        audit.log_cleaning_result(result, note_id=note.get("note_id"))

    # ── Step 4: Show before/after demo ────────────────────────────────────────
    print("\n► Before / After demo (3 notes):")
    print("─" * 60)
    for i in range(min(3, len(ner_results))):
        orig    = ner_results[i]["original_text"][:120]
        masked  = post_results[i].cleaned_text[:120]
        n_ents  = ner_results[i]["entity_count"]
        valid   = post_results[i].is_valid

        print(f"\nNote {i+1}:")
        print(f"  IN  : {orig}...")
        print(f"  OUT : {masked}...")
        print(f"  PHI entities masked : {n_ents} | Post-clean valid : {valid}")

    # ── Step 5: Log pipeline complete ─────────────────────────────────────────
    audit.log_batch_pipeline(
        ner_results=ner_results,
        clean_stats={**pre_stats, **{f"post_{k}": v for k, v in post_stats.items()}},
        pipeline_id="phase3-run-001",
    )

    # ── Step 6: Audit log summary ─────────────────────────────────────────────
    print("\n► Audit log summary:")
    print("─" * 60)
    summary = audit.get_summary()
    print(summary.to_string(index=False))
    print(f"\n  Total audit events logged : {audit.total_events()}")

    # ── Step 7: SQL cross-table query (JD requirement) ────────────────────────
    print("\n► SQL cross-table analysis:")
    cross = loader.sql_query("""
        SELECT
            cn.medical_specialty,
            COUNT(pn.id)              AS notes_processed,
            AVG(pn.entity_count)      AS avg_phi_entities,
            SUM(pn.entity_count)      AS total_phi_found
        FROM clinical_notes cn
        LEFT JOIN processed_notes pn ON cn.note_id = pn.note_id
        WHERE pn.id IS NOT NULL
        GROUP BY cn.medical_specialty
        ORDER BY total_phi_found DESC
        LIMIT 6
    """)
    print(cross.to_string(index=False))

    print("\n" + "="*60)
    print("  Phase 3 Complete!")
    print("  ─────────────────────────────────────────────────────")
    print("  DB tables : clinical_notes, processed_notes, audit_log")
    print("  Run tests : pytest tests/test_phase3.py -v")
    print("  Next      : Phase 4 — Flask REST API + dashboard")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()