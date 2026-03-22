# Listing data guide — DM + free-text export

## Purpose

Describes columns in **`dm_note_listing_*.csv`** produced by `ClinicalReportGenerator.generate_dm_free_text_listing()` after running `python run_structured_demo.py`.

This is a **portfolio demonstration** of joining CDISC-inspired **DM-style** fields to unstructured narrative and NER outputs—not a submission-ready SDTM package.

## File lineage

1. `data/dm_subject_demo.csv` — synthetic DM-style rows keyed by `note_id`.
2. `run_structured_demo.py` — loads into SQLite table `subject_dm`.
3. Listing — joins `subject_dm` → `clinical_notes` → `processed_notes` (if present).

## Column dictionary

| Column | Description | Source |
|--------|-------------|--------|
| STUDYID | Demo study identifier | `subject_dm.studyid` |
| USUBJID | Unique subject ID (demo) | `subject_dm.usubjid` |
| SITEID | Site / investigator site code | `subject_dm.siteid` |
| NOTE_ID | FK to `clinical_notes.note_id` | `subject_dm.note_id` |
| AGE | Age at reference (years) | `subject_dm.age` |
| SEX | Sex (M/F) | `subject_dm.sex` |
| BRTHDTC | Birth date (ISO-like demo) | `subject_dm.brthdtc` |
| medical_specialty | Clinical specialty of the note | `clinical_notes` |
| description | Short note description | `clinical_notes` |
| entity_count | PHI entities detected | `processed_notes` (null if not processed) |
| processed_at | Last NER run timestamp | `processed_notes` |

## Privacy note

Demo CSV uses synthetic values. Production listings must follow protocol, DMP, and transfer agreements.
