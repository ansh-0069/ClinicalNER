# ClinicalNER — Clinical Notes De-Identification Pipeline

> **Portfolio Project** — Built for the Associate Clinical Programmer JD

An end-to-end NLP pipeline that ingests unstructured clinical notes, extracts PHI entities (names, dates, hospitals, phone numbers, ages), de-identifies the text, and serves results via a Flask REST API — all containerized with Docker.

---

## Architecture

```
data/raw/          ← MTSamples CSV or synthetic notes
data/clinicalner.db ← SQLite (clinical_notes, processed_notes, audit_log)
data/eda_outputs/  ← EDA charts (PNG)
src/
  utils/
    data_loader.py ← DataLoader class (ingestion + SQL)
    eda.py         ← ClinicalEDA class (5 chart types)
  pipeline/
    ner_pipeline.py   ← NERPipeline class (Phase 2)
    data_cleaner.py   ← DataCleaner class (Phase 3)
    audit_logger.py   ← AuditLogger class (Phase 3)
  api/
    app.py         ← Flask app (Phase 4)
    routes.py      ← API routes
docker/
  Dockerfile
  docker-compose.yml
```

## Quick Start

```bash
# 1. Clone & create venv
git clone <repo>
cd ClinicalNER
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run Phase 1 (synthetic data, no Kaggle needed)
python run_phase1.py

# 4. Run Phase 1 with real MTSamples (after downloading)
#    → place mtsamples.csv in data/raw/
python run_phase1.py --real
```

## Dataset

**MTSamples** — 4,999 real clinical transcriptions across 40 medical specialties.
Download: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Alternatively, `run_phase1.py` (no flags) uses the built-in synthetic dataset — 500 notes with realistic PHI patterns across 10 specialties.

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data ingestion, SQL schema, EDA | ✅ Complete |
| 2 | NLP/NER pipeline (spaCy + OOP) | 🔄 Next |
| 3 | Data cleaning + audit logging | ⏳ |
| 4 | Flask REST API + dashboard | ⏳ |
| 5 | Docker + cloud deployment | ⏳ |

## Tech Stack

Python 3.11 · spaCy · HuggingFace Transformers · Pandas · SQLAlchemy · Flask · Docker · AWS EC2

## JD Requirements Covered

- ✅ Python OOP (DataLoader, ClinicalEDA classes)
- ✅ SQL (SQLAlchemy + raw SQL queries)
- ✅ EDA on clinical datasets
- ✅ Unstructured data handling (clinical free-text)
- ✅ Flask REST API (Phase 4)
- ✅ Docker deployment (Phase 5)
