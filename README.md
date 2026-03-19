# ClinicalNER — Clinical Notes De-Identification Pipeline

> **Portfolio Project** — Built for the Associate Clinical Programmer JD (0–2 yrs exp)

An end-to-end NLP pipeline that ingests unstructured clinical notes, extracts PHI entities (names, dates, hospitals, phone numbers, ages), de-identifies the text, and serves results via a Flask REST API — all containerized with Docker.

---

## Architecture

```
data/
  raw/              ← MTSamples CSV or synthetic notes
  clinicalner.db    ← SQLite (clinical_notes, processed_notes, audit_log)
  eda_outputs/      ← EDA charts (PNG)
src/
  utils/
    data_loader.py  ← DataLoader class (ingestion + SQL)
    eda.py          ← ClinicalEDA class (5 chart types)
  pipeline/
    ner_pipeline.py   ← NERPipeline class (hybrid regex + spaCy)
    data_cleaner.py   ← DataCleaner class (pre/post-NER cleaning)
    audit_logger.py   ← AuditLogger class (append-only event log)
  api/
    app.py            ← Flask application factory
docker/
  Dockerfile          ← spaCy model baked in, HEALTHCHECK, non-root user
  entrypoint.sh       ← DB seed on first boot, gunicorn in prod
docker-compose.yml
run_phase1.py  →  run_phase5.py
```

---

## Quick Start (Local — no Docker)

```bash
git clone <repo> && cd ClinicalNER
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python run_phase1.py   # seed 500 synthetic notes
python run_phase4.py   # start Flask on :5000
# open http://localhost:5000/dashboard
```

## Quick Start (Docker)

```bash
# Build + start (first run downloads spaCy model inside image — ~2 min)
docker compose up --build

# Subsequent runs (cached image — ~5 s)
docker compose up -d

# Logs
docker compose logs -f clinicalner

# Stop
docker compose down
```

Container auto-seeds the database with 500 synthetic notes on first boot.  
Dashboard: **http://localhost:5000/dashboard**

### Docker smoke test

```bash
python run_phase5.py           # build + test + leave container running
python run_phase5.py --teardown  # build + test + stop container
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/health` | Liveness probe (Docker / cloud LB) |
| `POST` | `/api/deidentify` | De-identify a clinical note |
| `GET`  | `/api/stats` | Corpus + audit statistics (JSON) |
| `GET`  | `/api/note/<id>` | Fetch a processed note by ID |
| `GET`  | `/dashboard` | Live EDA dashboard (Chart.js) |
| `GET`  | `/report/<id>` | Before/after diff view |

### Example — de-identify a note

```bash
curl -X POST http://localhost:5000/api/deidentify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145."}'
```

---

## Dataset

**MTSamples** — 4,999 real clinical transcriptions across 40 medical specialties.  
Download: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

No Kaggle account needed — `python run_phase1.py` uses built-in synthetic data (500 notes with realistic PHI patterns).

---

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data ingestion, SQL schema, EDA (DataLoader + ClinicalEDA) | ✅ Complete |
| 2 | Hybrid NER pipeline — regex + spaCy, 21 unit tests | ✅ Complete |
| 3 | DataCleaner (pre/post-NER) + AuditLogger, 24 unit tests | ✅ Complete |
| 4 | Flask REST API + Chart.js dashboard, 22 unit tests | ✅ Complete |
| 5 | Docker containerization + gunicorn + smoke test | ✅ Complete |

---

## Tech Stack

Python 3.11 · spaCy · pandas · SQLAlchemy · Flask · gunicorn · Docker · docker-compose

## JD Requirements Covered

- ✅ Python OOP (4 classes: DataLoader, NERPipeline, DataCleaner, AuditLogger)
- ✅ SQL — SQLite + SQLAlchemy ORM, analytical cross-table queries
- ✅ Unstructured clinical data — free-text NER and masking
- ✅ EDA on clinical datasets — 5 chart types via ClinicalEDA
- ✅ Flask REST API — 5 routes, consistent error codes
- ✅ Docker deployment — prod-grade Dockerfile, HEALTHCHECK, gunicorn
- ✅ NLP/ML — hybrid regex + spaCy model, PHI anomaly detection


---

## Architecture

```
data/
  raw/              ← MTSamples CSV or synthetic notes
  clinicalner.db    ← SQLite (clinical_notes, processed_notes, audit_log)
  eda_outputs/      ← EDA charts (PNG)
src/
  utils/
    data_loader.py  ← DataLoader class (ingestion + SQL)
    eda.py          ← ClinicalEDA class (5 chart types)
  pipeline/
    ner_pipeline.py   ← NERPipeline class (Phase 2)
    data_cleaner.py   ← DataCleaner class (Phase 3)
    audit_logger.py   ← AuditLogger class (Phase 3)
  api/
    app.py            ← Flask app (Phase 4)
    routes.py         ← API routes
tests/
  test_ner_pipeline.py  ← Phase 2 tests
  test_phase3.py        ← Phase 3 tests
docker/
  Dockerfile
docker-compose.yml
run_phase1.py
run_phase2.py
run_phase3.py
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
