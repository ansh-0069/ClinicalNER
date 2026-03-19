# ClinicalNER — NLP De-identification Pipeline

![CI](https://github.com/ansh-0069/ClinicalNER/actions/workflows/tests.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09a3d5)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Flask](https://img.shields.io/badge/flask-REST%20API-lightgrey)


> **Portfolio Project** — Built for the Associate Clinical Programmer JD (0–2 yrs exp)

An end-to-end NLP pipeline that ingests unstructured clinical notes, extracts PHI entities (names, dates, hospitals, phone numbers, MRNs), de-identifies the text, and serves results via a Flask REST API — all containerized with Docker.

---

## Architecture

```
ClinicalNER/
├── src/
│   ├── utils/
│   │   ├── data_loader.py    ← DataLoader class (ingestion + SQL)
│   │   └── eda.py            ← ClinicalEDA class (5 chart types)
│   ├── pipeline/
│   │   ├── ner_pipeline.py   ← NERPipeline class (hybrid regex + spaCy)
│   │   ├── data_cleaner.py   ← DataCleaner class (pre/post-NER cleaning)
│   │   └── audit_logger.py   ← AuditLogger class (append-only event log)
│   └── api/
│       └── app.py            ← Flask application factory (5 routes)
├── tests/
│   ├── test_ner_pipeline.py  ← 21 tests
│   ├── test_phase3.py        ← 50 tests (DataCleaner + AuditLogger)
│   └── test_phase4.py        ← 39 tests (Flask routes)
├── data/
│   ├── raw/                  ← MTSamples CSV or synthetic notes
│   ├── clinicalner.db        ← SQLite (clinical_notes, processed_notes, audit_log)
│   └── eda_outputs/          ← EDA charts (PNG)
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml
└── run_phase1.py → run_phase5.py
```

---

## Quick Start (Local)

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
docker compose up --build   # first run ~2 min (downloads spaCy model)
docker compose up -d        # subsequent runs ~5 s
docker compose logs -f clinicalner
docker compose down
```

Container auto-seeds the database with 500 synthetic notes on first boot.
Dashboard: **http://localhost:5000/dashboard**

---

## API Endpoints

| Method   | Route               | Description                        |
| -------- | ------------------- | ---------------------------------- |
| `GET`  | `/health`         | Liveness probe (Docker / cloud LB) |
| `POST` | `/api/deidentify` | De-identify a clinical note        |
| `GET`  | `/api/stats`      | Corpus + audit statistics (JSON)   |
| `GET`  | `/api/note/<id>`  | Fetch a processed note by ID       |
| `GET`  | `/dashboard`      | Live EDA dashboard (Chart.js)      |
| `GET`  | `/report/<id>`    | Before/after diff view             |

### Example

```bash
curl -X POST http://localhost:5000/api/deidentify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145."}'
```

Response:

```json
{
  "masked_text": "Patient DOB: [DATE]. Phone: [PHONE]. [MRN].",
  "entity_count": 3,
  "entity_types": {"DATE": 1, "PHONE": 1, "MRN": 1},
  "is_valid": true,
  "changes": []
}
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest --cov=src --cov-report=term-missing tests/
```

---

## Dataset

**MTSamples** — 4,999 real clinical transcriptions across 40 medical specialties.
Download: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

No Kaggle account needed — `python run_phase1.py` uses built-in synthetic data (500 notes with realistic PHI patterns).

---

## Build Phases

| Phase | Description                                                | Tests | Status      |
| ----- | ---------------------------------------------------------- | ----- | ----------- |
| 1     | Data ingestion, SQL schema, EDA (DataLoader + ClinicalEDA) | —    | ✅ Complete |
| 2     | Hybrid NER pipeline — regex + spaCy                       | 21    | ✅ Complete |
| 3     | DataCleaner (pre/post-NER) + AuditLogger                   | 50    | ✅ Complete |
| 4     | Flask REST API + Chart.js dashboard                        | 39    | ✅ Complete |
| 5     | Docker containerization + gunicorn + smoke test            | —    | ✅ Complete |

---

## Tech Stack

Python 3.11 · spaCy · pandas · SQLAlchemy · Flask · gunicorn · Docker · docker-compose · pytest

---

## JD Requirements Covered

| Requirement                | Implementation                                               |
| -------------------------- | ------------------------------------------------------------ |
| Python / OOP               | 4 classes: DataLoader, NERPipeline, DataCleaner, AuditLogger |
| SQL                        | SQLite + SQLAlchemy ORM, analytical cross-table queries      |
| Unstructured clinical data | Free-text NER and masking on MTSamples                       |
| EDA                        | 5 chart types via ClinicalEDA                                |
| ML models                  | Hybrid regex + spaCy NER pipeline                            |
| Anomaly detection          | Residual PHI scanning in DataCleaner                         |
| Flask / Django             | 5 REST routes, consistent HTTP status codes                  |
| Docker                     | Production Dockerfile, HEALTHCHECK, gunicorn                 |
| Cloud deployment           | Docker-ready, gunicorn WSGI server                           |
| Test coverage              | 110 tests, 81% coverage                                      |
