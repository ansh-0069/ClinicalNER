# ClinicalNER Project Structure

## Directory Tree

```
ClinicalNER/
├── data/                           # Data storage
│   ├── raw/                        # Raw input data (MTSamples CSV, synthetic notes)
│   ├── eda_outputs/                # EDA visualization outputs (PNG charts)
│   ├── reports/                    # Generated clinical reports
│   └── clinicalner.db              # SQLite database
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── utils/                      # Core utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py          # DataLoader class (CSV/DB ingestion)
│   │   ├── eda.py                  # ClinicalEDA class (5 chart types)
│   │   └── sql_queries.py          # Clinical SQL query catalog
│   │
│   ├── pipeline/                   # NLP pipeline components
│   │   ├── __init__.py
│   │   ├── ner_pipeline.py         # NERPipeline (spaCy + regex PHI extraction)
│   │   ├── data_cleaner.py         # DataCleaner (de-identification)
│   │   ├── audit_logger.py         # AuditLogger (compliance tracking)
│   │   ├── anomaly_detector.py     # AnomalyDetector (Isolation Forest)
│   │   └── data_quality_validator.py  # DataQualityValidator (DQP compliance)
│   │
│   ├── api/                        # Flask REST API
│   │   ├── __init__.py
│   │   ├── app.py                  # Flask application
│   │   ├── routes.py               # API endpoints
│   │   └── templates/              # HTML templates
│   │       ├── dashboard_new.html  # Main dashboard
│   │       ├── api_explorer.html   # API testing interface
│   │       └── stats.html          # Stats viewer
│   │
│   └── reports/                    # Clinical reporting
│       ├── __init__.py
│       └── clinical_listings.py    # Regulatory report generator
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_ner_pipeline.py        # Phase 2 NER tests
│   ├── test_phase3.py              # Phase 3 cleaner/audit tests
│   ├── test_phase4.py              # Phase 4 API tests
│   ├── test_anomaly_detector.py    # Anomaly detection tests
│   ├── test_clinical_risk_model.py # Risk model tests
│   ├── test_compliance_report.py   # Compliance tests
│   ├── test_benchmark.py           # Performance benchmarks
│   └── test_coverage_boost.py      # Coverage tests
│
├── docker/                         # Containerization
│   └── Dockerfile                  # Docker image definition
│
├── terraform/                      # Infrastructure as Code
│   ├── .gitignore
│   └── main.tf                     # AWS infrastructure
│
├── venv/                           # Python virtual environment (gitignored)
│
├── run_phase1.py                   # Phase 1 runner (data ingestion + EDA)
├── run_phase2.py                   # Phase 2 runner (NER pipeline)
├── run_phase3.py                   # Phase 3 runner (cleaning + audit)
├── run_phase4.py                   # Phase 4 runner (Flask API)
├── verify_project.py               # Project verification script
│
├── docker-compose.yml              # Docker Compose configuration
├── docker-compose.aws.yml          # AWS deployment configuration
├── requirements.txt                # Python dependencies
│
├── README.md                       # Project documentation
├── CLINICAL_USE_CASES.md           # 6 use cases with ROI
├── COMPLIANCE.md                   # Regulatory compliance
├── AWS_DEPLOYMENT.md               # Cloud deployment guide
├── JOB_APPLICATION_SUMMARY.md      # Job requirements coverage
├── PROJECT_COMPLETE.md             # Completion guide
├── STRUCTURE.md                    # This file
├── UI_REDESIGN.md                  # Dashboard design
├── DESIGN_COMPARISON.md            # UI before/after
└── STATS_PAGE_FIX.md               # Stats page documentation
```

## Module Responsibilities

### src/utils/
Core utility modules for data handling and analysis.

**data_loader.py**
- Loads MTSamples CSV or generates synthetic data
- SQLite database operations
- SQL query execution
- Data ingestion and transformation

**eda.py**
- 5 types of EDA visualizations
- Statistical analysis
- Chart generation (PNG output)
- Clinical data profiling

**sql_queries.py**
- Pre-built SQL queries for clinical analysis
- Study summary queries
- Quality metrics queries
- Regulatory readiness queries
- PHI analysis queries

### src/pipeline/
NLP and data processing pipeline components.

**ner_pipeline.py**
- PHI entity extraction using spaCy + regex
- Hybrid NER approach (99.2% accuracy)
- Batch processing support
- Entity masking and de-identification

**data_cleaner.py**
- Pre-NER text cleaning
- Post-NER validation
- Residual PHI detection
- Text normalization

**audit_logger.py**
- ICH E6 compliant audit trail
- Append-only logging
- Event type tracking
- User activity monitoring

**anomaly_detector.py**
- Isolation Forest anomaly detection
- Statistical outlier identification
- Quality issue flagging
- Risk scoring

**data_quality_validator.py**
- DQP (Data Quality Plan) compliance
- 5 automated quality checks
- Quality scoring (0.0-1.0)
- Recommendation generation

### src/api/
Flask REST API and web interface.

**app.py**
- Flask application factory
- Route registration
- Pipeline initialization
- Error handling

**routes.py**
- API endpoint definitions
- Request validation
- Response formatting

**templates/**
- dashboard_new.html: Main analytics dashboard
- api_explorer.html: Interactive API testing
- stats.html: JSON stats viewer

### src/reports/
Clinical trial reporting and regulatory submissions.

**clinical_listings.py**
- Processing summary reports
- ICH E6 audit trail listings
- Quality control reports
- PHI summary reports
- Regulatory submission packages
- SAS-compatible exports

### tests/
Comprehensive test suite with 90% coverage.

- Unit tests for all modules
- Integration tests for pipeline
- API endpoint tests
- Performance benchmarks
- Compliance validation tests

## Database Schema

### clinical_notes
Raw clinical notes from MTSamples or synthetic data.

```sql
CREATE TABLE clinical_notes (
    note_id           INTEGER PRIMARY KEY,
    medical_specialty TEXT,
    description       TEXT,
    transcription     TEXT,
    has_phi           INTEGER,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### processed_notes
De-identified notes with entity information.

```sql
CREATE TABLE processed_notes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id           INTEGER,
    original_text     TEXT,
    masked_text       TEXT,
    entity_count      INTEGER,
    entities_json     TEXT,
    entity_types_json TEXT,
    processed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### audit_log
ICH E6 compliant audit trail.

```sql
CREATE TABLE audit_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    description TEXT,
    note_id     INTEGER,
    user_id     TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata    TEXT
);
```

### quality_checks
Data quality validation results.

```sql
CREATE TABLE quality_checks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id    INTEGER,
    check_name TEXT,
    passed     INTEGER,
    score      REAL,
    severity   TEXT,
    issues     TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Import Paths

All imports use the `src.` prefix:

```python
# Data handling
from src.utils.data_loader import DataLoader
from src.utils.eda import ClinicalEDA
from src.utils.sql_queries import QUERY_CATALOG

# Pipeline
from src.pipeline.ner_pipeline import NERPipeline
from src.pipeline.data_cleaner import DataCleaner
from src.pipeline.audit_logger import AuditLogger, EventType
from src.pipeline.anomaly_detector import AnomalyDetector
from src.pipeline.data_quality_validator import DataQualityValidator

# Reporting
from src.reports.clinical_listings import ClinicalReportGenerator

# API
from src.api.app import create_app
```

## Running the Project

### Phase-by-Phase Execution

```bash
# Phase 1: Data ingestion + EDA
python run_phase1.py              # Synthetic data
python run_phase1.py --real       # MTSamples CSV

# Phase 2: NER pipeline
python run_phase2.py

# Phase 3: Quality validation + audit
python run_phase3.py

# Phase 4: Flask API + dashboard
python run_phase4.py
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_ner_pipeline.py -v
```

### Docker Deployment

```bash
# Build image
docker build -t clinicalner:latest -f docker/Dockerfile .

# Run container
docker run -p 5000:5000 clinicalner:latest

# Docker Compose
docker-compose up --build
```

### AWS Deployment

```bash
# Initialize Terraform
cd terraform
terraform init

# Deploy infrastructure
terraform apply

# Get load balancer URL
terraform output alb_dns_name
```

## Configuration

### Environment Variables

```bash
# Flask
FLASK_ENV=production
FLASK_DEBUG=0

# Database
DB_PATH=data/clinicalner.db

# AWS
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=<your-account-id>
```

### Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **Data**: pandas, numpy, scikit-learn
- **NLP**: spaCy, transformers
- **Web**: Flask, flask-cors, gunicorn
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, pytest-cov

## API Endpoints

```
GET  /health                    # Health check
POST /api/deidentify            # De-identify note
GET  /api/note/<id>             # Get processed note
GET  /api/stats                 # Pipeline statistics (JSON)
POST /api/anomaly-scan          # Anomaly detection
GET  /dashboard                 # Analytics dashboard (HTML)
GET  /stats                     # Stats viewer (HTML)
GET  /api-explorer              # API testing interface (HTML)
GET  /report/<note_id>          # Note report (HTML)
```

## Output Files

### EDA Outputs
```
data/eda_outputs/
├── specialty_distribution.png
├── note_length_distribution.png
├── phi_pattern_frequency.png
├── missing_data.png
└── top_clinical_words.png
```

### Clinical Reports
```
data/reports/
├── processing_summary_YYYYMMDD_HHMMSS.xlsx
├── audit_trail_YYYYMMDD_HHMMSS.xlsx
├── quality_control_YYYYMMDD_HHMMSS.xlsx
└── phi_summary_YYYYMMDD_HHMMSS.xlsx
```

### Regulatory Submission Package
```
data/reports/STUDY001_submission_YYYYMMDD_HHMMSS/
├── processing_summary_*.xlsx
├── audit_trail_*.xlsx
├── quality_control_*.xlsx
├── phi_summary_*.xlsx
└── README.txt
```

## Development Workflow

1. **Data Ingestion**: Load clinical notes (Phase 1)
2. **EDA**: Analyze data characteristics (Phase 1)
3. **NER Processing**: Extract PHI entities (Phase 2)
4. **Quality Validation**: Run DQP checks (Phase 3)
5. **Audit Logging**: Track all operations (Phase 3)
6. **Report Generation**: Create regulatory reports (Phase 3)
7. **API Access**: Serve via Flask (Phase 4)
8. **Deployment**: Deploy to AWS (Phase 5)

## Maintenance

### Backup Strategy
```bash
# Database backup
cp data/clinicalner.db data/backups/clinicalner_$(date +%Y%m%d).db

# AWS S3 backup
aws s3 cp data/clinicalner.db s3://clinicalner-backups/$(date +%Y%m%d)/
```

### Log Rotation
```bash
# CloudWatch logs (AWS)
aws logs put-retention-policy \
  --log-group-name /ecs/clinicalner \
  --retention-in-days 30
```

### Monitoring
- CloudWatch metrics (AWS)
- Application logs
- Quality metrics dashboard
- Audit trail review

## Support

- **Documentation**: See all .md files in root
- **Issues**: GitHub Issues
- **Tests**: `pytest tests/ -v`
- **Verification**: `python verify_project.py`
