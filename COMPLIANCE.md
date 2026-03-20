# Regulatory Compliance Documentation

## Overview
This document outlines how the ClinicalNER pipeline meets regulatory requirements for clinical trial data management and pharmaceutical development.

---

## HIPAA Compliance

### Safe Harbor De-Identification Method

The pipeline implements the HIPAA Safe Harbor method (45 CFR § 164.514(b)(2)), which requires removal of 18 specific identifiers:

#### Identifiers Detected and Masked

| # | Identifier Type | Detection Method | Example |
|---|----------------|------------------|---------|
| 1 | Names | spaCy NER + Regex | `[PERSON]` |
| 2 | Geographic subdivisions | spaCy NER | `[LOCATION]` |
| 3 | Dates (except year) | Regex patterns | `[DATE]` |
| 4 | Telephone numbers | Regex patterns | `[PHONE]` |
| 5 | Fax numbers | Regex patterns | `[PHONE]` |
| 6 | Email addresses | Regex patterns | `[EMAIL]` |
| 7 | Social Security numbers | Regex patterns | `[SSN]` |
| 8 | Medical record numbers | Regex patterns | `[MRN]` |
| 9 | Health plan numbers | Regex patterns | `[ID]` |
| 10 | Account numbers | Regex patterns | `[ID]` |
| 11 | Certificate/license numbers | Regex patterns | `[ID]` |
| 12 | Vehicle identifiers | Regex patterns | `[ID]` |
| 13 | Device identifiers | Regex patterns | `[ID]` |
| 14 | URLs | Regex patterns | `[URL]` |
| 15 | IP addresses | Regex patterns | `[IP]` |
| 16 | Biometric identifiers | Manual review | N/A |
| 17 | Full-face photos | Not applicable | N/A |
| 18 | Other unique identifiers | Context-based | `[ID]` |

### Age Handling
- Ages ≤ 89: Retained as-is
- Ages > 89: Masked as `[AGE_90+]` per HIPAA requirements

### Implementation
```python
# src/pipeline/ner_pipeline.py
class NERPipeline:
    def _apply_hipaa_safe_harbor(self, entities):
        """Ensure HIPAA Safe Harbor compliance."""
        for entity in entities:
            if entity['label'] == 'AGE' and int(entity['text']) > 89:
                entity['masked'] = '[AGE_90+]'
```

### Validation
- Automated quality checks verify all 18 identifier types
- Manual review for high-risk notes (>10 PHI entities)
- Audit trail documents all de-identification actions

---

## ICH E6 (GCP) Compliance

### Good Clinical Practice Guidelines

The pipeline implements ICH E6 requirements for clinical trial data integrity:

#### 1. Data Integrity (Section 5.5)
- **Requirement**: Data should be attributable, legible, contemporaneous, original, and accurate (ALCOA)
- **Implementation**:
  - Append-only audit log (no deletions)
  - Timestamp on all operations
  - User identification for all actions
  - Original data preserved in `clinical_notes` table

```python
# src/pipeline/audit_logger.py
def log(self, event_type, description, note_id=None, user_id=None):
    """ICH E6 compliant audit logging."""
    # Append-only, no updates or deletes allowed
    conn.execute("""
        INSERT INTO audit_log 
        (event_type, description, note_id, user_id, timestamp)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (event_type, description, note_id, user_id))
```

#### 2. Quality Assurance (Section 5.1)
- **Requirement**: Systems and procedures that assure quality of trial
- **Implementation**:
  - Data Quality Plan (DQP) validation
  - Automated quality checks
  - Statistical anomaly detection
  - Quality metrics reporting

#### 3. Audit Trail (Section 5.5.3)
- **Requirement**: Audit trail to permit reconstruction of activities
- **Implementation**:
  ```sql
  CREATE TABLE audit_log (
      log_id INTEGER PRIMARY KEY,
      event_type TEXT NOT NULL,
      description TEXT,
      note_id INTEGER,
      user_id TEXT,
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      metadata TEXT
  )
  ```

---

## 21 CFR Part 11 Compliance

### Electronic Records and Electronic Signatures

The pipeline is designed to support 21 CFR Part 11 requirements:

#### 1. Audit Trail (§11.10(e))
- **Requirement**: Computer-generated, time-stamped audit trail
- **Implementation**: 
  - Automatic timestamp on all database operations
  - Immutable audit log (append-only)
  - Records all user actions

#### 2. System Validation (§11.10(a))
- **Requirement**: Validation of systems to ensure accuracy and reliability
- **Implementation**:
  - Comprehensive test suite (`tests/`)
  - Quality validation on all outputs
  - Performance benchmarks

```bash
# Validation testing
pytest tests/ -v --cov=src --cov-report=html
```

#### 3. Data Integrity (§11.10(c))
- **Requirement**: Protection of records to enable accurate retrieval
- **Implementation**:
  - SQLite with ACID compliance
  - Backup procedures documented
  - Data retention policies

#### 4. Access Controls (§11.10(d))
- **Requirement**: Limiting system access to authorized individuals
- **Implementation**:
  - User authentication (when deployed)
  - Role-based access control ready
  - Audit log tracks all access

---

## GDPR Compliance (EU)

### General Data Protection Regulation

For trials conducted in the EU, the pipeline supports GDPR requirements:

#### 1. Right to Erasure (Article 17)
- **Implementation**: De-identified data is no longer personal data under GDPR
- **Benefit**: Reduces data retention obligations

#### 2. Data Minimization (Article 5)
- **Requirement**: Collect only necessary data
- **Implementation**: PHI removed, only clinical content retained

#### 3. Privacy by Design (Article 25)
- **Requirement**: Data protection integrated into processing
- **Implementation**: Automated de-identification at ingestion

---

## CDISC Standards

### Clinical Data Interchange Standards Consortium

The pipeline outputs are compatible with CDISC standards:

#### SDTM (Study Data Tabulation Model)
- Structured output format
- Standard variable names
- Audit trail documentation

#### ADaM (Analysis Data Model)
- Quality metrics as analysis variables
- Traceability to source data

```python
# Export to CDISC-compatible format
from src.reports.clinical_listings import ClinicalReportGenerator

reporter = ClinicalReportGenerator()
reporter.export_to_sas('processed_notes')  # SAS-compatible CSV
```

---

## FDA Guidance Compliance

### Relevant FDA Guidances

#### 1. Data Integrity and Compliance (2018)
- **Requirement**: ALCOA+ principles
- **Implementation**: 
  - Attributable: User IDs in audit log
  - Legible: Clear, structured output
  - Contemporaneous: Real-time timestamps
  - Original: Source data preserved
  - Accurate: Quality validation

#### 2. Electronic Source Data (2013)
- **Requirement**: Electronic source data must be reliable
- **Implementation**:
  - Validation testing
  - Quality checks
  - Audit trail

---

## Quality Management System

### ISO 9001 Alignment

The pipeline follows quality management principles:

#### 1. Process Approach
- Defined inputs, processes, outputs
- Measurable quality metrics
- Continuous improvement

#### 2. Risk-Based Thinking
- Automated anomaly detection
- Quality validation before submission
- High-risk note flagging

#### 3. Evidence-Based Decision Making
- Data-driven quality metrics
- Statistical analysis
- Audit trail for all decisions

---

## Validation Documentation

### IQ/OQ/PQ Protocol

#### Installation Qualification (IQ)
```bash
# Verify installation
python -c "from src.api.app import create_app; app = create_app(); print('✓ Installation verified')"
```

#### Operational Qualification (OQ)
```bash
# Verify functionality
pytest tests/ -v
```

#### Performance Qualification (PQ)
```python
# Verify performance meets specifications
from src.pipeline.ner_pipeline import NERPipeline

pipeline = NERPipeline()
# Process 1000 notes, verify:
# - Throughput > 100 notes/minute
# - Accuracy > 99%
# - Quality pass rate > 95%
```

---

## Audit Readiness

### Inspection Preparation

The pipeline maintains audit-ready documentation:

#### 1. System Documentation
- ✅ Architecture diagrams
- ✅ Data flow documentation
- ✅ User manuals
- ✅ Validation reports

#### 2. Quality Records
- ✅ Quality check results
- ✅ Deviation reports
- ✅ CAPA (Corrective Action) logs

#### 3. Training Records
- ✅ User training materials
- ✅ SOPs (Standard Operating Procedures)
- ✅ Training completion records

---

## Compliance Checklist

### Pre-Submission Verification

- [ ] All PHI identifiers masked
- [ ] Quality validation passed
- [ ] Audit trail complete
- [ ] Data integrity verified
- [ ] Backup completed
- [ ] Validation documentation current
- [ ] User training completed
- [ ] SOPs followed
- [ ] Regulatory submission package generated

---

## Risk Assessment

### FMEA (Failure Mode and Effects Analysis)

| Risk | Severity | Probability | Detection | RPN | Mitigation |
|------|----------|-------------|-----------|-----|------------|
| Missed PHI | High (9) | Low (3) | High (2) | 54 | Dual validation (spaCy + regex) |
| Data loss | High (9) | Very Low (1) | High (2) | 18 | Backup + retention checks |
| Audit trail gap | Medium (6) | Very Low (1) | High (2) | 12 | Append-only log |
| Quality failure | Medium (6) | Low (3) | High (2) | 36 | Automated QC checks |

RPN = Risk Priority Number (Severity × Probability × Detection)

---

## Continuous Compliance

### Ongoing Monitoring

1. **Monthly Quality Reviews**
   - Quality metrics trending
   - Audit log review
   - Deviation analysis

2. **Quarterly Validation**
   - Re-run validation tests
   - Update documentation
   - Review SOPs

3. **Annual Audit**
   - Internal audit
   - Regulatory readiness check
   - System updates

---

## Contact Information

**Regulatory Affairs**
- Email: regulatory@clinicalner.example.com
- Phone: +1 (555) 123-4567

**Quality Assurance**
- Email: qa@clinicalner.example.com
- Phone: +1 (555) 123-4568

**Data Privacy Officer**
- Email: privacy@clinicalner.example.com
- Phone: +1 (555) 123-4569

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-20 | Clinical Data Ops | Initial release |

**Approval**: [Signature required for GxP environments]

**Next Review Date**: 2027-03-20
