# Data Quality Plan — ClinicalNER De-identification Pipeline

## 1. Purpose

This document defines the data quality standards, validation checks,  
and remediation procedures for the ClinicalNER corpus and its processed outputs.  
It aligns with HIPAA § 164.514(b) and NIST SP 800-188 de-identification guidelines.

---

## 2. Data Sources

| Source | Format | Expected Volume | Update Frequency |
|--------|--------|-----------------|-----------------|
| MTSamples clinical notes | Plain text (CSV) | ~5,000 records | One-time bulk + quarterly refresh |
| HL7/FHIR export feeds | JSON | Variable | Nightly |
| Manual annotations | JSON/JSONL | ~500 verified | Per sprint |

---

## 3. Quality Dimensions

### 3.1 Completeness

- **Check:** No `NULL`/`NaN` in `text`, `specialty`, `note_date` columns.  
- **Threshold:** ≥ 98% completeness per column before processing.  
- **Remediation:** Rows below threshold are quarantined to `data/quarantine/` with reason logged.

### 3.2 Conformity

- **Check:** `note_date` parses as ISO-8601; `specialty` matches the controlled vocabulary in `data/specialty_vocab.json`.  
- **Threshold:** 100% conformity required (hard fail on import).  
- **Remediation:** Auto-normalise casing/whitespace; reject unparseable dates with a structured error.

### 3.3 Consistency (PHI Detection)

- **Check:** Every run of `NERPipeline.process()` must detect ≥ 1 PHI entity for notes > 200 chars (unless explicitly flagged `phi_free`).  
- **Threshold:** Suspicion flag raised if entity count = 0 for > 5% of batch.  
- **Remediation:** Route flagged notes to manual review queue; log to `audit_log` table with `EventType.ANOMALY`.

### 3.4 Accuracy (Benchmark Cadence)

| Metric | Minimum Acceptable | Target |
|--------|--------------------|--------|
| Precision (PHI detection) | 0.85 | 0.92 |
| Recall (PHI detection) | 0.90 | 0.97 |
| F1 Score | 0.88 | 0.94 |

Benchmarks run nightly via `.github/workflows/nightly-benchmark.yml`.  
Results persist in `data/benchmark_results.json`; alerts fire when any metric drops > 0.03 vs prior run.

### 3.5 Timeliness

- Processing latency SLA: < 2 seconds per note at 95th percentile.  
- Nightly job must complete before 04:00 UTC.

---

## 4. Validation Pipeline

```
Input note
   │
   ├─► [V1] Schema validation (non-null fields, type checks)
   │
   ├─► [V2] Text sanity checks (length ≥ 50 chars, UTF-8 decodable)
   │
   ├─► [V3] Duplicate detection (SHA-256 hash of raw text → skip if seen)
   │
   ├─► [V4] DataCleaner.clean() → strip artefacts, normalise whitespace
   │
   ├─► [V5] NERPipeline.process() → extract PHI entities
   │
   ├─► [V6] AnomalyDetector.score() → flag outlier entity densities
   │
   └─► Persist to processed_notes + audit_log
```

---

## 5. Automated Checks

| Check ID | Location | Tool | Trigger |
|----------|----------|------|---------|
| V1 | `DataLoader.load_notes()` | Python assertions | On import |
| V2 | `DataCleaner.clean()` | Custom rules | Pre-pipeline |
| V3 | `DataLoader.load_notes()` | SQLite UNIQUE constraint | On insert |
| V4 | `DataCleaner.clean()` | Regex + unicode normalisation | Pre-pipeline |
| V5 | `NERPipeline.process()` | spaCy + regex | Core pipeline |
| V6 | `AnomalyDetector.score()` | Isolation Forest | Post-pipeline |
| BM | `run_benchmark.py` | Precision/Recall/F1 | Nightly CI |

---

## 6. Data Quality Metrics API

`GET /api/data-quality` returns a live JSON report:

```json
{
  "completeness": {"score": 0.992, "null_counts": {"medical_specialty": 0, "transcription": 0}},
  "conformity":   {"malformed_dates": 0, "invalid_specialties": 0, "vocab_version": "data/specialty_vocab.json"},
  "consistency":  {"zero_phi_rate": 0.031},
  "accuracy":     {"precision": 0.921, "recall": 0.963, "f1": 0.942, "schema": "benchmark/v2"},
  "timeliness":   {"p95_latency_ms": 1340, "last_run_utc": "2024-01-15T03:47:22Z"}
}
```

---

## 7. Incident Response

| Severity | Condition | Owner | SLA |
|----------|-----------|-------|-----|
| P1 | F1 drops below 0.80 | ML Engineer | 2 hours |
| P2 | Completeness < 95% | Data Engineer | 8 hours |
| P3 | Processing latency > 5s p95 | Backend Engineer | 24 hours |
| P4 | Minor quality dips / informational alerts | On-call | Next sprint |

---

## 8. Versioning & Changelog

Changes to validation thresholds or rules MUST:
1. Be committed with a PR referencing the affected metric.
2. Update `models/model_registry.json` with the new model/config version.
3. Update `docs/TRACEABILITY_MATRIX.md` and record before/after benchmark metrics in the PR description.

---

_Last updated: 2025-Q1_
