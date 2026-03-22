# Traceability matrix — DQP to implementation

Maps **Data Quality Plan** (`docs/DATA_QUALITY_PLAN.md`) checks to code and automated tests.

| DQP / Check ID | Requirement (summary) | Implementation | Test / evidence |
|----------------|----------------------|----------------|-----------------|
| V1 | Schema: required fields present on ingest | `DataLoader.save_to_db`, `_validate_columns` | `tests/` (loader usage via phases) |
| V2 | Text sanity before NER | `DataCleaner.clean` | `test_phase3.py` |
| V3 | Duplicate / empty rows handled | `DataLoader._basic_clean` | `test_phase3.py` |
| V4 | Pre-pipeline normalisation | `DataCleaner` | `test_phase3.py` |
| V5 | PHI span detection | `NERPipeline.process_note` | `test_ner_pipeline.py` |
| V6 | Outlier entity density | `AnomalyDetector` | `test_anomaly_detector.py` |
| DQ-SPEC | Specialty codelist | `data/specialty_vocab.json`, `dq_vocab.py` | Conformity in `/api/data-quality` |
| BM | Benchmark precision/recall/F1 | `src/evaluation/benchmark.py`, `run_benchmark.py` | `test_benchmark.py`, `nightly-benchmark.yml` |
| REG | Model version governance | `models/model_registry.json` | `GET /api/model-registry` |
| DM-LIST | Structured + unstructured join | `subject_dm`, `run_structured_demo.py` | `generate_dm_free_text_listing` output CSV |

_Update this table when validation rules or module names change._
