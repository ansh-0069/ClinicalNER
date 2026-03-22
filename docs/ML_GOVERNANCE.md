# ML governance — NER and benchmark

## Scope

- **Primary model:** entry in `models/model_registry.json` with the same `id` as `active_model` (currently spaCy hybrid + regex).
- **Evaluation:** `run_benchmark.py` writes `data/benchmark_results.json` in **benchmark/v2** envelope format with top-level precision/recall/F1 for the production (spaCy) row.

## Responsibilities

1. **Change control:** Any change to thresholds in `docs/DATA_QUALITY_PLAN.md` or `model_registry.json` should be committed with a short rationale (PR description).
2. **Regression:** CI runs `tests.yml` on push; scheduled `nightly-benchmark.yml` compares metrics to DQP minima. Adjust workflow env `F1_MIN` / `PRECISION_MIN` / `RECALL_MIN` only when the validation set or model family changes.
3. **Failure modes:** False **negative** PHI is the highest-risk error; monitor **recall** on the held-out annotation set and zero-PHI rate in `/api/data-quality`.

## Not in scope (demo)

- Formal CSV/SAS validation (P21), locked production environment, or independent biostatistical QC.
