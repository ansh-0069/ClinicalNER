"""
test_benchmark.py
─────────────────
Tests for ModelBenchmark evaluation framework.
Run: pytest tests/test_benchmark.py -v
"""

import sys, json
sys.path.insert(0, ".")

import pytest
from src.evaluation.benchmark import ModelBenchmark, BenchmarkResult


@pytest.fixture
def bm():
    return ModelBenchmark()   # uses built-in test set

@pytest.fixture
def mini_bm():
    """Tiny 5-example set for fast unit tests."""
    return ModelBenchmark(test_data=[
        {"text": "DOB: 04/12/1985. Phone: (415) 555-9876.",
         "entities": [{"start": 0, "end": 15, "label": "DOB"},
                      {"start": 17, "end": 31, "label": "PHONE"}]},
        {"text": "MRN302145 admitted 01/15/2024.",
         "entities": [{"start": 0, "end": 9,  "label": "MRN"},
                      {"start": 19, "end": 29, "label": "DATE"}]},
        {"text": "Age: 58. Contact: (312) 555-0034.",
         "entities": [{"start": 0, "end": 7,  "label": "AGE"},
                      {"start": 18, "end": 32, "label": "PHONE"}]},
        {"text": "No PHI in this sentence at all.",
         "entities": []},
        {"text": "Follow-up on 09/24/2022.",
         "entities": [{"start": 13, "end": 23, "label": "DATE"}]},
    ])


def test_default_test_set_loads(bm):
    assert len(bm.test_data) == 30

def test_test_set_has_required_keys(bm):
    for item in bm.test_data:
        assert "text"     in item
        assert "entities" in item

def test_run_returns_results(mini_bm):
    results = mini_bm.run([{"name": "regex-only", "use_spacy": False}])
    assert len(results) == 1

def test_result_is_frozen_dataclass(mini_bm):
    results = mini_bm.run([{"name": "regex-only", "use_spacy": False}])
    with pytest.raises(Exception):
        results[0].f1 = 999.0     # frozen=True should raise

def test_precision_in_range(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    assert 0.0 <= results[0].precision <= 1.0

def test_recall_in_range(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    assert 0.0 <= results[0].recall <= 1.0

def test_f1_in_range(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    assert 0.0 <= results[0].f1 <= 1.0

def test_latency_is_positive(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    assert results[0].latency_ms >= 0.0

def test_notes_tested_matches_dataset(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    assert results[0].notes_tested == 5

def test_multiple_configs(mini_bm):
    configs = [
        {"name": "config-A", "use_spacy": False},
        {"name": "config-B", "use_spacy": False},
    ]
    results = mini_bm.run(configs)
    assert len(results) == 2
    assert results[0].model_name == "config-A"
    assert results[1].model_name == "config-B"

def test_readme_table_output(mini_bm):
    results = mini_bm.run([{"name": "regex-only", "use_spacy": False}])
    table = mini_bm.generate_readme_table(results)
    assert "| Model |" in table
    assert "regex-only" in table
    assert "Precision" in table

def test_save_report(mini_bm, tmp_path):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    path = str(tmp_path / "results.json")
    mini_bm.save_report(results, path)
    with open(path) as f:
        data = json.load(f)
    rows = data["results"] if isinstance(data, dict) else data
    assert len(rows) == 1
    assert rows[0]["model_name"] == "test"
    assert "f1" in rows[0]
    if isinstance(data, dict):
        assert data.get("schema_version") == "benchmark/v2"
        assert "precision" in data

def test_to_dict_has_all_fields(mini_bm):
    results = mini_bm.run([{"name": "test", "use_spacy": False}])
    d = results[0].to_dict()
    for key in ["model_name", "precision", "recall", "f1",
                "latency_ms", "entities_found", "notes_tested"]:
        assert key in d

def test_perfect_precision_on_easy_set():
    """All annotations are exact regex hits — expect precision = 1.0."""
    bm = ModelBenchmark(test_data=[
        {"text": "Phone: (415) 555-9876.",
         "entities": [{"start": 7, "end": 21, "label": "PHONE"}]},
    ])
    results = bm.run([{"name": "regex", "use_spacy": False}])
    assert results[0].precision == 1.0

def test_zero_recall_on_missed_entities():
    """Annotated entity that regex will never match → recall = 0."""
    bm = ModelBenchmark(test_data=[
        {"text": "Patient is well.",
         "entities": [{"start": 0, "end": 7, "label": "PERSON"}]},
    ])
    results = bm.run([{"name": "regex", "use_spacy": False}])
    assert results[0].recall == 0.0
