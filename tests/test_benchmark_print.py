"""Cover benchmark reporting helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark import BenchmarkResult, ModelBenchmark


def test_print_report_smoke(capsys):
    bm = ModelBenchmark(
        test_data=[
            {"text": "Phone: (415) 555-1111.", "entities": [{"start": 7, "end": 21, "label": "PHONE"}]},
        ]
    )
    results = [
        BenchmarkResult(
            model_name="m1",
            precision=1.0,
            recall=1.0,
            f1=1.0,
            latency_ms=1.0,
            entities_found=1,
            notes_tested=1,
        ),
        BenchmarkResult(
            model_name="m2-spacy",
            precision=0.9,
            recall=0.9,
            f1=0.9,
            latency_ms=2.0,
            entities_found=1,
            notes_tested=1,
        ),
    ]
    bm.print_report(results)
    out = capsys.readouterr().out
    assert "m1" in out or "Precision" in out
