"""
run_benchmark.py
────────────────
Compares regex-only vs spaCy-hybrid NER pipeline configurations.

Run:  python run_benchmark.py

Results are saved to data/benchmark_results.json and printed as a
Markdown table ready to paste into README.md.
"""

import sys
sys.path.insert(0, ".")

from src.evaluation.benchmark import ModelBenchmark

def main():
    print("\n" + "="*60)
    print("  ClinicalNER - Model Benchmark")
    print("="*60)

    bm = ModelBenchmark()   # uses built-in 30-example annotated test set
    print(f"\n  Test set: {len(bm.test_data)} annotated clinical notes\n")

    configs = [
        {"name": "regex-only (baseline)", "use_spacy": False},
        {"name": "spacy-hybrid",          "use_spacy": True},
    ]

    results = bm.run(configs)
    bm.save_report(results, "data/benchmark_results.json")
    bm.print_report(results)

    print("  Markdown table for README.md:")
    print("  " + "-"*50)
    print(bm.generate_readme_table(results))

    print("="*60)
    print("  Results saved -> data/benchmark_results.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
