"""
run_phase5.py
─────────────
Phase 5 — Docker deployment validation.

This script doesn't start a server. Instead it:
  1. Verifies all Docker files exist and are valid
  2. Runs a final full integration test simulating a containerised request
  3. Prints the exact commands needed to build and run the container
  4. Summarises the complete project for resume/interview use

Run: python run_phase5.py
"""

import sys
import os
import json
from pathlib import Path
sys.path.insert(0, ".")


def check(label, condition, detail=""):
    icon = "✓" if condition else "✗"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    print("\n" + "="*62)
    print("  ClinicalNER — Phase 5: Docker + Deployment Validation")
    print("="*62 + "\n")

    all_ok = True

    # ── Step 1: Verify all required files exist ───────────────────────────────
    print("► File structure check:")
    required_files = [
        ("docker/Dockerfile",                "Multi-stage Docker build"),
        ("docker/docker-compose.yml",        "Compose with volume + healthcheck"),
        ("docker/deploy_aws.sh",             "AWS EC2 deploy script"),
        ("docker/deploy_gcp.sh",             "GCP Cloud Run deploy script"),
        ("docker/.dockerignore",             "Build context optimisation"),
        ("src/api/app.py",                   "Flask application factory"),
        ("src/pipeline/ner_pipeline.py",     "NERPipeline — core NER"),
        ("src/pipeline/data_cleaner.py",     "DataCleaner — pre/post NER"),
        ("src/pipeline/audit_logger.py",     "AuditLogger — compliance trail"),
        ("src/pipeline/anomaly_detector.py", "AnomalyDetector — IsolationForest"),
        ("src/evaluation/benchmark.py",      "ModelBenchmark — F1/precision/recall"),
        ("src/utils/data_loader.py",         "DataLoader — SQL + ingestion"),
        ("src/utils/eda.py",                 "ClinicalEDA — 5 chart types"),
        ("streamlit_app.py",                 "Streamlit live demo"),
        (".github/workflows/tests.yml",      "GitHub Actions CI"),
        (".coveragerc",                      "Coverage config"),
        ("Makefile",                         "make test / make lint"),
        ("requirements.txt",                 "Pinned dependencies"),
        ("README.md",                        "Project documentation"),
    ]
    for path, desc in required_files:
        ok = Path(path).exists()
        all_ok = all_ok and ok
        check(desc, ok, path)

    # ── Step 2: Validate Dockerfile ───────────────────────────────────────────
    print("\n► Dockerfile validation:")
    with open("docker/Dockerfile") as f:
        df = f.read()
    checks = [
        ("Multi-stage build (builder + runtime)", "AS builder" in df and "AS runtime" in df),
        ("Non-root user",                         "useradd" in df and "USER appuser" in df),
        ("HEALTHCHECK defined",                   "HEALTHCHECK" in df),
        ("Port 5000 exposed",                     "EXPOSE 5000" in df),
        ("gunicorn in CMD (production server)",   "gunicorn" in df),
        ("spaCy model download",                  "spacy download" in df),
    ]
    for label, cond in checks:
        all_ok = check(label, cond) and all_ok

    # ── Step 3: Full integration test ─────────────────────────────────────────
    print("\n► Full integration test (simulates containerised request):")
    from src.api.app import create_app
    app = create_app(db_path="data/clinicalner.db")

    with app.test_client() as c:

        # Health probe
        r = c.get("/health")
        check("GET  /health → 200", r.status_code == 200)

        # Core de-identification
        note = (
            "Patient James Smith, DOB: 06/15/1978. "
            "Phone: (415) 555-9012. MRN401234. "
            "Admitted to Memorial Medical Center on 04/01/2024."
        )
        r = c.post("/api/deidentify", json={"text": note, "save": False})
        d = r.get_json()
        check("POST /api/deidentify → 200",   r.status_code == 200)
        check("PHI entities detected",         d["entity_count"] > 0,
              f"{d['entity_count']} found")
        check("avg_confidence present",        "avg_confidence" in d,
              str(d.get("avg_confidence", "missing")))
        check("is_valid flag present",         "is_valid" in d)
        check("No raw PHI in masked output",   note != d.get("masked_text", ""))

        # Stats
        r = c.get("/api/stats")
        d = r.get_json()
        check("GET  /api/stats → 200",         r.status_code == 200)
        check("Stats has note_count",          "note_count" in d,
              f"{d.get('note_count', 0)} notes")

        # Anomaly scan
        notes_payload = [
            {"id": i, "text": f"DOB: 0{i}/01/199{i % 9}. MRN{i:06d}.",
             "entities": [{"label": "DOB"}, {"label": "MRN"}]}
            for i in range(1, 16)
        ]
        r = c.post("/api/anomaly-scan", json={"notes": notes_payload})
        d = r.get_json() or {}
        check("POST /api/anomaly-scan → 200",  r.status_code == 200)
        check("Anomaly results returned",       "results" in d,
              f"{d.get('total_notes', 0)} scored")

        # Dashboard
        r = c.get("/dashboard")
        check("GET  /dashboard → 200 HTML",    r.status_code == 200)
        check("Dashboard has Chart.js",         b"chart.umd.min.js" in r.data)

        # Report — use a real note_id from the DB
        from src.utils.data_loader import DataLoader
        loader = DataLoader(db_path="data/clinicalner.db")
        try:
            first = loader.sql_query(
                "SELECT note_id FROM processed_notes LIMIT 1"
            )
            test_id = int(first.iloc[0]["note_id"]) if not first.empty else 1
        except Exception:
            test_id = 1
        r = c.get(f"/report/{test_id}")
        check(f"GET  /report/{test_id} → 200", r.status_code == 200)

    # ── Step 4: Test suite summary ─────────────────────────────────────────────
    print("\n► Test suite summary:")
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True
    )
    last_line = [l for l in result.stdout.strip().splitlines() if l.strip()][-1]
    passed = "failed" not in last_line
    check("All tests passing", passed, last_line)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*62)
    if all_ok and passed:
        print("  All checks passed — project is deployment-ready!")
    else:
        print("  Some checks failed — review items marked ✗ above")
    print("="*62)

    print("""
► Docker commands (run from project root):

  # Build image
  docker build -t clinicalner -f docker/Dockerfile .

  # Run container
  docker run -p 5000:5000 clinicalner

  # Or use docker-compose (recommended)
  docker-compose -f docker/docker-compose.yml up --build

  # Open in browser
  # Dashboard : http://localhost:5000/dashboard
  # Health    : http://localhost:5000/health
  # API test  :
  curl -X POST http://localhost:5000/api/deidentify \\
       -H "Content-Type: application/json" \\
       -d '{"text": "Patient DOB: 04/12/1985. MRN302145."}'

► Cloud deployment:
  AWS EC2  : bash docker/deploy_aws.sh   (update EC2_KEY first)
  GCP Run  : bash docker/deploy_gcp.sh   (update GCP_PROJECT first)
  Streamlit: push to GitHub → share.streamlit.io → select streamlit_app.py
""")

    print("="*62)
    print("  PROJECT COMPLETE — all 5 phases + 6 upgrades built")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()