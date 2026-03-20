"""
run_phase4.py
─────────────
Starts the ClinicalNER Flask development server.

Usage:
    python run_phase4.py               # dev server on port 5000
    python run_phase4.py --port 8080   # custom port

Endpoints:
    POST http://localhost:5000/api/deidentify   ← core API
    GET  http://localhost:5000/api/stats        ← pipeline stats (JSON)
    GET  http://localhost:5000/dashboard        ← live dashboard
    GET  http://localhost:5000/report/<note_id> ← before/after view
    GET  http://localhost:5000/health           ← liveness probe

Quick test (run while server is up):
    curl -X POST http://localhost:5000/api/deidentify \
         -H "Content-Type: application/json" \
         -d '{"text": "Patient DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145."}'
"""

import sys
sys.path.insert(0, ".")

from src.api.app import create_app

if __name__ == "__main__":
    port = 5000
    if "--port" in sys.argv:
        idx  = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    app = create_app(db_path="data/clinicalner.db")

    print("\n" + "="*60)
    print("  ClinicalNER — Phase 4: Flask API")
    print("="*60)
    print(f"\n  Dashboard  : http://localhost:{port}/dashboard")
    print(f"  API        : POST http://localhost:{port}/api/deidentify")
    print(f"  Stats      : http://localhost:{port}/stats")
    print(f"  API Stats  : http://localhost:{port}/api/stats (JSON)")
    print(f"  Explorer   : http://localhost:{port}/api-explorer")
    print(f"  Health     : http://localhost:{port}/health")
    print(f"  Report     : http://localhost:{port}/report/1")
    print("\n  Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=port, debug=False)