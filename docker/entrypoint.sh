#!/bin/sh
# entrypoint.sh
# ─────────────────────────────────────────────────────────────────────────────
# Container startup script for ClinicalNER.
#
# What this does:
#   1. Creates the data directory if it doesn't exist (volume mount may be empty)
#   2. Runs the Phase 1 seeding script on first boot (creates SQLite DB + 500
#      synthetic notes) — idempotent: skip if DB already has data
#   3. Starts the Flask app with gunicorn in production mode
#      (or plain Flask in dev mode via FLASK_ENV=development)
#
# Why gunicorn over Flask dev server in production?
#   Flask's built-in server is single-threaded and not designed for concurrent
#   requests. Gunicorn spins up multiple workers to handle real load.
#   gunicorn is listed in requirements.txt for this reason.
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error

echo "==> ClinicalNER container starting..."

# ── 1. Ensure data directory exists ──────────────────────────────────────────
mkdir -p /app/data/raw /app/data/eda_outputs

# ── 2. Seed database on first boot ───────────────────────────────────────────
DB_FILE="/app/data/clinicalner.db"

if [ ! -f "$DB_FILE" ]; then
    echo "==> No database found — seeding with synthetic data (first boot)..."
    python run_phase1.py && echo "==> Phase 1 seed complete."
else
    echo "==> Database already exists — skipping seed."
fi

# ── 3. Start the server ───────────────────────────────────────────────────────
if [ "$FLASK_ENV" = "development" ]; then
    echo "==> Starting Flask dev server on :5000 ..."
    python -c "
import sys
sys.path.insert(0, '/app')
from src.api.app import create_app
app = create_app(db_path='/app/data/clinicalner.db')
app.run(host='0.0.0.0', port=5000, debug=True)
"
else
    echo "==> Starting gunicorn (4 workers) on :5000 ..."
    # --workers 4     : handle concurrent requests
    # --timeout 120   : NER on long notes can take a few seconds
    # --access-logfile - : stream access logs to stdout (captured by Docker)
    exec gunicorn \
        --workers 4 \
        --timeout 120 \
        --bind 0.0.0.0:5000 \
        --access-logfile - \
        --error-logfile - \
        "src.api.app:create_app()"
fi
