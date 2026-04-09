#!/bin/sh
# entrypoint.sh
# ─────────────────────────────────────────────────────────────────────────────
# Container startup script for ClinicalNER.
#
# What this does:
#   1. Creates the data directory if it doesn't exist (volume mount may be empty)
#   2. Seeds the DB with synthetic notes on first boot (no EDA plots)
#      idempotent: skip if DB already has data
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

# Azure App Service: route traffic to the port we bind. Default image uses 5000.
# WEBSITES_PORT app setting + PORT are both seen in some SKUs — prefer PORT.
PORT_VALUE="${PORT:-${WEBSITES_PORT:-5000}}"
export PORT="$PORT_VALUE"
echo "==> Listening on PORT=$PORT_VALUE"

# Small plans OOM with 4 spaCy workers; override with GUNICORN_WORKERS=1 if needed.
GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"

# ── 1. Ensure data directory exists ──────────────────────────────────────────
mkdir -p /app/data/raw /app/data/eda_outputs

# ── 2. Seed database on first boot ───────────────────────────────────────────
DB_FILE="${DB_PATH:-/app/data/clinicalner.db}"
DB_DIR=$(dirname "$DB_FILE")

# Some hosts may provide a DB_PATH on a location that this container user
# cannot create. Fall back to a guaranteed writable in-container path.
if ! mkdir -p "$DB_DIR"; then
    echo "==> WARN: cannot create DB directory '$DB_DIR' (DB_PATH=$DB_FILE)."
    DB_FILE="/app/data/clinicalner.db"
    DB_DIR="/app/data"
    mkdir -p "$DB_DIR"
    export DB_PATH="$DB_FILE"
    echo "==> Falling back to DB_PATH=$DB_PATH"
fi

HAS_NOTES=0

if [ -f "$DB_FILE" ]; then
    HAS_NOTES=$(python - <<'PY'
import sqlite3

import os

db = os.getenv("DB_PATH", "/app/data/clinicalner.db")
count = 0
try:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM clinical_notes")
    row = cur.fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
except Exception:
    count = 0
finally:
    try:
        conn.close()
    except Exception:
        pass
print(count)
PY
)
fi

if [ "$HAS_NOTES" -eq 0 ]; then
    echo "==> Seeding synthetic data into clinical_notes (first boot)..."
    python - <<'PY'
from src.utils.data_loader import DataLoader

import os

db_path = os.getenv("DB_PATH", "data/clinicalner.db")
loader = DataLoader(raw_dir="data/raw", db_path=db_path)
df = loader.generate_synthetic_dataset(n_records=500)
loader.save_to_db(df, table="clinical_notes")
print("==> Seed complete: 500 rows in clinical_notes")
PY
else
    echo "==> Existing clinical_notes detected ($HAS_NOTES rows) — skipping seed."
fi

# ── 3. Start the server ───────────────────────────────────────────────────────
if [ "$FLASK_ENV" = "development" ]; then
    echo "==> Starting Flask dev server on :$PORT_VALUE ..."
    python -c "
import sys
import os
sys.path.insert(0, '/app')
from src.api.app import create_app
app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True)
"
else
    echo "==> Starting gunicorn (${GUNICORN_WORKERS} workers) on :$PORT_VALUE ..."
    # --timeout 120   : NER on long notes can take a few seconds
    # --access-logfile - : stream access logs to stdout (captured by Docker)
    exec gunicorn \
        --workers "${GUNICORN_WORKERS}" \
        --timeout 120 \
        --bind "0.0.0.0:${PORT_VALUE}" \
        --access-logfile - \
        --error-logfile - \
        "src.api.app:create_app()"
fi
