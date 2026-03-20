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

# ── 1. Ensure data directory exists ──────────────────────────────────────────
mkdir -p /app/data/raw /app/data/eda_outputs

# ── 2. Seed database on first boot ───────────────────────────────────────────
DB_FILE="/app/data/clinicalner.db"

HAS_NOTES=0

if [ -f "$DB_FILE" ]; then
    HAS_NOTES=$(python - <<'PY'
import sqlite3

db = "/app/data/clinicalner.db"
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

loader = DataLoader(raw_dir="data/raw", db_path="data/clinicalner.db")
df = loader.generate_synthetic_dataset(n_records=500)
loader.save_to_db(df, table="clinical_notes")
print("==> Seed complete: 500 rows in clinical_notes")
PY
else
    echo "==> Existing clinical_notes detected ($HAS_NOTES rows) — skipping seed."
fi

# ── 3. Start the server ───────────────────────────────────────────────────────
if [ "$FLASK_ENV" = "development" ]; then
    PORT_VALUE="${PORT:-5000}"
    echo "==> Starting Flask dev server on :$PORT_VALUE ..."
    python -c "
import sys
import os
sys.path.insert(0, '/app')
from src.api.app import create_app
app = create_app(db_path='/app/data/clinicalner.db')
app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True)
"
else
    PORT_VALUE="${PORT:-5000}"
    echo "==> Starting gunicorn (4 workers) on :$PORT_VALUE ..."
    # --workers 4     : handle concurrent requests
    # --timeout 120   : NER on long notes can take a few seconds
    # --access-logfile - : stream access logs to stdout (captured by Docker)
    exec gunicorn \
        --workers 4 \
        --timeout 120 \
        --bind 0.0.0.0:${PORT_VALUE} \
        --access-logfile - \
        --error-logfile - \
        "src.api.app:create_app()"
fi
