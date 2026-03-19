"""
app.py
──────
Flask application factory for ClinicalNER.

Routes
------
  POST /api/deidentify          — de-identify a clinical note (core API)
  GET  /api/note/<id>           — fetch a processed note by ID
  GET  /api/stats               — corpus + pipeline statistics (JSON)
  GET  /dashboard               — live EDA + audit dashboard (HTML)
  GET  /report/<note_id>        — before/after diff view (HTML)
  GET  /health                  — liveness probe (for Docker/cloud)

Design decisions:
  - Application factory pattern (create_app()) — standard Flask practice,
    makes the app testable and importable without side effects.
  - All pipeline objects initialised ONCE at app startup via app.config,
    not on every request — avoids reloading spaCy model per call.
  - Errors return JSON with a consistent shape:
    {"error": "...", "status": 400} — matches what a frontend expects.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_loader import DataLoader
from src.pipeline.ner_pipeline import NERPipeline
from src.pipeline.data_cleaner import DataCleaner
from src.pipeline.audit_logger import AuditLogger, EventType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ── Application factory ───────────────────────────────────────────────────────

def create_app(db_path: str = "data/clinicalner.db") -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    db_path : path to SQLite database (override for testing)
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app)   # allow cross-origin requests (needed when frontend is separate)

    # ── Initialise pipeline components once at startup ────────────────────────
    # Key decision: store in app.config so all request handlers share
    # the same instances — avoids reloading the spaCy model on every request.
    app.config["DB_PATH"]  = db_path
    app.config["LOADER"]   = DataLoader(db_path=db_path)
    app.config["PIPELINE"] = NERPipeline(db_path=db_path, use_spacy=True)
    app.config["CLEANER"]  = DataCleaner(strict_mode=False)
    app.config["AUDIT"]    = AuditLogger(db_path=db_path)

    logger.info("ClinicalNER Flask app initialised | db=%s", db_path)

    # ── Register routes ───────────────────────────────────────────────────────
    _register_api_routes(app)
    _register_ui_routes(app)

    return app


# ── API routes ────────────────────────────────────────────────────────────────

def _register_api_routes(app: Flask) -> None:

    @app.route("/health")
    def health():
        """Liveness probe — Docker HEALTHCHECK and cloud load balancers call this."""
        return jsonify({"status": "ok", "service": "ClinicalNER"})

    @app.route("/api/deidentify", methods=["POST"])
    def deidentify():
        """
        De-identify a clinical note.

        Request body (JSON):
          {
            "text":     "Patient James Smith...",   ← required
            "note_id":  42,                          ← optional
            "save":     true                         ← optional, default true
          }

        Response (JSON):
          {
            "note_id":       42,
            "original_text": "...",
            "masked_text":   "...",
            "entities":      [...],
            "entity_count":  5,
            "entity_types":  {"DATE": 2, "PHONE": 1, ...},
            "is_valid":      true,
            "changes":       [...]
          }
        """
        pipeline: NERPipeline = app.config["PIPELINE"]
        cleaner:  DataCleaner = app.config["CLEANER"]
        audit:    AuditLogger = app.config["AUDIT"]

        # ── Input validation ──────────────────────────────────────────────────
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "status": 400}), 400

        data    = request.get_json()
        text    = data.get("text", "").strip()
        note_id = data.get("note_id")
        save    = data.get("save", True)

        if not text:
            return jsonify({"error": "Field 'text' is required and cannot be empty", "status": 400}), 400

        if len(text) > 50_000:
            return jsonify({"error": "Text exceeds 50,000 character limit", "status": 413}), 413

        # ── Log API request ───────────────────────────────────────────────────
        audit.log(
            EventType.API_REQUEST,
            description=f"POST /api/deidentify | note_id={note_id} | chars={len(text)}",
            note_id=note_id,
        )

        # ── Pre-NER clean ─────────────────────────────────────────────────────
        pre_result = cleaner.clean_pre_ner(text)

        # ── NER + masking ─────────────────────────────────────────────────────
        ner_result = pipeline.process_note(
            pre_result.cleaned_text,
            note_id=note_id,
            save_to_db=save,
        )

        # ── Post-NER clean + validation ───────────────────────────────────────
        post_result = cleaner.clean_post_ner(ner_result["masked_text"])

        # ── Log NER result ────────────────────────────────────────────────────
        audit.log_ner_result(ner_result)
        audit.log_cleaning_result(post_result, note_id=note_id)

        # ── Build response ────────────────────────────────────────────────────
        response = {
            **ner_result,
            "masked_text": post_result.cleaned_text,
            "is_valid":    post_result.is_valid,
            "changes":     pre_result.changes + post_result.changes,
        }

        audit.log(
            EventType.API_RESPONSE,
            description=f"POST /api/deidentify | {ner_result['entity_count']} entities",
            note_id=note_id,
        )

        return jsonify(response), 200

    @app.route("/api/note/<int:note_id>")
    def get_note(note_id: int):
        """Fetch a processed note by note_id."""
        loader: DataLoader = app.config["LOADER"]
        try:
            df = loader.sql_query(
                f"SELECT * FROM processed_notes WHERE note_id = {note_id} "
                f"ORDER BY id DESC LIMIT 1"
            )
            if df.empty:
                return jsonify({"error": f"Note {note_id} not found", "status": 404}), 404
            return jsonify(df.iloc[0].to_dict()), 200
        except Exception as e:
            logger.error("get_note error: %s", e)
            return jsonify({"error": str(e), "status": 500}), 500

    @app.route("/api/stats")
    def stats():
        """
        Return corpus + pipeline statistics as JSON.
        Consumed by the /dashboard template to render charts.
        """
        loader: DataLoader = app.config["LOADER"]
        audit:  AuditLogger = app.config["AUDIT"]

        try:
            # Note counts — return 0 gracefully if tables don't exist yet
            try:
                note_count = loader.sql_query(
                    "SELECT COUNT(*) as n FROM clinical_notes"
                ).iloc[0]["n"]
            except Exception:
                note_count = 0

            try:
                processed_count = loader.sql_query(
                    "SELECT COUNT(*) as n FROM processed_notes"
                ).iloc[0]["n"]
            except Exception:
                processed_count = 0

            # Entity breakdown
            entity_totals: dict = {}
            try:
                entity_sql = loader.sql_query(
                    "SELECT entity_types_json FROM processed_notes "
                    "WHERE entity_types_json IS NOT NULL"
                )
                for row in entity_sql["entity_types_json"]:
                    try:
                        for k, v in json.loads(row).items():
                            entity_totals[k] = entity_totals.get(k, 0) + v
                    except Exception:
                        pass
            except Exception:
                pass

            # Specialty breakdown
            try:
                specialty = loader.sql_query(
                    "SELECT medical_specialty, COUNT(*) as count "
                    "FROM clinical_notes GROUP BY medical_specialty "
                    "ORDER BY count DESC LIMIT 10"
                ).to_dict(orient="records")
            except Exception:
                specialty = []

            # Avg entities per specialty
            try:
                phi_by_spec = loader.sql_query(
                    "SELECT cn.medical_specialty, "
                    "ROUND(AVG(pn.entity_count), 1) as avg_phi "
                    "FROM clinical_notes cn "
                    "JOIN processed_notes pn ON cn.note_id = pn.note_id "
                    "GROUP BY cn.medical_specialty ORDER BY avg_phi DESC LIMIT 8"
                ).to_dict(orient="records")
            except Exception:
                phi_by_spec = []

            # Audit summary
            audit_summary = audit.get_summary().to_dict(orient="records")

            return jsonify({
                "note_count":       int(note_count),
                "processed_count":  int(processed_count),
                "entity_totals":    entity_totals,
                "specialty":        specialty,
                "phi_by_specialty": phi_by_spec,
                "audit_summary":    audit_summary,
                "total_audit_events": audit.total_events(),
            }), 200

        except Exception as e:
            logger.error("stats error: %s", e)
            return jsonify({"error": str(e), "status": 500}), 500


# ── UI routes ─────────────────────────────────────────────────────────────────

def _register_ui_routes(app: Flask) -> None:

    @app.route("/dashboard")
    def dashboard():
        """
        Live EDA dashboard — renders stats as interactive charts.
        Uses Chart.js loaded from CDN, data fetched from /api/stats.
        """
        return render_template_string(DASHBOARD_TEMPLATE)

    @app.route("/report/<int:note_id>")
    def report(note_id: int):
        """Before/after de-identification diff view for a single note."""
        loader: DataLoader = app.config["LOADER"]
        try:
            try:
                orig_df = loader.sql_query(
                    f"SELECT * FROM clinical_notes WHERE note_id = {note_id}"
                )
            except Exception:
                return f"<h3>Note {note_id} not found</h3>", 404
            proc_df = loader.sql_query(
                f"SELECT * FROM processed_notes WHERE note_id = {note_id} "
                f"ORDER BY id DESC LIMIT 1"
            )

            if orig_df is None or orig_df.empty:
                return f"<h3>Note {note_id} not found</h3>", 404

            orig_text = orig_df.iloc[0].get("transcription", "")
            proc_text = proc_df.iloc[0].get("masked_text", "Not yet processed") \
                if not proc_df.empty else "Not yet processed"
            specialty = orig_df.iloc[0].get("medical_specialty", "Unknown")
            entity_count = proc_df.iloc[0].get("entity_count", 0) \
                if not proc_df.empty else 0

            return render_template_string(
                REPORT_TEMPLATE,
                note_id=note_id,
                specialty=specialty,
                orig_text=orig_text,
                proc_text=proc_text,
                entity_count=entity_count,
            )
        except Exception as e:
            return f"<h3>Error: {e}</h3>", 500


# ── HTML Templates ────────────────────────────────────────────────────────────
# Inline templates — no separate file needed for a portfolio project.
# In production these would be Jinja2 .html files in templates/.

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ClinicalNER Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e2e8f0; min-height: 100vh; }
    header { background: #1a1d2e; border-bottom: 1px solid #2d3748;
             padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
    header h1 { font-size: 1.4rem; font-weight: 600; color: #63b3ed; }
    header span { font-size: 0.85rem; color: #718096; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem; padding: 1.5rem 2rem 0; }
    .stat-card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 10px;
                 padding: 1.2rem; }
    .stat-card .label { font-size: 0.75rem; color: #718096; text-transform: uppercase;
                        letter-spacing: .05em; margin-bottom: .4rem; }
    .stat-card .value { font-size: 2rem; font-weight: 700; color: #63b3ed; }
    .stat-card .sub   { font-size: 0.8rem; color: #4a5568; margin-top: .2rem; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
              padding: 1.5rem 2rem; }
    .chart-card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 10px;
                  padding: 1.2rem; }
    .chart-card h3 { font-size: 0.85rem; color: #a0aec0; margin-bottom: 1rem;
                     text-transform: uppercase; letter-spacing: .05em; }
    canvas { max-height: 280px; }
    .audit-table { margin: 0 2rem 2rem; background: #1a1d2e;
                   border: 1px solid #2d3748; border-radius: 10px; overflow: hidden; }
    .audit-table h3 { padding: 1rem 1.2rem; font-size: 0.85rem; color: #a0aec0;
                      text-transform: uppercase; letter-spacing: .05em;
                      border-bottom: 1px solid #2d3748; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { padding: .6rem 1.2rem; text-align: left; color: #4a5568;
         border-bottom: 1px solid #2d3748; font-weight: 500; }
    td { padding: .6rem 1.2rem; border-bottom: 1px solid #1e2436; }
    tr:last-child td { border-bottom: none; }
    .badge { display: inline-block; padding: .15rem .5rem; border-radius: 4px;
             font-size: 0.75rem; font-weight: 600; }
    .badge-blue   { background: #1e3a5f; color: #63b3ed; }
    .badge-green  { background: #1a3a2a; color: #68d391; }
    .badge-yellow { background: #3a2e0e; color: #f6e05e; }
    .badge-red    { background: #3a1a1a; color: #fc8181; }
    .error { color: #fc8181; padding: 2rem; text-align: center; }
    @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>ClinicalNER</h1>
    <span>De-identification pipeline dashboard</span>
  </header>

  <div class="grid" id="stat-cards">
    <div class="stat-card"><div class="label">Total notes</div>
      <div class="value" id="s-notes">—</div></div>
    <div class="stat-card"><div class="label">Processed</div>
      <div class="value" id="s-processed">—</div></div>
    <div class="stat-card"><div class="label">PHI entities found</div>
      <div class="value" id="s-entities">—</div></div>
    <div class="stat-card"><div class="label">Audit events</div>
      <div class="value" id="s-audit">—</div></div>
  </div>

  <div class="charts">
    <div class="chart-card">
      <h3>PHI entity breakdown</h3>
      <canvas id="entityChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Notes by specialty</h3>
      <canvas id="specialtyChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Avg PHI entities per specialty</h3>
      <canvas id="phiSpecChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Audit event types</h3>
      <canvas id="auditChart"></canvas>
    </div>
  </div>

  <div class="audit-table">
    <h3>Audit event log</h3>
    <table>
      <thead>
        <tr><th>Event type</th><th>Count</th><th>First seen</th><th>Last seen</th></tr>
      </thead>
      <tbody id="audit-tbody"></tbody>
    </table>
  </div>

<script>
const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7','#fbb6ce','#9ae6b4'];

function badge(type) {
  const map = {
    NER_COMPLETED:'blue', DATA_CLEANED_POST:'green', PIPELINE_COMPLETE:'green',
    RESIDUAL_PHI_FOUND:'red', API_REQUEST:'yellow', API_RESPONSE:'yellow',
    DATA_CLEANED_PRE:'blue', PIPELINE_START:'yellow', DATA_INGESTED:'blue'
  };
  const cls = map[type] || 'blue';
  return `<span class="badge badge-${cls}">${type}</span>`;
}

function makeChart(id, type, labels, data, opts={}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type,
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: type === 'doughnut' ? COLORS : COLORS[0],
        borderColor: type === 'bar' ? COLORS[0] : COLORS,
        borderWidth: 1,
        borderRadius: type === 'bar' ? 4 : 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: type === 'doughnut',
                            labels: { color: '#a0aec0', font: { size: 11 } } } },
      scales: type === 'bar' ? {
        x: { ticks: { color: '#718096', font: { size: 10 } }, grid: { color: '#1e2436' } },
        y: { ticks: { color: '#718096', font: { size: 10 } }, grid: { color: '#1e2436' } }
      } : {},
      ...opts
    }
  });
}

async function loadDashboard() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();

    document.getElementById('s-notes').textContent     = data.note_count.toLocaleString();
    document.getElementById('s-processed').textContent = data.processed_count.toLocaleString();
    document.getElementById('s-audit').textContent     = data.total_audit_events.toLocaleString();

    const totalEntities = Object.values(data.entity_totals).reduce((a,b)=>a+b, 0);
    document.getElementById('s-entities').textContent  = totalEntities.toLocaleString();

    makeChart('entityChart', 'doughnut',
      Object.keys(data.entity_totals), Object.values(data.entity_totals));

    makeChart('specialtyChart', 'bar',
      data.specialty.map(d=>d.medical_specialty),
      data.specialty.map(d=>d.count));

    makeChart('phiSpecChart', 'bar',
      data.phi_by_specialty.map(d=>d.medical_specialty),
      data.phi_by_specialty.map(d=>d.avg_phi));

    makeChart('auditChart', 'doughnut',
      data.audit_summary.map(d=>d.event_type),
      data.audit_summary.map(d=>d.count));

    const tbody = document.getElementById('audit-tbody');
    tbody.innerHTML = data.audit_summary.map(r => `
      <tr>
        <td>${badge(r.event_type)}</td>
        <td><strong>${r.count}</strong></td>
        <td style="color:#718096;font-size:.8rem">${r.first_seen ? r.first_seen.slice(0,19).replace('T',' ') : '—'}</td>
        <td style="color:#718096;font-size:.8rem">${r.last_seen  ? r.last_seen.slice(0,19).replace('T',' ')  : '—'}</td>
      </tr>`).join('');
  } catch(e) {
    document.body.innerHTML += `<div class="error">Failed to load stats: ${e.message}</div>`;
  }
}

loadDashboard();
setInterval(loadDashboard, 30000);   // auto-refresh every 30s
</script>
</body>
</html>
"""

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Note {{ note_id }} — De-identification Report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e2e8f0; padding: 2rem; }
    .header { margin-bottom: 1.5rem; }
    .header h1 { font-size: 1.3rem; color: #63b3ed; margin-bottom: .3rem; }
    .meta { font-size: 0.85rem; color: #718096; }
    .meta span { margin-right: 1.5rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .panel { background: #1a1d2e; border: 1px solid #2d3748;
             border-radius: 10px; overflow: hidden; }
    .panel-header { padding: .8rem 1.2rem; font-size: .8rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: .05em;
                    border-bottom: 1px solid #2d3748; }
    .panel-header.orig  { color: #f6ad55; background: #1e2010; }
    .panel-header.masked { color: #68d391; background: #0e1e18; }
    .panel-body { padding: 1.2rem; font-family: 'Courier New', monospace;
                  font-size: 0.82rem; line-height: 1.7; white-space: pre-wrap;
                  word-break: break-word; max-height: 500px; overflow-y: auto; }
    .masked-token { background: #1e3a5f; color: #63b3ed; border-radius: 3px;
                    padding: 0 3px; font-weight: 600; }
    .back { display: inline-block; margin-bottom: 1rem; color: #63b3ed;
            text-decoration: none; font-size: .85rem; }
    .back:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <a href="/dashboard" class="back">← Back to dashboard</a>
  <div class="header">
    <h1>De-identification report — note {{ note_id }}</h1>
    <div class="meta">
      <span>Specialty: <strong>{{ specialty }}</strong></span>
      <span>PHI entities masked: <strong>{{ entity_count }}</strong></span>
    </div>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="panel-header orig">Original (contains PHI)</div>
      <div class="panel-body">{{ orig_text }}</div>
    </div>
    <div class="panel">
      <div class="panel-header masked">De-identified output</div>
      <div class="panel-body" id="masked-body"></div>
    </div>
  </div>
<script>
  const raw = {{ proc_text | tojson }};
  const highlighted = raw.replace(new RegExp('\\[([A-Z]+)\\]', 'g'),
    '<span class="masked-token">[$1]</span>');
  document.getElementById('masked-body').innerHTML = highlighted;
</script>
</body>
</html>
"""