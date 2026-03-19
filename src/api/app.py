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
from src.pipeline.anomaly_detector import AnomalyDetector

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
    app.config["DETECTOR"] = AnomalyDetector(contamination=0.05)

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
            "masked_text":    post_result.cleaned_text,
            "is_valid":       post_result.is_valid,
            "changes":        pre_result.changes + post_result.changes,
            "avg_confidence": ner_result.get("avg_confidence", 0.0),
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


    @app.route("/api/anomaly-scan", methods=["POST"])
    def anomaly_scan():
        """Fit IsolationForest on submitted notes, return anomaly scores."""
        detector: AnomalyDetector = app.config["DETECTOR"]
        audit:    AuditLogger     = app.config["AUDIT"]

        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "status": 400}), 400

        notes = request.get_json().get("notes", [])
        if len(notes) < 10:
            return jsonify({
                "error": "Need at least 10 notes to fit the anomaly model",
                "status": 400,
            }), 400

        try:
            results = detector.fit_predict(notes)
            summary = detector.summary(results)
            audit.log(
                EventType.PIPELINE_COMPLETE,
                description=f"Anomaly scan: {summary['anomalies_found']} flagged / {summary['total_notes']}",
                metadata=summary,
            )
            return jsonify({**summary, "results": [r.to_dict() for r in results]}), 200
        except Exception as e:
            logger.error("anomaly_scan error: %s", e)
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

    @app.route("/api-explorer")
    def api_explorer():
        """Interactive API explorer page."""
        return render_template_string(API_EXPLORER_TEMPLATE)


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

# ── HTML Templates ────────────────────────────────────────────────────────────

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClinicalNER · Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:       #060B14;
  --bg2:      #0C1524;
  --bg3:      #111D2E;
  --border:   rgba(0,229,180,0.12);
  --border2:  rgba(0,229,180,0.22);
  --teal:     #00E5B4;
  --teal2:    #00C49A;
  --amber:    #FFB547;
  --red:      #FF5C6A;
  --blue:     #4F8EF7;
  --txt:      #E8F0F7;
  --txt2:     #7A9AB8;
  --txt3:     #3D5A72;
  --glow:     0 0 40px rgba(0,229,180,0.12);
  --card-bg:  rgba(12,21,36,0.85);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Instrument Sans', sans-serif;
  background: var(--bg);
  color: var(--txt);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Animated grid background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,229,180,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,180,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

body::after {
  content: '';
  position: fixed;
  top: -40%;
  right: -20%;
  width: 80vw;
  height: 80vh;
  background: radial-gradient(ellipse, rgba(0,229,180,0.05) 0%, transparent 65%);
  pointer-events: none;
  z-index: 0;
}

/* ── Layout ── */
.layout { display: flex; min-height: 100vh; position: relative; z-index: 1; }

/* ── Sidebar ── */
.sidebar {
  width: 220px;
  flex-shrink: 0;
  background: rgba(6,11,20,0.9);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 28px 0;
  position: sticky;
  top: 0;
  height: 100vh;
  backdrop-filter: blur(12px);
}

.sidebar-logo {
  padding: 0 24px 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.logo-mark {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--teal);
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-dot {
  width: 8px; height: 8px;
  background: var(--teal);
  border-radius: 50%;
  box-shadow: 0 0 12px var(--teal);
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%,100% { opacity: 1; transform: scale(1); box-shadow: 0 0 12px var(--teal); }
  50%      { opacity: 0.6; transform: scale(0.8); box-shadow: 0 0 4px var(--teal); }
}

.logo-sub {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--txt3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 4px;
}

.nav-section {
  padding: 0 12px;
  flex: 1;
}

.nav-label {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--txt3);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 0 12px;
  margin-bottom: 6px;
  margin-top: 16px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: var(--txt2);
  transition: all 0.18s;
  margin-bottom: 2px;
  text-decoration: none;
}
.nav-item:hover { background: rgba(0,229,180,0.07); color: var(--teal); }
.nav-item.active { background: rgba(0,229,180,0.1); color: var(--teal); }
.nav-icon { width: 16px; opacity: 0.7; }

.sidebar-footer {
  padding: 20px 24px 0;
  border-top: 1px solid var(--border);
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--teal);
  letter-spacing: 0.05em;
}

.status-dot {
  width: 6px; height: 6px;
  background: var(--teal);
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}

/* ── Main content ── */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

/* ── Topbar ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: rgba(6,11,20,0.6);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 10;
}

.topbar-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--txt);
}

.topbar-path {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--txt3);
  letter-spacing: 0.08em;
}

.topbar-right { display: flex; align-items: center; gap: 12px; }

.tag {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.1em;
  padding: 4px 10px;
  border-radius: 20px;
  text-transform: uppercase;
}

.tag-teal { background: rgba(0,229,180,0.1); color: var(--teal); border: 1px solid rgba(0,229,180,0.2); }
.tag-amber { background: rgba(255,181,71,0.1); color: var(--amber); border: 1px solid rgba(255,181,71,0.2); }

/* ── Content ── */
.content { padding: 28px 32px; flex: 1; }

/* ── Section header ── */
.section-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 20px;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--txt3);
}

.section-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Stat cards ── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.stat-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
  backdrop-filter: blur(8px);
  opacity: 0;
  transform: translateY(12px);
  animation: card-in 0.5s ease forwards;
}

.stat-card:nth-child(1) { animation-delay: 0.05s; }
.stat-card:nth-child(2) { animation-delay: 0.12s; }
.stat-card:nth-child(3) { animation-delay: 0.19s; }
.stat-card:nth-child(4) { animation-delay: 0.26s; }

@keyframes card-in {
  to { opacity: 1; transform: translateY(0); }
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}

.stat-card:hover { border-color: var(--border2); transform: translateY(-1px); }
.stat-card:hover::before { opacity: 1; }

.stat-label {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--txt3);
  margin-bottom: 10px;
}

.stat-value {
  font-family: 'Syne', sans-serif;
  font-size: 32px;
  font-weight: 800;
  color: var(--txt);
  line-height: 1;
  letter-spacing: -1px;
}

.stat-value.teal  { color: var(--teal); }
.stat-value.amber { color: var(--amber); }
.stat-value.blue  { color: var(--blue); }

.stat-delta {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--txt3);
  margin-top: 6px;
}

.stat-icon {
  position: absolute;
  top: 18px; right: 18px;
  font-size: 18px;
  opacity: 0.15;
}

/* ── Charts grid ── */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 28px;
}

.chart-wide { grid-column: span 2; }

.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 24px;
  backdrop-filter: blur(8px);
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.35s;
}

.chart-card:nth-child(2) { animation-delay: 0.42s; }
.chart-card:nth-child(3) { animation-delay: 0.49s; }
.chart-card:nth-child(4) { animation-delay: 0.56s; }

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}

.chart-title {
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  color: var(--txt2);
  letter-spacing: 0.05em;
}

.chart-badge {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--txt3);
  letter-spacing: 0.1em;
}

canvas { max-height: 220px; }

/* ── Audit table ── */
.audit-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  backdrop-filter: blur(8px);
  margin-bottom: 16px;
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.6s;
}

.audit-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 22px;
  border-bottom: 1px solid var(--border);
}

.audit-title {
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  color: var(--txt2);
}

table { width: 100%; border-collapse: collapse; }

thead th {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--txt3);
  padding: 10px 22px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

tbody td {
  padding: 11px 22px;
  font-size: 12px;
  color: var(--txt2);
  border-bottom: 1px solid rgba(0,229,180,0.05);
  font-family: 'Instrument Sans', sans-serif;
}

tbody tr:last-child td { border-bottom: none; }
tbody tr { transition: background 0.15s; }
tbody tr:hover td { background: rgba(0,229,180,0.03); color: var(--txt); }

/* ── Event badges ── */
.ev {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.08em;
  padding: 3px 9px;
  border-radius: 20px;
  font-weight: 500;
}

.ev-teal   { background: rgba(0,229,180,0.08); color: var(--teal);  border: 1px solid rgba(0,229,180,0.18); }
.ev-amber  { background: rgba(255,181,71,0.08); color: var(--amber); border: 1px solid rgba(255,181,71,0.18); }
.ev-red    { background: rgba(255,92,106,0.08); color: var(--red);   border: 1px solid rgba(255,92,106,0.18); }
.ev-blue   { background: rgba(79,142,247,0.08); color: var(--blue);  border: 1px solid rgba(79,142,247,0.18); }

.ev-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; }

/* ── Count col ── */
.count {
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: var(--txt);
}

.ts {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--txt3);
}

/* ── Loading shimmer ── */
.shimmer {
  background: linear-gradient(90deg, var(--bg3) 25%, rgba(0,229,180,0.06) 50%, var(--bg3) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 4px;
  height: 20px;
  width: 80px;
}
@keyframes shimmer { to { background-position: -200% 0; } }

/* ── Error ── */
.error-banner {
  background: rgba(255,92,106,0.08);
  border: 1px solid rgba(255,92,106,0.2);
  border-radius: 8px;
  padding: 12px 18px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--red);
  margin-bottom: 16px;
}

@media (max-width: 1000px) {
  .sidebar { display: none; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .charts-grid { grid-template-columns: 1fr; }
  .chart-wide { grid-column: span 1; }
}
</style>
</head>
<body>
<div class="layout">

  <!-- ── Sidebar ── -->
  <nav class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-mark">
        <div class="logo-dot"></div>
        ClinicalNER
      </div>
      <div class="logo-sub">De-identification Platform</div>
    </div>
    <div class="nav-section">
      <div class="nav-label">Analytics</div>
      <a href="/dashboard" class="nav-item active">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="1" y="1" width="6" height="6" rx="1.5"/>
          <rect x="9" y="1" width="6" height="6" rx="1.5"/>
          <rect x="1" y="9" width="6" height="6" rx="1.5"/>
          <rect x="9" y="9" width="6" height="6" rx="1.5"/>
        </svg>
        Dashboard
      </a>
      <a href="/api/stats" class="nav-item" target="_blank">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M2 12 L5 8 L8 10 L11 5 L14 7"/>
        </svg>
        Live Stats JSON
      </a>
      <div class="nav-label">Tools</div>
      <a href="/api-explorer" class="nav-item">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/>
        </svg>
        API Explorer
      </a>
      <a href="/health" class="nav-item" target="_blank">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M8 2 L8 14 M2 8 L14 8" stroke-linecap="round"/>
        </svg>
        Health Probe
      </a>
    </div>
    <div class="sidebar-footer">
      <div class="status-badge">
        <div class="status-dot"></div>
        System Online
      </div>
    </div>
  </nav>

  <!-- ── Main ── -->
  <div class="main">

    <!-- Topbar -->
    <div class="topbar">
      <div>
        <div class="topbar-title">Pipeline Intelligence</div>
        <div class="topbar-path">ClinicalNER / dashboard</div>
      </div>
      <div class="topbar-right">
        <span class="tag tag-teal">HIPAA-aware</span>
        <span class="tag tag-amber">spaCy + Regex NER</span>
      </div>
    </div>

    <!-- Content -->
    <div class="content">
      <div id="error-container"></div>

      <!-- Stats -->
      <div class="section-header">
        <span class="section-title">Corpus Overview</span>
        <div class="section-line"></div>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Clinical Notes</div>
          <div class="stat-value teal" id="s-notes"><div class="shimmer"></div></div>
          <div class="stat-delta" id="s-notes-sub">Loading…</div>
          <div class="stat-icon">📄</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">De-identified</div>
          <div class="stat-value" id="s-processed"><div class="shimmer"></div></div>
          <div class="stat-delta" id="s-proc-sub">Loading…</div>
          <div class="stat-icon">🔒</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">PHI Entities Found</div>
          <div class="stat-value amber" id="s-entities"><div class="shimmer"></div></div>
          <div class="stat-delta">across all processed notes</div>
          <div class="stat-icon">🔍</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Audit Events</div>
          <div class="stat-value blue" id="s-audit"><div class="shimmer"></div></div>
          <div class="stat-delta">append-only trail</div>
          <div class="stat-icon">📋</div>
        </div>
      </div>

      <!-- Charts -->
      <div class="section-header">
        <span class="section-title">PHI Distribution</span>
        <div class="section-line"></div>
      </div>

      <div class="charts-grid">
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title">Entity type breakdown</div>
            <div class="chart-badge">DONUT</div>
          </div>
          <canvas id="entityChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title">Notes by specialty</div>
            <div class="chart-badge">BAR</div>
          </div>
          <canvas id="specialtyChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title">Avg PHI density / specialty</div>
            <div class="chart-badge">HORIZONTAL BAR</div>
          </div>
          <canvas id="phiSpecChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title">Audit event distribution</div>
            <div class="chart-badge">DONUT</div>
          </div>
          <canvas id="auditChart"></canvas>
        </div>
      </div>

      <!-- Audit log -->
      <div class="section-header">
        <span class="section-title">Audit Trail</span>
        <div class="section-line"></div>
      </div>

      <div class="audit-card">
        <div class="audit-header">
          <div class="audit-title">Event log</div>
          <span class="tag tag-teal">Append-only · ICH E6 compliant</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Event</th>
              <th>Count</th>
              <th>First seen</th>
              <th>Last seen</th>
            </tr>
          </thead>
          <tbody id="audit-tbody">
            <tr><td colspan="4" style="text-align:center;padding:24px;font-family:'DM Mono',monospace;font-size:11px;color:var(--txt3)">Loading audit events…</td></tr>
          </tbody>
        </table>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /layout -->

<script>
// ── Chart defaults ─────────────────────────────────────────────────────
Chart.defaults.color = '#7A9AB8';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 10;

const PALETTE = ['#00E5B4','#4F8EF7','#FFB547','#FF5C6A','#B794F4','#76E4F7','#FBB6CE','#9AE6B4','#F6AD55'];

const GRID = { color: 'rgba(0,229,180,0.05)', drawBorder: false };

const _charts = {};

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

function donut(id, labels, values) {
  destroyChart(id);
  _charts[id] = new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: PALETTE, borderColor: '#060B14', borderWidth: 3, hoverBorderWidth: 0 }]
    },
    options: {
      cutout: '68%',
      plugins: {
        legend: {
          position: 'right',
          labels: { boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: 'circle', padding: 14, font: { size: 10 } }
        }
      }
    }
  });
}

function bar(id, labels, values, horizontal) {
  destroyChart(id);
  const axis = horizontal ? 'y' : 'x';
  _charts[id] = new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(0,229,180,0.15)',
        borderColor: '#00E5B4',
        borderWidth: 1,
        borderRadius: 4,
        hoverBackgroundColor: 'rgba(0,229,180,0.28)',
      }]
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      plugins: { legend: { display: false } },
      scales: {
        [axis]: { grid: GRID, ticks: { font: { size: 9 } } },
        [axis === 'x' ? 'y' : 'x']: { grid: GRID, ticks: { font: { size: 9 } } }
      }
    }
  });
}

// ── Animated counter ───────────────────────────────────────────────────
function animateCount(el, target) {
  const start = Date.now();
  const duration = 1200;
  const fmt = n => n >= 1000 ? (n/1000).toFixed(1) + 'k' : Math.round(n).toString();
  function step() {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = fmt(target * ease);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = fmt(target);
  }
  requestAnimationFrame(step);
}

// ── Event badge ────────────────────────────────────────────────────────
function evBadge(type) {
  const map = {
    NER_COMPLETED: 'teal', DATA_CLEANED_POST: 'teal', PIPELINE_COMPLETE: 'blue',
    RESIDUAL_PHI_FOUND: 'red', API_REQUEST: 'amber', API_RESPONSE: 'amber',
    DATA_CLEANED_PRE: 'teal', PIPELINE_START: 'blue', DATA_INGESTED: 'blue', ERROR: 'red'
  };
  const cls = map[type] || 'blue';
  return `<span class="ev ev-${cls}"><span class="ev-dot"></span>${type}</span>`;
}

// ── Load dashboard ─────────────────────────────────────────────────────
async function load() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Stat counters
    animateCount(document.getElementById('s-notes'),     data.note_count);
    animateCount(document.getElementById('s-processed'), data.processed_count);
    animateCount(document.getElementById('s-entities'),  Object.values(data.entity_totals).reduce((a,b)=>a+b,0));
    animateCount(document.getElementById('s-audit'),     data.total_audit_events);

    const pct = data.note_count > 0
      ? ((data.processed_count / data.note_count) * 100).toFixed(1) + '% coverage'
      : '—';
    document.getElementById('s-notes-sub').textContent  = data.specialty?.length + ' specialties';
    document.getElementById('s-proc-sub').textContent   = pct;

    // Charts
    donut('entityChart',
      Object.keys(data.entity_totals),
      Object.values(data.entity_totals));

    bar('specialtyChart',
      data.specialty.map(d => d.medical_specialty.slice(0,14)),
      data.specialty.map(d => d.count), false);

    bar('phiSpecChart',
      data.phi_by_specialty.map(d => d.medical_specialty.slice(0,14)),
      data.phi_by_specialty.map(d => d.avg_phi), true);

    donut('auditChart',
      data.audit_summary.map(d => d.event_type),
      data.audit_summary.map(d => d.count));

    // Audit table
    document.getElementById('audit-tbody').innerHTML =
      data.audit_summary.map(r => `
        <tr>
          <td>${evBadge(r.event_type)}</td>
          <td><span class="count">${r.count.toLocaleString()}</span></td>
          <td><span class="ts">${r.first_seen ? r.first_seen.slice(0,19).replace('T',' ') : '—'}</span></td>
          <td><span class="ts">${r.last_seen  ? r.last_seen.slice(0,19).replace('T',' ')  : '—'}</span></td>
        </tr>`).join('');

  } catch(e) {
    document.getElementById('error-container').innerHTML =
      `<div class="error-banner">⚠ Failed to load pipeline data: ${e.message}</div>`;
  }
}

load();
setInterval(load, 30000);
</script>
</body>
</html>"""

REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Note {{ note_id }} · De-identification Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:    #060B14;
  --bg2:   #0C1524;
  --bg3:   #111D2E;
  --border: rgba(0,229,180,0.12);
  --teal:  #00E5B4;
  --amber: #FFB547;
  --red:   #FF5C6A;
  --txt:   #E8F0F7;
  --txt2:  #7A9AB8;
  --txt3:  #3D5A72;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Instrument Sans', sans-serif;
  background: var(--bg);
  color: var(--txt);
  min-height: 100vh;
  padding: 0;
}
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(0,229,180,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,180,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}
.page { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 32px; }

.breadcrumb {
  display: flex; align-items: center; gap: 8px;
  font-family: 'DM Mono', monospace; font-size: 10px;
  color: var(--txt3); margin-bottom: 28px; letter-spacing: 0.08em;
}
.breadcrumb a { color: var(--teal); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.breadcrumb-sep { opacity: 0.4; }

.report-header {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 28px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}

.report-title {
  font-family: 'Syne', sans-serif;
  font-size: 24px; font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--txt);
}

.report-title span { color: var(--teal); }

.report-meta {
  display: flex; gap: 20px; margin-top: 10px; flex-wrap: wrap;
}

.meta-item {
  font-family: 'DM Mono', monospace;
  font-size: 10px; letter-spacing: 0.1em;
  color: var(--txt3); text-transform: uppercase;
}
.meta-item strong { color: var(--txt2); font-weight: 500; margin-left: 6px; }

.header-right { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start; }

.badge {
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 0.1em;
  padding: 5px 12px; border-radius: 20px; text-transform: uppercase;
}
.badge-teal  { background: rgba(0,229,180,0.1); color: var(--teal);  border: 1px solid rgba(0,229,180,0.2); }
.badge-amber { background: rgba(255,181,71,0.1); color: var(--amber); border: 1px solid rgba(255,181,71,0.2); }

/* Diff panels */
.diff-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

.diff-panel {
  background: rgba(12,21,36,0.85);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  backdrop-filter: blur(8px);
}

.panel-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
}

.panel-head-title {
  font-family: 'DM Mono', monospace;
  font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase;
  display: flex; align-items: center; gap: 8px;
}

.panel-dot {
  width: 6px; height: 6px; border-radius: 50%;
}

.panel-orig  .panel-dot { background: var(--amber); }
.panel-masked .panel-dot { background: var(--teal);  }
.panel-orig  .panel-head { border-bottom-color: rgba(255,181,71,0.12); }
.panel-masked .panel-head { border-bottom-color: rgba(0,229,180,0.12); }
.panel-orig  .panel-head-title { color: var(--amber); }
.panel-masked .panel-head-title { color: var(--teal);  }

.panel-body {
  padding: 18px;
  font-family: 'DM Mono', monospace;
  font-size: 12px; line-height: 1.8;
  white-space: pre-wrap; word-break: break-word;
  max-height: 460px; overflow-y: auto;
  color: var(--txt2);
}

.panel-body::-webkit-scrollbar { width: 4px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.masked-token {
  background: rgba(0,229,180,0.1);
  color: var(--teal);
  border: 1px solid rgba(0,229,180,0.22);
  border-radius: 4px;
  padding: 1px 5px;
  font-weight: 500;
  font-size: 11px;
}

/* Stats row */
.stats-row {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  margin-bottom: 0;
}

.mini-card {
  background: rgba(12,21,36,0.85);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
  backdrop-filter: blur(8px);
}
.mini-label {
  font-family: 'DM Mono', monospace; font-size: 8px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--txt3); margin-bottom: 6px;
}
.mini-value {
  font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
  color: var(--teal);
}
.mini-value.white { color: var(--txt); }
.mini-value.amber { color: var(--amber); }

@media (max-width: 800px) {
  .diff-grid { grid-template-columns: 1fr; }
  .stats-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="page">

  <div class="breadcrumb">
    <a href="/dashboard">← Dashboard</a>
    <span class="breadcrumb-sep">/</span>
    <span>Report</span>
    <span class="breadcrumb-sep">/</span>
    <span>Note {{ note_id }}</span>
  </div>

  <div class="report-header">
    <div>
      <div class="report-title">De-identification <span>#{{ note_id }}</span></div>
      <div class="report-meta">
        <div class="meta-item">Specialty <strong>{{ specialty }}</strong></div>
        <div class="meta-item">PHI Masked <strong>{{ entity_count }}</strong></div>
        <div class="meta-item">Status <strong style="color:var(--teal)">Processed</strong></div>
      </div>
    </div>
    <div class="header-right">
      <span class="badge badge-teal">De-identified</span>
      <span class="badge badge-amber">{{ specialty }}</span>
    </div>
  </div>

  <div class="diff-grid">
    <div class="diff-panel panel-orig">
      <div class="panel-head">
        <div class="panel-head-title">
          <div class="panel-dot"></div>
          Original — contains PHI
        </div>
      </div>
      <div class="panel-body">{{ orig_text }}</div>
    </div>

    <div class="diff-panel panel-masked">
      <div class="panel-head">
        <div class="panel-head-title">
          <div class="panel-dot"></div>
          De-identified output
        </div>
      </div>
      <div class="panel-body" id="masked-body"></div>
    </div>
  </div>

  <div class="stats-row">
    <div class="mini-card">
      <div class="mini-label">PHI entities masked</div>
      <div class="mini-value">{{ entity_count }}</div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Specialty</div>
      <div class="mini-value white" style="font-size:14px;margin-top:4px">{{ specialty }}</div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Compliance</div>
      <div class="mini-value" style="font-size:13px;margin-top:4px">HIPAA-aware</div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Pipeline</div>
      <div class="mini-value amber" style="font-size:13px;margin-top:4px">spaCy + Regex</div>
    </div>
  </div>

</div>

<script>
const raw = {{ proc_text | tojson }};
const highlighted = raw.replace(/(\\[[A-Z]+\\])/g,
  '<span class="masked-token">$1</span>');
document.getElementById('masked-body').innerHTML = highlighted;
</script>
</body>
</html>"""

API_EXPLORER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Explorer · ClinicalNER</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#060B14;--bg2:#0C1524;--bg3:#111D2E;
  --border:rgba(0,229,180,0.12);--border2:rgba(0,229,180,0.22);
  --teal:#00E5B4;--amber:#FFB547;--red:#FF5C6A;--blue:#4F8EF7;
  --txt:#E8F0F7;--txt2:#7A9AB8;--txt3:#3D5A72;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,229,180,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,180,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:220px;flex-shrink:0;background:rgba(6,11,20,0.9);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;backdrop-filter:blur(12px);}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 12px var(--teal);animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;margin-top:4px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:9px;color:var(--txt3);letter-spacing:.14em;text-transform:uppercase;padding:0 12px;margin-bottom:6px;margin-top:16px;}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:500;color:var(--txt2);transition:all .18s;margin-bottom:2px;text-decoration:none;}
.nav-item:hover{background:rgba(0,229,180,0.07);color:var(--teal);}
.nav-item.active{background:rgba(0,229,180,0.1);color:var(--teal);}
.nav-icon{width:16px;opacity:.7;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:10px;color:var(--teal);letter-spacing:.05em;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:18px 32px;border-bottom:1px solid var(--border);background:rgba(6,11,20,0.6);backdrop-filter:blur(10px);}
.topbar-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:600;}
.topbar-path{font-family:'DM Mono',monospace;font-size:10px;color:var(--txt3);letter-spacing:.08em;}
.content{padding:28px 32px;flex:1;}
.tag{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.1em;padding:4px 10px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:rgba(0,229,180,0.1);color:var(--teal);border:1px solid rgba(0,229,180,.2);}
.tag-post{background:rgba(0,229,180,0.1);color:var(--teal);border:1px solid rgba(0,229,180,.2);}
.tag-get{background:rgba(79,142,247,0.1);color:var(--blue);border:1px solid rgba(79,142,247,.2);}

/* Endpoint cards */
.endpoint{background:rgba(12,21,36,0.85);border:1px solid var(--border);border-radius:12px;margin-bottom:16px;overflow:hidden;backdrop-filter:blur(8px);}
.ep-header{display:flex;align-items:center;gap:14px;padding:16px 22px;cursor:pointer;transition:background .15s;}
.ep-header:hover{background:rgba(0,229,180,0.03);}
.ep-method{font-family:'DM Mono',monospace;font-size:10px;font-weight:500;padding:4px 10px;border-radius:6px;letter-spacing:.08em;}
.ep-method.post{background:rgba(0,229,180,0.1);color:var(--teal);}
.ep-method.get{background:rgba(79,142,247,0.1);color:var(--blue);}
.ep-path{font-family:'DM Mono',monospace;font-size:13px;color:var(--txt);}
.ep-desc{font-size:12px;color:var(--txt3);margin-left:auto;}
.ep-body{border-top:1px solid var(--border);padding:20px 22px;display:none;}
.ep-body.open{display:block;}

label{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);display:block;margin-bottom:6px;}
textarea,input[type=text]{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px 14px;color:var(--txt);font-family:'DM Mono',monospace;font-size:12px;resize:vertical;outline:none;transition:border-color .2s;}
textarea:focus,input[type=text]:focus{border-color:var(--border2);}
textarea{min-height:100px;}

.btn{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;padding:9px 20px;border-radius:8px;border:none;cursor:pointer;transition:all .18s;margin-top:14px;}
.btn-primary{background:rgba(0,229,180,0.15);color:var(--teal);border:1px solid rgba(0,229,180,.3);}
.btn-primary:hover{background:rgba(0,229,180,0.25);}

.response-box{margin-top:16px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:14px;display:none;}
.response-box.visible{display:block;}
.response-label{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:8px;}
.response-content{font-family:'DM Mono',monospace;font-size:11px;color:var(--teal);white-space:pre-wrap;word-break:break-all;max-height:320px;overflow-y:auto;}
.response-content::-webkit-scrollbar{width:4px;}
.response-content::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.status-ok{color:var(--teal);}
.status-err{color:var(--red);}

.section-header{display:flex;align-items:baseline;gap:12px;margin-bottom:20px;}
.section-title{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);}
.section-line{flex:1;height:1px;background:var(--border);}
</style>
</head>
<body>
<div class="layout">
<nav class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-mark"><div class="logo-dot"></div>ClinicalNER</div>
    <div class="logo-sub">De-identification Platform</div>
  </div>
  <div class="nav-section">
    <div class="nav-label">Analytics</div>
    <a href="/dashboard" class="nav-item">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1" y="1" width="6" height="6" rx="1.5"/><rect x="9" y="1" width="6" height="6" rx="1.5"/><rect x="1" y="9" width="6" height="6" rx="1.5"/><rect x="9" y="9" width="6" height="6" rx="1.5"/></svg>
      Dashboard
    </a>
    <a href="/api/stats" class="nav-item" target="_blank">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 12 L5 8 L8 10 L11 5 L14 7"/></svg>
      Live Stats JSON
    </a>
    <div class="nav-label">Tools</div>
    <a href="/api-explorer" class="nav-item active">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/></svg>
      API Explorer
    </a>
    <a href="/health" class="nav-item" target="_blank">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 2 L8 14 M2 8 L14 8" stroke-linecap="round"/></svg>
      Health Probe
    </a>
  </div>
  <div class="sidebar-footer">
    <div class="status-badge"><div class="status-dot"></div>System Online</div>
  </div>
</nav>

<div class="main">
  <div class="topbar">
    <div>
      <div class="topbar-title">API Explorer</div>
      <div class="topbar-path">ClinicalNER / api-explorer</div>
    </div>
    <span class="tag tag-teal">Interactive</span>
  </div>

  <div class="content">
    <div class="section-header">
      <span class="section-title">Endpoints</span>
      <div class="section-line"></div>
    </div>

    <!-- De-identify -->
    <div class="endpoint">
      <div class="ep-header" onclick="toggle('ep1')">
        <span class="ep-method post">POST</span>
        <span class="ep-path">/api/deidentify</span>
        <span class="ep-desc">De-identify a clinical note</span>
      </div>
      <div class="ep-body" id="ep1">
        <label>Request body</label>
        <textarea id="body-deidentify">{"text": "Patient James Smith, DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145. Admitted to St. Mary's Hospital on 01/15/2024."}</textarea>
        <button class="btn btn-primary" onclick="send('POST','/api/deidentify','body-deidentify','res-deidentify')">Send Request</button>
        <div class="response-box" id="res-deidentify">
          <div class="response-label">Response <span id="res-deidentify-status"></span></div>
          <pre class="response-content" id="res-deidentify-body"></pre>
        </div>
      </div>
    </div>

    <!-- Stats -->
    <div class="endpoint">
      <div class="ep-header" onclick="toggle('ep2')">
        <span class="ep-method get">GET</span>
        <span class="ep-path">/api/stats</span>
        <span class="ep-desc">Corpus + pipeline statistics</span>
      </div>
      <div class="ep-body" id="ep2">
        <button class="btn btn-primary" onclick="send('GET','/api/stats',null,'res-stats')">Send Request</button>
        <div class="response-box" id="res-stats">
          <div class="response-label">Response <span id="res-stats-status"></span></div>
          <pre class="response-content" id="res-stats-body"></pre>
        </div>
      </div>
    </div>

    <!-- Anomaly scan -->
    <div class="endpoint">
      <div class="ep-header" onclick="toggle('ep3')">
        <span class="ep-method post">POST</span>
        <span class="ep-path">/api/anomaly-scan</span>
        <span class="ep-desc">IsolationForest anomaly detection</span>
      </div>
      <div class="ep-body" id="ep3">
        <label>Request body (min 10 notes)</label>
        <textarea id="body-anomaly" style="min-height:60px">{"notes": [{"id":1,"text":"Patient DOB: 04/12/1985. MRN100001.","entities":[{"label":"DOB"},{"label":"MRN"}]},{"id":2,"text":"Admitted 01/01/2023. Phone: (415)555-9876.","entities":[{"label":"DATE"},{"label":"PHONE"}]},{"id":3,"text":"Age: 72. MRN200002. Admitted to hospital.","entities":[{"label":"AGE"},{"label":"MRN"}]},{"id":4,"text":"DOB: 07/22/1978. Contact: (312)555-0034.","entities":[{"label":"DOB"},{"label":"PHONE"}]},{"id":5,"text":"Patient MRN300003 discharged 08/30/2022.","entities":[{"label":"MRN"},{"label":"DATE"}]},{"id":6,"text":"Follow-up 09/24/2022. Phone: (800)555-0000.","entities":[{"label":"DATE"},{"label":"PHONE"}]},{"id":7,"text":"DOB: 11/30/1990. MRN400004. Age: 32.","entities":[{"label":"DOB"},{"label":"MRN"},{"label":"AGE"}]},{"id":8,"text":"Admitted 04/01/2021. MRN: MRN500005.","entities":[{"label":"DATE"},{"label":"MRN"}]},{"id":9,"text":"Phone: (713)555-4444. Age: 63.","entities":[{"label":"PHONE"},{"label":"AGE"}]},{"id":10,"text":"Patient DOB: 03/15/1988. MRN600006.","entities":[{"label":"DOB"},{"label":"MRN"}]},{"id":11,"text":"x","entities":[]},{"id":12,"text":"DOB: 02/28/1965. Age: 58. MRN700007.","entities":[{"label":"DOB"},{"label":"AGE"},{"label":"MRN"}]},{"id":13,"text":"MRN800008. Admitted 01/01/2020.","entities":[{"label":"MRN"},{"label":"DATE"}]},{"id":14,"text":"Contact: (404)555-3333. Admitted 01/01/2020.","entities":[{"label":"PHONE"},{"label":"DATE"}]},{"id":15,"text":"DOB: 09/09/1958. Admitted to hospital on 12/12/2022.","entities":[{"label":"DOB"},{"label":"DATE"}]}]}</textarea>
        <button class="btn btn-primary" onclick="send('POST','/api/anomaly-scan','body-anomaly','res-anomaly')">Send Request</button>
        <div class="response-box" id="res-anomaly">
          <div class="response-label">Response <span id="res-anomaly-status"></span></div>
          <pre class="response-content" id="res-anomaly-body"></pre>
        </div>
      </div>
    </div>

    <!-- Health -->
    <div class="endpoint">
      <div class="ep-header" onclick="toggle('ep4')">
        <span class="ep-method get">GET</span>
        <span class="ep-path">/health</span>
        <span class="ep-desc">Liveness probe</span>
      </div>
      <div class="ep-body" id="ep4">
        <button class="btn btn-primary" onclick="send('GET','/health',null,'res-health')">Send Request</button>
        <div class="response-box" id="res-health">
          <div class="response-label">Response <span id="res-health-status"></span></div>
          <pre class="response-content" id="res-health-body"></pre>
        </div>
      </div>
    </div>

  </div>
</div>
</div>

<script>
function toggle(id) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
}

async function send(method, path, bodyId, resId) {
  const statusEl = document.getElementById(resId + '-status');
  const bodyEl   = document.getElementById(resId + '-body');
  const boxEl    = document.getElementById(resId);
  bodyEl.textContent = 'Loading…';
  boxEl.classList.add('visible');
  statusEl.textContent = '';

  try {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (bodyId) {
      const raw = document.getElementById(bodyId).value;
      JSON.parse(raw); // validate JSON
      opts.body = raw;
    }
    const res = await fetch(path, opts);
    const data = await res.json();
    statusEl.className = res.ok ? 'status-ok' : 'status-err';
    statusEl.textContent = `${res.status} ${res.ok ? 'OK' : 'ERROR'}`;
    bodyEl.textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    statusEl.className = 'status-err';
    statusEl.textContent = 'ERROR';
    bodyEl.textContent = e.message;
  }
}
</script>
</body>
</html>"""