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

from flask import Flask, jsonify, request, render_template_string, render_template
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
    # Set template folder to absolute path
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir), static_folder="static")
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
                min_conf = request.args.get("min_confidence", 0.0, type=float)
                entity_sql = loader.sql_query(
                    "SELECT entity_types_json, avg_confidence FROM processed_notes "
                    "WHERE entity_types_json IS NOT NULL"
                )
                for _, row in entity_sql.iterrows():
                    try:
                        conf = float(row.get("avg_confidence") or 0)
                        if conf < min_conf:
                            continue
                        for k, v in json.loads(row["entity_types_json"]).items():
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


    @app.route("/api/upload", methods=["POST"])
    def upload_csv():
        """
        Batch de-identification via CSV upload.

        Accepts: multipart/form-data with a CSV file containing a 'text' column.
        Returns: downloadable de-identified CSV with masked_text + entity_count columns.

        Maps to JD Activity 2: Clean and manage unstructured clinical data.
        """
        import io
        import csv
        import pandas as pd
        from flask import Response

        pipeline: NERPipeline = app.config["PIPELINE"]
        cleaner:  DataCleaner = app.config["CLEANER"]
        audit:    AuditLogger = app.config["AUDIT"]

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Send a CSV as multipart 'file' field.", "status": 400}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are accepted.", "status": 400}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Could not parse CSV: {e}", "status": 400}), 400

        # Auto-detect text column — try common names
        text_col = None
        for candidate in ["text", "transcription", "note", "clinical_note", "content", "body"]:
            if candidate in df.columns:
                text_col = candidate
                break

        if text_col is None:
            return jsonify({
                "error": f"No text column found. Expected one of: text, transcription, note. Got: {list(df.columns)}",
                "status": 400
            }), 400

        if len(df) > 500:
            return jsonify({"error": "Maximum 500 rows per upload.", "status": 413}), 413

        # Process each row through the full pipeline
        results = []
        total_entities = 0

        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if not raw_text.strip():
                results.append({"masked_text": "", "entity_count": 0, "entity_types": "", "is_valid": True, "avg_confidence": 0.0})
                continue

            pre     = cleaner.clean_pre_ner(raw_text)
            ner     = pipeline.process_note(pre.cleaned_text, save_to_db=False)
            post    = cleaner.clean_post_ner(ner["masked_text"])

            entity_types_str = ", ".join(f"{k}:{v}" for k, v in (ner.get("entity_types") or {}).items())
            total_entities  += ner["entity_count"]

            results.append({
                "masked_text":    post.cleaned_text,
                "entity_count":   ner["entity_count"],
                "entity_types":   entity_types_str,
                "is_valid":       post.is_valid,
                "avg_confidence": round(ner.get("avg_confidence", 0.0), 3),
            })

        # Build output DataFrame — keep all original columns + add de-id columns
        out_df = df.copy()
        out_df["masked_text"]    = [r["masked_text"]    for r in results]
        out_df["entity_count"]   = [r["entity_count"]   for r in results]
        out_df["entity_types"]   = [r["entity_types"]   for r in results]
        out_df["is_valid"]       = [r["is_valid"]        for r in results]
        out_df["avg_confidence"] = [r["avg_confidence"] for r in results]

        # Log batch job
        audit.log(
            EventType.PIPELINE_COMPLETE,
            description=f"Batch upload: {len(df)} notes, {total_entities} entities found",
            metadata={"rows": len(df), "total_entities": total_entities, "file": file.filename},
        )

        # Return as downloadable CSV
        output = io.StringIO()
        out_df.to_csv(output, index=False)
        csv_bytes = output.getvalue().encode("utf-8")

        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=deidentified_{file.filename}",
                "X-Total-Rows":       str(len(df)),
                "X-Total-Entities":   str(total_entities),
            }
        )


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

    @app.route("/stats")
    def stats_page():
        """
        Stats page — displays JSON data in a readable format.
        """
        return render_template("stats.html")

    @app.route("/upload")
    def upload_page():
        """Batch CSV de-identification page."""
        return render_template_string(UPLOAD_TEMPLATE)


    @app.route("/system-status")
    def system_status():
        """Visual system status page."""
        return render_template_string(SYSTEM_STATUS_TEMPLATE)


    @app.route("/api-explorer")
    def api_explorer():
        """Interactive API explorer page."""
        return render_template_string(API_EXPLORER_TEMPLATE)


    @app.route("/report/summary")
    def report_summary():
        """
        Study Status Summary — aggregate data quality report across all specialties.
        Maps to JD Activity 7: Generate data listings and study status reports.
        """
        loader: DataLoader = app.config["LOADER"]
        try:
            import datetime
            stats = loader.sql_query("""
                SELECT
                  cn.medical_specialty,
                  COUNT(cn.note_id)                             AS total_notes,
                  COUNT(pn.id)                                  AS processed_notes,
                  ROUND(AVG(pn.entity_count), 1)                AS avg_phi_per_note,
                  COALESCE(SUM(pn.entity_count), 0)             AS total_phi_found,
                  ROUND(COUNT(pn.id)*100.0/COUNT(cn.note_id),1) AS pct_processed
                FROM clinical_notes cn
                LEFT JOIN processed_notes pn ON cn.note_id = pn.note_id
                GROUP BY cn.medical_specialty
                ORDER BY total_notes DESC
            """).to_dict(orient="records")
            return jsonify({
                "report_type":       "Study Status Summary",
                "generated_at":      datetime.datetime.utcnow().isoformat() + "Z",
                "compliance":        "ICH E6 (R2) GCP",
                "total_specialties": len(stats),
                "total_notes":       sum(r["total_notes"] for r in stats),
                "total_processed":   sum(r["processed_notes"] for r in stats),
                "total_phi_found":   sum(r["total_phi_found"] or 0 for r in stats),
                "specialties":       stats,
            }), 200
        except Exception as e:
            logger.error("report_summary error: %s", e)
            return jsonify({"error": str(e), "status": 500}), 500


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
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:       #F8FAFC;
  --bg2:      #FFFFFF;
  --bg3:      #F1F5F9;
  --border:   #E2E8F0;
  --border2:  #CBD5E1;
  --teal:     #0F766E;
  --teal-light: rgba(15,118,110,0.1);
  --amber:    #D97706;
  --amber-light: rgba(217,119,6,0.1);
  --red:      #E11D48;
  --red-light: rgba(225,29,72,0.1);
  --blue:     #2563EB;
  --blue-light: rgba(37,99,235,0.1);
  --txt:      #0F172A;
  --txt2:     #334155;
  --txt3:     #64748B;
  --shadow:   0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);
  --shadow-hover: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
  --card-bg:  #FFFFFF;
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
    linear-gradient(rgba(15,118,110,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,118,110,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

body::after {
  content: '';
  position: fixed;
  top: -20%;
  right: -10%;
  width: 50vw;
  height: 60vh;
  background: radial-gradient(ellipse, rgba(37,99,235,0.04) 0%, transparent 65%);
  pointer-events: none;
  z-index: 0;
}

/* ── Layout ── */
.layout { display: flex; min-height: 100vh; position: relative; z-index: 1; }

/* ── Sidebar ── */
.sidebar {
  width: 240px;
  flex-shrink: 0;
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(12px);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 28px 0;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 50;
}

.sidebar-logo {
  padding: 0 24px 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.logo-mark {
  font-family: 'Syne', sans-serif;
  font-size: 19px;
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
  box-shadow: 0 0 10px rgba(15,118,110,0.4);
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%,100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px rgba(15,118,110,0.4); }
  50%      { opacity: 0.6; transform: scale(0.8); box-shadow: 0 0 4px rgba(15,118,110,0.2); }
}

.logo-sub {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--txt3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 6px;
  font-weight: 600;
}

.nav-section {
  padding: 0 12px;
  flex: 1;
}

.nav-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--txt3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0 16px;
  margin-bottom: 8px;
  margin-top: 20px;
  font-weight: 600;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  color: var(--txt2);
  transition: all 0.2s ease;
  margin-bottom: 4px;
  text-decoration: none;
}
.nav-item:hover { background: var(--bg3); color: var(--teal); transform: translateX(2px); }
.nav-item.active { background: var(--teal-light); color: var(--teal); border-left: 3px solid var(--teal); }
.nav-icon { width: 18px; stroke-width: 2px; }

.sidebar-footer {
  padding: 20px 24px 0;
  border-top: 1px solid var(--border);
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--teal);
  background: var(--teal-light);
  padding: 8px 14px;
  border-radius: 20px;
  font-weight: 600;
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
  padding: 20px 36px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 20;
}

.topbar-title {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--txt);
}

.topbar-path {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--txt3);
  letter-spacing: 0.08em;
  font-weight: 500;
  margin-top: 4px;
}

.topbar-right { display: flex; align-items: center; gap: 12px; }

.tag {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  padding: 6px 14px;
  border-radius: 20px;
  text-transform: uppercase;
}

.tag-teal { background: var(--teal-light); color: var(--teal); border: 1px solid rgba(15,118,110,0.2); }
.tag-amber { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }

/* ── Content ── */
.content { padding: 32px 36px; flex: 1; }

/* ── Section header ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--txt2);
}

.section-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Stat cards ── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 36px;
}

.stat-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
  opacity: 0;
  transform: translateY(16px);
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
  height: 4px;
  background: linear-gradient(90deg, var(--teal), var(--blue));
  opacity: 0;
  transition: opacity 0.3s;
}

.stat-card:hover { border-color: var(--border2); transform: translateY(-4px); box-shadow: var(--shadow-hover); }
.stat-card:hover::before { opacity: 1; }

.stat-label {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt3);
  margin-bottom: 12px;
}

.stat-value {
  font-family: 'Syne', sans-serif;
  font-size: 36px;
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
  font-size: 12px;
  font-weight: 500;
  color: var(--txt3);
  margin-top: 10px;
}

.stat-icon {
  position: absolute;
  top: 24px; right: 24px;
  font-size: 24px;
  opacity: 0.1;
  filter: grayscale(100%);
}

/* ── Charts grid ── */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 36px;
}

.chart-wide { grid-column: span 2; }

.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow);
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.35s;
  transition: box-shadow 0.3s ease;
}

.chart-card:hover { box-shadow: var(--shadow-hover); }

.chart-card:nth-child(2) { animation-delay: 0.42s; }
.chart-card:nth-child(3) { animation-delay: 0.49s; }
.chart-card:nth-child(4) { animation-delay: 0.56s; }

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.chart-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--txt);
}

.chart-badge {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  color: var(--txt3);
  letter-spacing: 0.1em;
  background: var(--bg3);
  padding: 4px 10px;
  border-radius: 6px;
}

canvas { max-height: 240px; }

/* ── Audit table ── */
.audit-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  overflow: hidden;
  margin-bottom: 24px;
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.6s;
}

.audit-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.audit-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--txt);
}

table { width: 100%; border-collapse: collapse; }

thead th {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt3);
  padding: 14px 24px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
}

tbody td {
  padding: 16px 24px;
  font-size: 13px;
  font-weight: 500;
  color: var(--txt2);
  border-bottom: 1px solid var(--border);
}

tbody tr:last-child td { border-bottom: none; }
tbody tr { transition: background 0.15s; }
tbody tr:hover td { background: var(--bg3); color: var(--txt); }

/* ── Event badges ── */
.ev {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 4px 10px;
  border-radius: 20px;
}

.ev-teal   { background: var(--teal-light); color: var(--teal); border: 1px solid rgba(15,118,110,0.2); }
.ev-amber  { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }
.ev-red    { background: var(--red-light); color: var(--red); border: 1px solid rgba(225,29,72,0.2); }
.ev-blue   { background: var(--blue-light); color: var(--blue); border: 1px solid rgba(37,99,235,0.2); }

.ev-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; }

/* ── Count col ── */
.count {
  font-family: 'DM Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  color: var(--txt);
}

.ts {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  color: var(--txt3);
}

/* ── Loading shimmer ── */
.shimmer {
  background: linear-gradient(90deg, var(--bg3) 25%, #FFFFFF 50%, var(--bg3) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 6px;
  height: 24px;
  width: 90px;
}
@keyframes shimmer { to { background-position: -200% 0; } }

/* ── Error ── */
.error-banner {
  background: var(--red-light);
  border: 1px solid rgba(225,29,72,0.2);
  border-radius: 12px;
  padding: 16px 20px;
  font-family: 'Instrument Sans', sans-serif;
  font-weight: 500;
  font-size: 14px;
  color: var(--red);
  margin-bottom: 24px;
  box-shadow: var(--shadow);
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
      <div class="nav-label">Tools</div>
      <a href="/api-explorer" class="nav-item">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/>
        </svg>
        API Explorer
      </a>
      <a href="/upload" class="nav-item">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M8 10V3M5 6l3-3 3 3" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2 11v2a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2" stroke-linecap="round"/>
        </svg>
        Batch Upload
      </a>
      <a href="/system-status" class="nav-item">
        <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2zm0 4v4m0 2v.5" stroke-linecap="round"/>
        </svg>
        System Status
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


      <!-- Confidence Threshold Slider -->
      <div class="section-header" style="margin-top:8px;">
        <span class="section-title">Confidence Filter</span>
        <div class="section-line"></div>
      </div>

      <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:16px;padding:24px 28px;box-shadow:var(--shadow);margin-bottom:36px;opacity:0;animation:card-in .5s ease .58s forwards;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;">
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--txt);margin-bottom:4px;">Entity confidence threshold</div>
            <div style="font-family:'DM Mono',monospace;font-size:12px;font-weight:500;color:var(--txt3);">Filter entity chart and table to show only notes with avg confidence ≥ threshold</div>
          </div>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:var(--teal);line-height:1;" id="conf-display">0%</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--txt3);margin-top:4px;">Min confidence</div>
            </div>
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:var(--txt);line-height:1;" id="conf-notes-shown">—</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--txt3);margin-top:4px;">Notes shown</div>
            </div>
            <button onclick="resetConfidence()" style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.08em;padding:8px 16px;border-radius:8px;border:1px solid var(--border2);background:var(--card-bg);color:var(--txt3);cursor:pointer;transition:all .2s;" onmouseover="this.style.borderColor='var(--teal)';this.style.color='var(--teal)'" onmouseout="this.style.borderColor='var(--border2)';this.style.color='var(--txt3)'">Reset</button>
          </div>
        </div>

        <!-- Slider track -->
        <div style="position:relative;margin-bottom:16px;">
          <input type="range" id="conf-slider" min="0" max="100" value="0" step="5"
            style="width:100%;height:6px;border-radius:3px;outline:none;-webkit-appearance:none;appearance:none;background:linear-gradient(90deg,var(--teal) 0%,var(--teal) 0%,var(--border) 0%,var(--border) 100%);cursor:pointer;"
            oninput="onSliderMove(this.value)">
        </div>

        <!-- Tick labels -->
        <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.06em;margin-bottom:18px;">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>

        <!-- Confidence band legend -->
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:7px;font-size:13px;font-weight:500;color:var(--txt2);">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--teal);"></div>High confidence (≥ 90%)
          </div>
          <div style="display:flex;align-items:center;gap:7px;font-size:13px;font-weight:500;color:var(--txt2);">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--amber);"></div>Medium (70–89%)
          </div>
          <div style="display:flex;align-items:center;gap:7px;font-size:13px;font-weight:500;color:var(--txt2);">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--red);"></div>Low (< 70%) — review recommended
          </div>
          <div style="margin-left:auto;font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);" id="conf-filter-label">No filter active</div>
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
Chart.defaults.color = '#64748B'; /* Slate 500 */ Chart.defaults.font.weight = '500';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 10;

const PALETTE = ['#0F766E','#2563EB','#D97706','#E11D48','#7C3AED','#0EA5E9','#DB2777','#10B981','#F59E0B'];

const GRID = { color: '#F1F5F9', drawBorder: false };

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
        backgroundColor: 'rgba(15,118,110,0.15)', borderColor: '#0F766E', borderWidth: 1.5, borderRadius: 6,
        
        
        
        hoverBackgroundColor: 'rgba(15,118,110,0.25)',
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

// ── Confidence threshold slider ────────────────────────────────────────
let currentConfidence = 0;
let debounceConf = null;

function updateSliderTrack(val) {
  const slider = document.getElementById('conf-slider');
  slider.style.background = `linear-gradient(90deg, var(--teal) ${val}%, var(--teal) ${val}%, var(--border) ${val}%, var(--border) 100%)`;
}

function onSliderMove(val) {
  val = parseInt(val);
  currentConfidence = val;
  document.getElementById('conf-display').textContent = val + '%';
  updateSliderTrack(val);

  const label = document.getElementById('conf-filter-label');
  if (val === 0) {
    label.textContent = 'No filter active';
    label.style.color = 'var(--txt3)';
  } else {
    label.textContent = `Showing notes with avg confidence ≥ ${val}%`;
    label.style.color = val >= 90 ? 'var(--teal)' : val >= 70 ? 'var(--amber)' : 'var(--red)';
  }

  clearTimeout(debounceConf);
  debounceConf = setTimeout(() => loadWithConfidence(val / 100), 350);
}

function resetConfidence() {
  currentConfidence = 0;
  document.getElementById('conf-slider').value = 0;
  onSliderMove(0);
}

async function loadWithConfidence(minConf) {
  try {
    const url   = minConf > 0 ? `/api/stats?min_confidence=${minConf}` : '/api/stats';
    const res   = await fetch(url);
    const data  = await res.json();
    if (data.error) return;

    // Update entity donut with filtered data
    destroyChart('entityChart');
    if (Object.keys(data.entity_totals).length > 0) {
      donut('entityChart', Object.keys(data.entity_totals), Object.values(data.entity_totals));
    } else {
      const canvas = document.getElementById('entityChart');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px DM Mono, monospace';
      ctx.fillStyle = 'var(--txt3)';
      ctx.textAlign = 'center';
      ctx.fillText('No entities at this threshold', canvas.width/2, canvas.height/2);
    }

    // Update processed count display
    document.getElementById('conf-notes-shown').textContent =
      data.processed_count.toLocaleString();

  } catch(e) {
    console.error('Confidence filter error:', e);
  }
}


</script>

</script>

<!-- ── Live De-identification Widget ── -->
<div style="margin:0 36px 48px;opacity:0;animation:card-in .5s ease .7s forwards;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;">
    <span style="font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--txt2);">Live De-identification</span>
    <div style="flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);"></div>
    <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--txt3);letter-spacing:.1em;background:var(--bg3);padding:4px 12px;border-radius:6px;" id="widget-status">READY</span>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border);background:var(--amber-light);">
        <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--amber);display:flex;align-items:center;gap:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:var(--amber);"></div>Raw clinical note
        </div>
        <button onclick="loadSample()" style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;padding:5px 14px;border-radius:6px;border:1px solid var(--border2);background:var(--card-bg);color:var(--txt3);cursor:pointer;transition:all .2s;" onmouseover="this.style.color='var(--teal)';this.style.borderColor='var(--teal)'" onmouseout="this.style.color='var(--txt3)';this.style.borderColor='var(--border2)'">Load sample</button>
      </div>
      <textarea id="deid-input" placeholder="Paste a clinical note here and watch it de-identify in real time..." style="width:100%;height:240px;background:transparent;border:none;outline:none;padding:18px 20px;font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:var(--txt2);resize:none;line-height:1.7;"></textarea>
      <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;border-top:1px solid var(--border);background:var(--bg3);">
        <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);" id="char-count">0 chars</span>
        <div style="display:flex;gap:8px;">
          <button onclick="clearWidget()" style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;padding:6px 14px;border-radius:6px;border:1px solid var(--border2);background:var(--card-bg);color:var(--txt3);cursor:pointer;">Clear</button>
          <button onclick="runDeid()" style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;padding:6px 18px;border-radius:6px;border:1px solid var(--teal);background:rgba(15,118,110,0.1);color:var(--teal);cursor:pointer;">Run now</button>
        </div>
      </div>
    </div>

    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border);background:rgba(15,118,110,0.08);">
        <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);display:flex;align-items:center;gap:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:var(--teal);"></div>De-identified output
        </div>
        <div id="entity-summary" style="font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--txt3);"></div>
      </div>
      <div id="deid-output" style="height:240px;padding:18px 20px;font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:var(--txt2);line-height:1.7;overflow-y:auto;white-space:pre-wrap;word-break:break-word;">
        <span style="color:var(--txt3);font-style:italic;">Output will appear here...</span>
      </div>
      <div style="padding:12px 20px;border-top:1px solid var(--border);background:var(--bg3);">
        <div id="entity-chips" style="display:flex;flex-wrap:wrap;gap:6px;min-height:24px;"></div>
      </div>
    </div>
  </div>

  <div id="metrics-row" style="display:none;margin-top:16px;grid-template-columns:repeat(4,1fr);gap:16px;">
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);">
      <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:8px;">Entities found</div>
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--teal);" id="m-count">—</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);">
      <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:8px;">Avg confidence</div>
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--txt);" id="m-conf">—</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);">
      <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:8px;">PHI types</div>
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--txt);" id="m-types">—</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);">
      <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:8px;">Valid output</div>
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;" id="m-valid">—</div>
    </div>
  </div>
</div>

<script>
const ENTITY_COLORS = {
  DATE:     {bg:'rgba(37,99,235,0.1)',   border:'rgba(37,99,235,0.3)',   text:'#2563EB'},
  DOB:      {bg:'rgba(37,99,235,0.07)',  border:'rgba(37,99,235,0.2)',   text:'#60A5FA'},
  PHONE:    {bg:'rgba(15,118,110,0.1)',  border:'rgba(15,118,110,0.3)',  text:'#0F766E'},
  MRN:      {bg:'rgba(217,119,6,0.1)',   border:'rgba(217,119,6,0.3)',   text:'#D97706'},
  HOSPITAL: {bg:'rgba(124,58,237,0.1)',  border:'rgba(124,58,237,0.3)',  text:'#7C3AED'},
  AGE:      {bg:'rgba(14,165,233,0.1)',  border:'rgba(14,165,233,0.3)',  text:'#0EA5E9'},
  PERSON:   {bg:'rgba(225,29,72,0.1)',   border:'rgba(225,29,72,0.3)',   text:'#E11D48'},
  LOCATION: {bg:'rgba(15,118,110,0.07)', border:'rgba(15,118,110,0.2)', text:'#10B981'},
};

const SAMPLE = `Patient: James R. Smith, DOB: 06/15/1978\nMRN: MRN302145. Admitted to St. Mary's Hospital on 01/15/2024.\nContact: (415) 555-9876. Referred by Dr. Emily Chen.\nAge: 45-year-old male with Type 2 Diabetes.\nFollow-up at Memorial Medical Center on 03/01/2024.`;

let debounceTimer = null;
let lastText = '';
const inp = document.getElementById('deid-input');
const out = document.getElementById('deid-output');
const wst = document.getElementById('widget-status');
const mtr = document.getElementById('metrics-row');

function loadSample() { inp.value = SAMPLE; inp.dispatchEvent(new Event('input')); }
function clearWidget() {
  inp.value = ''; out.innerHTML = '<span style="color:var(--txt3);font-style:italic;">Output will appear here...</span>';
  document.getElementById('entity-chips').innerHTML = '';
  document.getElementById('entity-summary').innerHTML = '';
  mtr.style.display = 'none'; wst.textContent = 'READY'; lastText = '';
}
function runDeid() { clearTimeout(debounceTimer); callAPI(inp.value.trim()); }

inp.addEventListener('input', () => {
  document.getElementById('char-count').textContent = inp.value.length + ' chars';
  clearTimeout(debounceTimer);
  if (!inp.value.trim()) return;
  wst.textContent = 'TYPING...';
  debounceTimer = setTimeout(() => callAPI(inp.value.trim()), 600);
});

async function callAPI(text) {
  if (!text || text === lastText) return;
  lastText = text;
  wst.textContent = 'PROCESSING...';
  out.innerHTML = '<span style="color:var(--txt3);">Running NER pipeline...</span>';
  try {
    const res = await fetch('/api/deidentify', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, save:false})
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'API error');
    let html = d.masked_text;
    html = html.replace(/\[([A-Z]+)\]/g, (m, label) => {
      const c = ENTITY_COLORS[label] || {bg:'rgba(0,0,0,0.05)',border:'rgba(0,0,0,0.1)',text:'#64748B'};
      return `<span style="background:${c.bg};border:1px solid ${c.border};color:${c.text};border-radius:5px;padding:1px 7px;font-weight:700;font-size:12px;">${m}</span>`;
    });
    out.innerHTML = html;
    const counts = d.entity_types || {};
    document.getElementById('entity-chips').innerHTML = Object.entries(counts).map(([label, n]) => {
      const c = ENTITY_COLORS[label] || {bg:'rgba(0,0,0,0.05)',border:'rgba(0,0,0,0.1)',text:'#64748B'};
      return `<span style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;padding:4px 10px;border-radius:20px;background:${c.bg};border:1px solid ${c.border};color:${c.text};">${label} ×${n}</span>`;
    }).join('');
    document.getElementById('entity-summary').textContent = d.entity_count + ' entities';
    document.getElementById('m-count').textContent = d.entity_count;
    const confPct = ((d.avg_confidence||0)*100).toFixed(0);
    const confEl = document.getElementById('m-conf');
    confEl.textContent = confPct + '%';
    confEl.style.color = confPct >= 90 ? 'var(--teal)' : confPct >= 70 ? 'var(--amber)' : 'var(--red)';
    document.getElementById('m-types').textContent = Object.keys(counts).length;
    const ve = document.getElementById('m-valid');
    ve.textContent = d.is_valid ? 'Yes' : 'No';
    ve.style.color = d.is_valid ? 'var(--teal)' : 'var(--red)';
    mtr.style.display = 'grid';
    wst.textContent = 'COMPLETE';
  } catch(e) {
    out.innerHTML = `<span style="color:var(--red);">Error: ${e.message}</span>`;
    wst.textContent = 'ERROR';
  }
}
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
  --bg:       #F8FAFC;
  --bg2:      #FFFFFF;
  --bg3:      #F1F5F9;
  --border:   #E2E8F0;
  --teal:     #0F766E;
  --teal-light: rgba(15,118,110,0.1);
  --amber:    #D97706;
  --amber-light: rgba(217,119,6,0.1);
  --red:      #E11D48;
  --txt:      #0F172A;
  --txt2:     #334155;
  --txt3:     #64748B;
  --shadow:   0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);
  --card-bg:  #FFFFFF;
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
    linear-gradient(rgba(15,118,110,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,118,110,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}
.page { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 40px; }

.breadcrumb {
  display: flex; align-items: center; gap: 8px;
  font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500;
  color: var(--txt3); margin-bottom: 32px; letter-spacing: 0.08em;
}
.breadcrumb a { color: var(--teal); text-decoration: none; font-weight: 600;}
.breadcrumb a:hover { text-decoration: underline; }
.breadcrumb-sep { opacity: 0.4; }

.report-header {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}

.report-title {
  font-family: 'Syne', sans-serif;
  font-size: 28px; font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--txt);
}

.report-title span { color: var(--teal); }

.report-meta {
  display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap;
}

.meta-item {
  font-family: 'DM Mono', monospace;
  font-size: 11px; letter-spacing: 0.1em;
  color: var(--txt3); text-transform: uppercase; font-weight: 600;
}
.meta-item strong { color: var(--txt); font-weight: 700; margin-left: 8px; }

.header-right { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }

.badge {
  font-family: 'DM Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  padding: 6px 14px; border-radius: 20px; text-transform: uppercase;
}
.badge-teal  { background: var(--teal-light); color: var(--teal);  border: 1px solid rgba(15,118,110,0.2); }
.badge-amber { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }

/* Diff panels */
.diff-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 32px;
}

.diff-panel {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: var(--shadow);
}

.panel-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.panel-head-title {
  font-family: 'DM Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  display: flex; align-items: center; gap: 10px;
}

.panel-dot {
  width: 8px; height: 8px; border-radius: 50%;
}

.panel-orig  .panel-dot { background: var(--amber); }
.panel-masked .panel-dot { background: var(--teal);  }
.panel-orig  .panel-head { border-bottom-color: rgba(217,119,6,0.2); background: var(--amber-light); }
.panel-masked .panel-head { border-bottom-color: rgba(15,118,110,0.2); background: var(--teal-light); }
.panel-orig  .panel-head-title { color: var(--amber); }
.panel-masked .panel-head-title { color: var(--teal);  }

.panel-body {
  padding: 24px;
  font-family: 'DM Mono', monospace;
  font-size: 13px; line-height: 1.8;
  white-space: pre-wrap; word-break: break-word;
  max-height: 500px; overflow-y: auto;
  color: var(--txt2);
}

.panel-body::-webkit-scrollbar { width: 6px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.masked-token {
  background: var(--teal-light);
  color: var(--teal);
  border: 1px solid rgba(15,118,110,0.3);
  border-radius: 6px;
  padding: 2px 6px;
  font-weight: 600;
  font-size: 12px;
}

/* Stats row */
.stats-row {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;
  margin-bottom: 0;
}

.mini-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
  box-shadow: var(--shadow);
}
.mini-label {
  font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--txt3); margin-bottom: 8px;
}
.mini-value {
  font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
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
:root{--bg:#F8FAFC;--bg2:#FFFFFF;--bg3:#F1F5F9;--border:#E2E8F0;--border2:#CBD5E1;--teal:#0F766E;--amber:#D97706;--red:#E11D48;--blue:#2563EB;--txt:#0F172A;--txt2:#334155;--txt3:#64748B; --shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025); --shadow-hover: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(15,118,110,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(15,118,110,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:240px;flex-shrink:0;background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:19px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 10px rgba(15,118,110,.4);animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;color:var(--txt3);font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;margin-top:20px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--txt2);transition:all .2s;margin-bottom:4px;text-decoration:none;}
.nav-item:hover{background:var(--bg3);color:var(--teal);transform:translateX(2px);}
.nav-item.active{background:rgba(15,118,110,0.1);color:var(--teal);border-left:3px solid var(--teal);}
.nav-icon{width:18px;stroke-width:2px;opacity:.8;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--teal);background:rgba(15,118,110,0.1);padding:8px 14px;border-radius:20px;letter-spacing:.05em;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 36px;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);}
.topbar-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.topbar-path{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);letter-spacing:.08em;margin-top:4px;}
.content{padding:32px 36px;flex:1;}
.tag{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;padding:6px 14px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}
.tag-post{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}
.tag-get{background:rgba(37,99,235,0.1);color:var(--blue);border:1px solid rgba(37,99,235,.2);}

/* Endpoint cards */
.endpoint{background:var(--bg2);border:1px solid var(--border);border-radius:16px;margin-bottom:20px;overflow:hidden;box-shadow:var(--shadow);transition:box-shadow .3s;}
.endpoint:hover{box-shadow:var(--shadow-hover);}
.ep-header{display:flex;align-items:center;gap:16px;padding:20px 24px;cursor:pointer;transition:background .15s;}
.ep-header:hover{background:var(--bg3);}
.ep-method{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;padding:6px 12px;border-radius:8px;letter-spacing:.08em;}
.ep-method.post{background:rgba(15,118,110,0.1);color:var(--teal);}
.ep-method.get{background:rgba(37,99,235,0.1);color:var(--blue);}
.ep-path{font-family:'DM Mono',monospace;font-size:14px;font-weight:600;color:var(--txt);}
.ep-desc{font-size:13px;font-weight:500;color:var(--txt3);margin-left:auto;}
.ep-body{border-top:1px solid var(--border);padding:24px;display:none;background:var(--bg2);}
.ep-body.open{display:block;}

label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);display:block;margin-bottom:8px;}
textarea,input[type=text]{width:100%;font-weight:500;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:16px 18px;color:var(--txt);font-family:'DM Mono',monospace;font-size:13px;resize:vertical;outline:none;transition:border-color .2s;box-shadow:inset 0 1px 2px rgba(0,0,0,0.02);}
textarea:focus,input[type=text]:focus{border-color:var(--teal);background:#FFF;}
textarea{min-height:120px;}

.btn{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;padding:12px 24px;border-radius:8px;border:none;cursor:pointer;transition:all .2s;margin-top:16px;}
.btn-primary{background:var(--teal);color:#FFF;box-shadow:0 4px 6px -1px rgba(15,118,110,0.2);}
.btn-primary:hover{background:#0D9488;transform:translateY(-1px);box-shadow:0 6px 8px -1px rgba(15,118,110,0.3);}

.response-box{margin-top:20px;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:16px;display:none;}
.response-box.visible{display:block;}
.response-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:12px;}
.response-content{font-family:'DM Mono',monospace;font-size:12px;font-weight:500;color:var(--teal);white-space:pre-wrap;word-break:break-all;max-height:360px;overflow-y:auto;}
.response-content::-webkit-scrollbar{width:6px;}
.response-content::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
.status-ok{color:var(--teal);}
.status-err{color:var(--red);}

.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:24px;}
.section-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
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
    <div class="nav-label">Tools</div>
    <a href="/api-explorer" class="nav-item active">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/></svg>
      API Explorer
    </a>
    <a href="/system-status" class="nav-item">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2zm0 4v4m0 2v.5" stroke-linecap="round"/></svg>
      System Status
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

SYSTEM_STATUS_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>System Status · ClinicalNER</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#F8FAFC;--bg2:#FFFFFF;--bg3:#F1F5F9;--border:#E2E8F0;--border2:#CBD5E1;--teal:#0F766E;--amber:#D97706;--red:#E11D48;--blue:#2563EB;--txt:#0F172A;--txt2:#334155;--txt3:#64748B;--shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);--shadow-hover:0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(15,118,110,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(15,118,110,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:240px;flex-shrink:0;background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:19px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 10px rgba(15,118,110,.4);animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;margin-top:20px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--txt2);transition:all .2s;margin-bottom:4px;text-decoration:none;}
.nav-item:hover{background:var(--bg3);color:var(--teal);transform:translateX(2px);}
.nav-item.active{background:rgba(15,118,110,0.1);color:var(--teal);border-left:3px solid var(--teal);}
.nav-icon{width:18px;stroke-width:2px;opacity:.8;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--teal);background:rgba(15,118,110,0.1);padding:8px 14px;border-radius:20px;letter-spacing:.05em;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 36px;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);}
.topbar-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.topbar-path{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);letter-spacing:.08em;margin-top:4px;}
.content{padding:32px 36px;flex:1;}
.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:24px;}
.section-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.tag{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;padding:6px 14px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}

/* Status cards */
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:36px;}
.status-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:24px;box-shadow:var(--shadow);opacity:0;transform:translateY(16px);animation:fadein .5s cubic-bezier(0.16, 1, 0.3, 1) forwards;}
.status-card:nth-child(2){animation-delay:.08s;}
.status-card:nth-child(3){animation-delay:.16s;}
@keyframes fadein{to{opacity:1;transform:translateY(0);}}
.status-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.status-card-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--txt);}
.status-indicator{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;}
.ind-dot{width:8px;height:8px;border-radius:50%;}
.ind-ok{background:var(--teal);box-shadow:0 0 8px rgba(15,118,110,0.5);}
.ind-warn{background:var(--amber);box-shadow:0 0 8px rgba(217,119,6,0.5);}
.ind-err{background:var(--red);box-shadow:0 0 8px rgba(225,29,72,0.5);}
.ok{color:var(--teal);}
.warn{color:var(--amber);}
.err{color:var(--red);}

/* Check rows */
.check-list{display:flex;flex-direction:column;gap:10px;}
.check-row{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--bg3);border-radius:10px;border:1px solid var(--border);}
.check-label{font-size:13px;font-weight:600;color:var(--txt2);}
.check-value{font-family:'DM Mono',monospace;font-size:12px;font-weight:600;}

/* Route table */
.route-table{background:var(--bg2);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);}
.route-header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;border-bottom:1px solid var(--border);background:var(--bg3);}
.route-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--txt);}
table{width:100%;border-collapse:collapse;}
thead th{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);padding:14px 24px;text-align:left;border-bottom:1px solid var(--border);background:var(--bg2);}
tbody td{padding:16px 24px;font-size:13px;font-weight:500;color:var(--txt2);border-bottom:1px solid var(--border);}
tbody tr:last-child td{border-bottom:none;}
tbody tr{transition:background .2s;}
tbody tr:hover td{background:rgba(15,118,110,0.02);color:var(--txt);}
.method{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;letter-spacing:0.05em;}
.method-get{background:rgba(37,99,235,0.1);color:var(--blue);}
.method-post{background:rgba(15,118,110,0.1);color:var(--teal);}
.test-btn{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.08em;padding:6px 14px;border-radius:8px;border:1px solid var(--border2);background:#FFF;color:var(--txt2);cursor:pointer;transition:all .2s;text-decoration:none;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
.test-btn:hover{border-color:var(--teal);color:var(--teal);transform:translateY(-1px);box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);}
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
    <div class="nav-label">Tools</div>
    <a href="/api-explorer" class="nav-item">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/></svg>
      API Explorer
    </a>
    <a href="/system-status" class="nav-item active">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2zm0 4v4m0 2v.5" stroke-linecap="round"/></svg>
      System Status
    </a>
  </div>
  <div class="sidebar-footer">
    <div class="status-badge"><div class="status-dot"></div>System Online</div>
  </div>
</nav>

<div class="main">
  <div class="topbar">
    <div>
      <div class="topbar-title">System Status</div>
      <div class="topbar-path">ClinicalNER / system-status</div>
    </div>
    <span class="tag tag-teal" id="overall-status">Checking…</span>
  </div>

  <div class="content">
    <div class="section-header">
      <span class="section-title">Service Health</span>
      <div class="section-line"></div>
    </div>

    <div class="status-grid">
      <div class="status-card">
        <div class="status-card-header">
          <div class="status-card-title">Flask API</div>
          <div class="status-indicator" id="api-indicator">
            <div class="ind-dot ind-ok"></div>
            <span class="ok">Operational</span>
          </div>
        </div>
        <div class="check-list">
          <div class="check-row"><span class="check-label">Service</span><span class="check-value ok" id="service-name">—</span></div>
          <div class="check-row"><span class="check-label">Status</span><span class="check-value ok" id="service-status">—</span></div>
          <div class="check-row"><span class="check-label">Endpoint</span><span class="check-value" style="color:var(--txt3)">localhost:5000</span></div>
        </div>
      </div>

      <div class="status-card">
        <div class="status-card-header">
          <div class="status-card-title">Database</div>
          <div class="status-indicator" id="db-indicator">
            <div class="ind-dot ind-ok"></div>
            <span class="ok">Connected</span>
          </div>
        </div>
        <div class="check-list">
          <div class="check-row"><span class="check-label">Clinical notes</span><span class="check-value ok" id="note-count">—</span></div>
          <div class="check-row"><span class="check-label">Processed</span><span class="check-value ok" id="proc-count">—</span></div>
          <div class="check-row"><span class="check-label">Audit events</span><span class="check-value ok" id="audit-count">—</span></div>
        </div>
      </div>

      <div class="status-card">
        <div class="status-card-header">
          <div class="status-card-title">NLP Pipeline</div>
          <div class="status-indicator">
            <div class="ind-dot ind-ok"></div>
            <span class="ok">Active</span>
          </div>
        </div>
        <div class="check-list">
          <div class="check-row"><span class="check-label">NER mode</span><span class="check-value ok">spaCy + Regex</span></div>
          <div class="check-row"><span class="check-label">PHI types</span><span class="check-value ok">6 entity labels</span></div>
          <div class="check-row"><span class="check-label">Anomaly detect</span><span class="check-value ok">IsolationForest</span></div>
        </div>
      </div>
    </div>

    <div class="section-header">
      <span class="section-title">API Routes</span>
      <div class="section-line"></div>
    </div>

    <div class="route-table">
      <div class="route-header">
        <div class="route-title">Available endpoints</div>
        <span class="tag tag-teal">7 routes</span>
      </div>
      <table>
        <thead>
          <tr><th>Method</th><th>Route</th><th>Description</th><th>Test</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="method method-get">GET</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/health</span></td>
            <td>Liveness probe · Docker HEALTHCHECK</td>
            <td><a href="/health" class="test-btn" target="_blank">Open</a></td>
          </tr>
          <tr>
            <td><span class="method method-get">GET</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/dashboard</span></td>
            <td>Live pipeline analytics dashboard</td>
            <td><a href="/dashboard" class="test-btn">Open</a></td>
          </tr>
          <tr>
            <td><span class="method method-get">GET</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/api/stats</span></td>
            <td>Corpus statistics JSON</td>
            <td><a href="/api/stats" class="test-btn" target="_blank">Open</a></td>
          </tr>
          <tr>
            <td><span class="method method-post">POST</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/api/deidentify</span></td>
            <td>De-identify a clinical note · returns masked text + entities</td>
            <td><a href="/api-explorer" class="test-btn">Try it</a></td>
          </tr>
          <tr>
            <td><span class="method method-post">POST</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/api/anomaly-scan</span></td>
            <td>IsolationForest anomaly detection on note batch</td>
            <td><a href="/api-explorer" class="test-btn">Try it</a></td>
          </tr>
          <tr>
            <td><span class="method method-get">GET</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/report/&lt;id&gt;</span></td>
            <td>Before / after de-identification diff view</td>
            <td><a href="/report/1" class="test-btn">Open</a></td>
          </tr>
          <tr>
            <td><span class="method method-get">GET</span></td>
            <td><span style="font-family:'DM Mono',monospace;font-size:12px;">/api/note/&lt;id&gt;</span></td>
            <td>Fetch a processed note by ID</td>
            <td><a href="/api/note/1" class="test-btn" target="_blank">Open</a></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
</div>

<script>
async function checkStatus() {
  try {
    const [health, stats] = await Promise.all([
      fetch('/health').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()),
    ]);
    document.getElementById('service-name').textContent  = health.service || '—';
    document.getElementById('service-status').textContent = health.status  || '—';
    document.getElementById('note-count').textContent    = (stats.note_count  || 0).toLocaleString();
    document.getElementById('proc-count').textContent    = (stats.processed_count || 0).toLocaleString();
    document.getElementById('audit-count').textContent   = (stats.total_audit_events || 0).toLocaleString();
    document.getElementById('overall-status').textContent = 'All Systems Operational';
  } catch(e) {
    document.getElementById('overall-status').textContent = 'Degraded';
    document.getElementById('overall-status').style.color = 'var(--red)';
  }
}
checkStatus();
</script>
</body>
</html>"""
UPLOAD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Upload · ClinicalNER</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500;600&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#F8FAFC;--bg2:#FFFFFF;--bg3:#F1F5F9;--border:#E2E8F0;--border2:#CBD5E1;
  --teal:#0F766E;--teal-l:rgba(15,118,110,0.1);--amber:#D97706;--amber-l:rgba(217,119,6,0.1);
  --red:#E11D48;--blue:#2563EB;--blue-l:rgba(37,99,235,0.1);
  --txt:#0F172A;--txt2:#334155;--txt3:#64748B;
  --shadow:0 4px 6px -1px rgba(0,0,0,0.05),0 2px 4px -2px rgba(0,0,0,0.025);
  --shadow-h:0 10px 15px -3px rgba(0,0,0,0.08),0 4px 6px -4px rgba(0,0,0,0.04);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(15,118,110,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(15,118,110,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:240px;flex-shrink:0;background:rgba(255,255,255,0.9);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;z-index:50;}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:19px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 10px rgba(15,118,110,.4);animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;margin-top:20px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;font-size:14px;font-weight:600;color:var(--txt2);transition:all .2s;margin-bottom:4px;text-decoration:none;}
.nav-item:hover{background:var(--bg3);color:var(--teal);transform:translateX(2px);}
.nav-item.active{background:var(--teal-l);color:var(--teal);border-left:3px solid var(--teal);}
.nav-icon{width:18px;stroke-width:2px;opacity:.8;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--teal);background:var(--teal-l);padding:8px 14px;border-radius:20px;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 36px;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.9);}
.topbar-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.topbar-path{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);letter-spacing:.08em;margin-top:4px;}
.topbar-right{display:flex;gap:12px;align-items:center;}
.tag{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;padding:6px 14px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:var(--teal-l);color:var(--teal);border:1px solid rgba(15,118,110,.2);}
.tag-amber{background:var(--amber-l);color:var(--amber);border:1px solid rgba(217,119,6,.2);}
.content{padding:36px;flex:1;max-width:900px;}
.section-hdr{display:flex;align-items:center;gap:16px;margin-bottom:28px;}
.section-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--txt2);}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
/* Drop zone */
.drop-zone{background:var(--bg2);border:2px dashed var(--border2);border-radius:20px;padding:56px 40px;text-align:center;cursor:pointer;transition:all .25s;box-shadow:var(--shadow);}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--teal);background:var(--teal-l);transform:translateY(-2px);box-shadow:var(--shadow-h);}
.drop-icon{width:52px;height:52px;margin:0 auto 18px;background:var(--teal-l);border-radius:16px;display:flex;align-items:center;justify-content:center;}
.drop-title{font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:var(--txt);margin-bottom:8px;}
.drop-sub{font-size:14px;font-weight:500;color:var(--txt3);margin-bottom:24px;}
.btn{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;padding:13px 28px;border-radius:10px;border:none;cursor:pointer;transition:all .2s;}
.btn-primary{background:var(--teal);color:#fff;box-shadow:0 4px 6px -1px rgba(15,118,110,.3);}
.btn-primary:hover{background:#0D9488;transform:translateY(-1px);box-shadow:0 8px 10px -1px rgba(15,118,110,.3);}
.btn-outline{background:var(--bg2);color:var(--txt2);border:1px solid var(--border2);box-shadow:var(--shadow);}
.btn-outline:hover{border-color:var(--teal);color:var(--teal);}
.requirements{margin-top:20px;display:flex;justify-content:center;gap:24px;flex-wrap:wrap;}
.req-item{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);display:flex;align-items:center;gap:6px;}
.req-dot{width:5px;height:5px;border-radius:50%;background:var(--teal);}
/* Progress */
.progress-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:28px;box-shadow:var(--shadow);display:none;}
.progress-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;}
.progress-title{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;}
.progress-bar-wrap{background:var(--bg3);border-radius:99px;height:8px;overflow:hidden;margin-bottom:12px;}
.progress-bar-fill{height:8px;background:linear-gradient(90deg,var(--teal),#10B981);border-radius:99px;width:0%;transition:width .4s ease;}
.progress-text{font-family:'DM Mono',monospace;font-size:12px;font-weight:500;color:var(--txt3);}
/* Results */
.results-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);display:none;}
.results-header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;background:var(--bg3);border-bottom:1px solid var(--border);}
.results-title{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;}
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:20px 24px;border-bottom:1px solid var(--border);}
.stat-mini{text-align:center;}
.stat-mini-val{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--teal);}
.stat-mini-lbl{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--txt3);margin-top:4px;}
.preview-wrap{padding:0 24px 20px;overflow-x:auto;}
.preview-wrap table{width:100%;border-collapse:collapse;font-size:13px;margin-top:16px;}
.preview-wrap th{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--txt3);padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap;}
.preview-wrap td{padding:10px 12px;border-bottom:1px solid var(--bg3);color:var(--txt2);font-weight:500;vertical-align:top;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.preview-wrap tr:last-child td{border-bottom:none;}
.preview-wrap tr:hover td{background:var(--bg3);}
.token{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;}
.tok-date{background:var(--blue-l);color:var(--blue);}
.tok-phone{background:var(--teal-l);color:var(--teal);}
.tok-mrn{background:var(--amber-l);color:var(--amber);}
.tok-hospital{background:rgba(124,58,237,0.1);color:#7C3AED;}
.tok-age{background:rgba(14,165,233,0.1);color:#0EA5E9;}
.tok-person{background:rgba(225,29,72,0.1);color:var(--red);}
.tok-default{background:var(--bg3);color:var(--txt3);}
.download-row{padding:20px 24px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:var(--bg3);}
.download-info{font-size:13px;font-weight:500;color:var(--txt3);}
.sample-template{margin-top:20px;background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.sample-header{padding:14px 18px;background:var(--bg3);border-bottom:1px solid var(--border);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--txt3);letter-spacing:.1em;text-transform:uppercase;display:flex;align-items:center;justify-content:space-between;}
.sample-body{padding:16px 18px;font-family:'DM Mono',monospace;font-size:12px;font-weight:500;color:var(--txt2);line-height:1.8;}
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
    <div class="nav-label">Tools</div>
    <a href="/api-explorer" class="nav-item">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/></svg>
      API Explorer
    </a>
    <a href="/upload" class="nav-item active">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 10V3M5 6l3-3 3 3" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11v2a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2" stroke-linecap="round"/></svg>
      Batch Upload
    </a>
    <a href="/system-status" class="nav-item">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2zm0 4v4m0 2v.5" stroke-linecap="round"/></svg>
      System Status
    </a>
  </div>
  <div class="sidebar-footer">
    <div class="status-badge"><div class="status-dot"></div>System Online</div>
  </div>
</nav>

<div class="main">
  <div class="topbar">
    <div>
      <div class="topbar-title">Batch De-identification</div>
      <div class="topbar-path">ClinicalNER / upload</div>
    </div>
    <div class="topbar-right">
      <span class="tag tag-teal">CSV Upload</span>
      <span class="tag tag-amber">Max 500 rows</span>
    </div>
  </div>

  <div class="content">
    <div class="section-hdr">
      <span class="section-title">Upload clinical notes</span>
      <div class="section-line"></div>
    </div>

    <!-- Drop zone -->
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <div class="drop-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div class="drop-title">Drop your CSV here</div>
      <div class="drop-sub">or click to browse — de-identifies all notes instantly</div>
      <button class="btn btn-primary" onclick="event.stopPropagation();document.getElementById('file-input').click()">Choose CSV file</button>
      <div class="requirements">
        <div class="req-item"><div class="req-dot"></div>Must have a 'text' or 'transcription' column</div>
        <div class="req-item"><div class="req-dot"></div>Maximum 500 rows</div>
        <div class="req-item"><div class="req-dot"></div>UTF-8 encoded CSV</div>
      </div>
    </div>
    <input type="file" id="file-input" accept=".csv" style="display:none" onchange="handleFile(this.files[0])">

    <!-- Progress -->
    <div class="progress-card" id="progress-card" style="margin-top:20px;">
      <div class="progress-header">
        <div class="progress-title" id="progress-title">Processing…</div>
        <span class="tag tag-teal" id="progress-tag">Running</span>
      </div>
      <div class="progress-bar-wrap"><div class="progress-bar-fill" id="progress-bar"></div></div>
      <div class="progress-text" id="progress-text">Uploading file…</div>
    </div>

    <!-- Results -->
    <div class="results-card" id="results-card" style="margin-top:20px;">
      <div class="results-header">
        <div class="results-title">De-identification complete</div>
        <span class="tag tag-teal">ICH E6 compliant</span>
      </div>
      <div class="stats-row">
        <div class="stat-mini">
          <div class="stat-mini-val" id="r-rows">—</div>
          <div class="stat-mini-lbl">Notes processed</div>
        </div>
        <div class="stat-mini">
          <div class="stat-mini-val" id="r-entities" style="color:var(--amber)">—</div>
          <div class="stat-mini-lbl">PHI entities masked</div>
        </div>
        <div class="stat-mini">
          <div class="stat-mini-val" id="r-valid" style="color:var(--blue)">—</div>
          <div class="stat-mini-lbl">Valid outputs</div>
        </div>
      </div>
      <div class="preview-wrap">
        <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--txt3);margin-bottom:4px;">Preview (first 5 rows)</div>
        <table id="preview-table"></table>
      </div>
      <div class="download-row">
        <div class="download-info" id="download-info">Ready to download</div>
        <button class="btn btn-primary" id="download-btn" onclick="downloadResult()">Download de-identified CSV</button>
      </div>
    </div>

    <!-- CSV template -->
    <div class="section-hdr" style="margin-top:36px;">
      <span class="section-title">Expected format</span>
      <div class="section-line"></div>
    </div>
    <div class="sample-template">
      <div class="sample-header">
        <span>sample_notes.csv</span>
        <button class="btn btn-outline" style="padding:6px 14px;font-size:10px;" onclick="downloadTemplate()">Download template</button>
      </div>
      <div class="sample-body">text,medical_specialty,note_id<br>
"Patient James Smith DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145.",Cardiology,1<br>
"Admitted to St. Mary's Hospital on 01/15/2024. Age: 45-year-old male.",Surgery,2<br>
"Follow-up at Memorial Medical Center on 03/01/2024. MRN: MRN456789.",Neurology,3</div>
    </div>
  </div>
</div>
</div>

<script>
let csvBlob = null;
let resultData = null;

const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.csv')) handleFile(file);
  else alert('Please drop a CSV file.');
});

function handleFile(file) {
  if (!file) return;
  document.getElementById('progress-card').style.display = 'block';
  document.getElementById('results-card').style.display  = 'none';
  document.getElementById('progress-title').textContent = `Processing ${file.name}`;
  document.getElementById('progress-text').textContent = 'Uploading and running NER pipeline…';
  document.getElementById('progress-bar').style.width = '15%';

  const formData = new FormData();
  formData.append('file', file);

  // Animate progress bar while waiting
  let pct = 15;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 3, 85);
    document.getElementById('progress-bar').style.width = pct + '%';
  }, 400);

  fetch('/api/upload', { method: 'POST', body: formData })
    .then(async res => {
      clearInterval(ticker);
      document.getElementById('progress-bar').style.width = '100%';

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Upload failed');
      }

      const totalRows     = res.headers.get('X-Total-Rows')     || '?';
      const totalEntities = res.headers.get('X-Total-Entities') || '?';
      csvBlob = await res.blob();

      document.getElementById('progress-tag').textContent = 'Complete';
      document.getElementById('progress-text').textContent = `${totalRows} notes processed · ${totalEntities} entities masked`;

      // Parse CSV for preview
      const text = await csvBlob.text();
      showResults(text, parseInt(totalRows), parseInt(totalEntities), file.name);
    })
    .catch(err => {
      clearInterval(ticker);
      document.getElementById('progress-tag').textContent = 'Error';
      document.getElementById('progress-tag').style.background = 'rgba(225,29,72,0.1)';
      document.getElementById('progress-tag').style.color = '#E11D48';
      document.getElementById('progress-text').textContent = 'Error: ' + err.message;
      document.getElementById('progress-bar').style.background = '#E11D48';
    });
}

function showResults(csvText, totalRows, totalEntities, filename) {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.replace(/"/g,'').trim());

  // Count valid rows
  const maskedIdx = headers.indexOf('masked_text');
  const validIdx  = headers.indexOf('is_valid');
  let validCount  = 0;
  lines.slice(1).forEach(line => {
    if (line.includes('True') || line.includes('true')) validCount++;
  });

  document.getElementById('r-rows').textContent     = totalRows || lines.length - 1;
  document.getElementById('r-entities').textContent = totalEntities;
  document.getElementById('r-valid').textContent    = validCount;
  document.getElementById('download-info').textContent = `${filename} → deidentified_${filename}`;

  // Build preview table (first 5 data rows, key columns only)
  const SHOW_COLS = ['text','masked_text','entity_count','entity_types','is_valid'];
  const colIdxs   = SHOW_COLS.map(c => headers.indexOf(c)).filter(i => i >= 0);
  const showHdrs  = colIdxs.map(i => headers[i]);

  let tableHtml = '<tr>' + showHdrs.map(h => `<th>${h}</th>`).join('') + '</tr>';

  lines.slice(1, 6).forEach(line => {
    const cells = parseCsvLine(line);
    tableHtml += '<tr>' + colIdxs.map(i => {
      let val = (cells[i] || '').replace(/^"|"$/g,'');
      if (headers[i] === 'masked_text') val = highlightTokens(val);
      else val = escHtml(val.slice(0, 60) + (val.length > 60 ? '…' : ''));
      return `<td title="${escHtml(cells[i]||'')}">${val}</td>`;
    }).join('') + '</tr>';
  });

  document.getElementById('preview-table').innerHTML = tableHtml;
  document.getElementById('results-card').style.display = 'block';
}

function parseCsvLine(line) {
  const result = []; let cur = ''; let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQ = !inQ; continue; }
    if (ch === ',' && !inQ) { result.push(cur); cur = ''; continue; }
    cur += ch;
  }
  result.push(cur);
  return result;
}

const TOK_CLASS = {DATE:'tok-date',DOB:'tok-date',PHONE:'tok-phone',MRN:'tok-mrn',
  HOSPITAL:'tok-hospital',AGE:'tok-age',PERSON:'tok-person'};

function highlightTokens(text) {
  return escHtml(text).replace(/\[([A-Z]+)\]/g, (m, label) =>
    `<span class="token ${TOK_CLASS[label]||'tok-default'}">${m}</span>`);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function downloadResult() {
  if (!csvBlob) return;
  const url = URL.createObjectURL(csvBlob);
  const a   = document.createElement('a');
  a.href = url; a.download = 'deidentified_output.csv'; a.click();
  URL.revokeObjectURL(url);
}

function downloadTemplate() {
  const csv = `text,medical_specialty,note_id\n"Patient James Smith DOB: 04/12/1985. Phone: (415) 555-9876. MRN302145.",Cardiology,1\n"Admitted to St. Mary's Hospital on 01/15/2024. Age: 45-year-old male.",Surgery,2\n"Follow-up at Memorial Medical Center on 03/01/2024. MRN: MRN456789.",Neurology,3`;
  const blob = new Blob([csv], {type:'text/csv'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a'); a.href=url; a.download='sample_notes.csv'; a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""