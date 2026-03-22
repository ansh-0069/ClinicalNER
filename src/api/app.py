"""
app.py
â”€â”€â”€â”€â”€â”€
Flask application factory for ClinicalNER.

Routes
------
  POST /api/deidentify          â€” de-identify a clinical note (core API)
  GET  /api/note/<id>           â€” fetch a processed note by ID
  GET  /api/stats               â€” corpus + pipeline statistics (JSON)
  GET  /dashboard               â€” live EDA + audit dashboard (HTML)
  GET  /report/<note_id>        â€” before/after diff view (HTML)
  GET  /health                  â€” liveness probe (for Docker/cloud)

Design decisions:
  - Application factory pattern (create_app()) â€” standard Flask practice,
    makes the app testable and importable without side effects.
  - All pipeline objects initialised ONCE at app startup via app.config,
    not on every request â€” avoids reloading spaCy model per call.
  - Errors return JSON with a consistent shape:
    {"error": "...", "status": 400} â€” matches what a frontend expects.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from ipaddress import ip_address, ip_network
from pathlib import Path

from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_loader import DataLoader
from src.pipeline.ner_pipeline import NERPipeline
from src.pipeline.data_cleaner import DataCleaner
from src.pipeline.audit_logger import AuditLogger, EventType
from src.pipeline.anomaly_detector import AnomalyDetector
from src.pipeline.readmission_predictor import ReadmissionPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# â”€â”€ Application factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def create_app(db_path: str | None = None) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    db_path : path to SQLite database (override for testing)
    """
    # Resolve DB path from function arg first, then environment, then fallback.
    resolved_db_path = db_path or os.getenv("DB_PATH", "data/clinicalner.db")

    # Set template folder to absolute path
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir), static_folder="static")
    CORS(app)   # allow cross-origin requests (needed when frontend is separate)

    # â”€â”€ Initialise pipeline components once at startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Key decision: store in app.config so all request handlers share
    # the same instances â€” avoids reloading the spaCy model on every request.
    app.config["DB_PATH"]  = resolved_db_path
    app.config["LOADER"]   = DataLoader(db_path=resolved_db_path)
    app.config["PIPELINE"] = NERPipeline(db_path=resolved_db_path, use_spacy=True)
    app.config["CLEANER"]  = DataCleaner(strict_mode=False)
    app.config["AUDIT"]    = AuditLogger(db_path=resolved_db_path)
    app.config["DETECTOR"] = AnomalyDetector(contamination=0.05)
    app.config["PREDICTOR"] = ReadmissionPredictor()
    app.config["BACKFILL_JOBS"] = {}
    app.config["BACKFILL_LOCK"] = threading.Lock()
    app.config["BACKFILL_ACTIVE_JOB_ID"] = None
    app.config["RATE_LIMIT_BUCKETS"] = defaultdict(deque)
    app.config["RATE_LIMIT_LOCK"] = threading.Lock()
    app.config["APP_STARTED_AT"] = datetime.now(timezone.utc).isoformat()

    @app.before_request
    def _attach_request_context() -> None:
      g.request_id = uuid.uuid4().hex[:12]
      g.request_started_at = time.perf_counter()

    @app.after_request
    def _emit_request_observability(response):
      request_id = getattr(g, "request_id", "unknown")
      response.headers["X-Request-ID"] = request_id

      started = getattr(g, "request_started_at", None)
      latency_ms = round((time.perf_counter() - started) * 1000, 2) if started else -1
      logger.info(
        "req_id=%s method=%s path=%s status=%s latency_ms=%s ip=%s",
        request_id,
        request.method,
        request.path,
        response.status_code,
        latency_ms,
        request.headers.get("X-Forwarded-For", request.remote_addr),
      )
      return response

    logger.info("ClinicalNER Flask app initialised | db=%s", resolved_db_path)

    # â”€â”€ Register routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _register_api_routes(app)
    _register_ui_routes(app)

    from src.api.routes import api_bp

    app.register_blueprint(api_bp)

    return app

def _build_report_summary_payload(loader):
    raise NotImplementedError


# â”€â”€ API routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _register_api_routes(app: Flask) -> None:

  def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

  def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded:
      return forwarded.split(",")[0].strip()
    return (request.remote_addr or "").strip()

  def _is_ip_allowed(ip_str: str, cidr_list_raw: str) -> bool:
    if not cidr_list_raw.strip():
      return True
    try:
      addr = ip_address(ip_str)
      for cidr in [c.strip() for c in cidr_list_raw.split(",") if c.strip()]:
        if addr in ip_network(cidr, strict=False):
          return True
      return False
    except Exception:
      return False

  def _check_rate_limit(route_key: str, limit: int, window_sec: int) -> tuple[bool, tuple[dict, int] | None]:
    buckets = app.config["RATE_LIMIT_BUCKETS"]
    lock: threading.Lock = app.config["RATE_LIMIT_LOCK"]
    now = time.time()
    ip = _client_ip() or "unknown"
    bucket_key = f"{route_key}:{ip}"

    with lock:
      q: deque = buckets[bucket_key]
      cutoff = now - window_sec
      while q and q[0] < cutoff:
        q.popleft()

      if len(q) >= limit:
        retry_after = int(max(1, window_sec - (now - q[0])))
        return False, ({
          "error": f"Rate limit exceeded for {route_key}. Retry in {retry_after}s",
          "status": 429,
        }, 429)

      q.append(now)
    return True, None

  def _ensure_review_table(loader: DataLoader) -> None:
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_decisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          note_id INTEGER NOT NULL,
          decision TEXT NOT NULL,
          reviewer TEXT,
          comments TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
      )

  def _ensure_backfill_job_table() -> None:
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backfill_jobs (
          job_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          processed_notes INTEGER NOT NULL DEFAULT 0,
          total_notes INTEGER NOT NULL DEFAULT 0,
          notes_with_phi INTEGER NOT NULL DEFAULT 0,
          total_entities INTEGER NOT NULL DEFAULT 0,
          cleared_previous_rows INTEGER NOT NULL DEFAULT 0,
          progress_pct REAL NOT NULL DEFAULT 0,
          clear_existing INTEGER NOT NULL DEFAULT 1,
          job_limit INTEGER,
          error TEXT
        )
        """
      )
      conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backfill_jobs_created_at ON backfill_jobs(created_at DESC)"
      )

  def _row_to_backfill_job(row: sqlite3.Row | None) -> dict | None:
    if row is None:
      return None
    return {
      "job_id": row["job_id"],
      "status": row["status"],
      "created_at": row["created_at"],
      "started_at": row["started_at"],
      "finished_at": row["finished_at"],
      "processed_notes": int(row["processed_notes"] or 0),
      "total_notes": int(row["total_notes"] or 0),
      "notes_with_phi": int(row["notes_with_phi"] or 0),
      "total_entities": int(row["total_entities"] or 0),
      "cleared_previous_rows": int(row["cleared_previous_rows"] or 0),
      "progress_pct": float(row["progress_pct"] or 0.0),
      "clear_existing": bool(int(row["clear_existing"] or 0)),
      "limit": int(row["job_limit"]) if row["job_limit"] is not None else None,
      "error": row["error"],
    }

  def _create_backfill_job(clear_existing: bool, limit: int | None) -> tuple[dict | None, dict | None]:
    _ensure_backfill_job_table()
    job_id = uuid.uuid4().hex
    created_at = _utc_now()

    with sqlite3.connect(app.config["DB_PATH"], timeout=30) as conn:
      conn.row_factory = sqlite3.Row
      conn.execute("BEGIN IMMEDIATE")

      active_row = conn.execute(
        """
        SELECT * FROM backfill_jobs
        WHERE status IN ('queued', 'running')
        ORDER BY created_at DESC
        LIMIT 1
        """
      ).fetchone()

      if active_row:
        conn.rollback()
        return None, _row_to_backfill_job(active_row)

      conn.execute(
        """
        INSERT INTO backfill_jobs (
          job_id, status, created_at, started_at, finished_at,
          processed_notes, total_notes, notes_with_phi, total_entities,
          cleared_previous_rows, progress_pct, clear_existing, job_limit, error
        ) VALUES (?, 'queued', ?, NULL, NULL, 0, 0, 0, 0, 0, 0.0, ?, ?, NULL)
        """,
        (job_id, created_at, int(bool(clear_existing)), limit),
      )

      row = conn.execute(
        "SELECT * FROM backfill_jobs WHERE job_id = ?",
        (job_id,),
      ).fetchone()

    return _row_to_backfill_job(row), None

  def _update_backfill_job(job_id: str, **fields) -> None:
    if not fields:
      return

    db_field_map = {
      "status": "status",
      "started_at": "started_at",
      "finished_at": "finished_at",
      "processed_notes": "processed_notes",
      "total_notes": "total_notes",
      "notes_with_phi": "notes_with_phi",
      "total_entities": "total_entities",
      "cleared_previous_rows": "cleared_previous_rows",
      "progress_pct": "progress_pct",
      "clear_existing": "clear_existing",
      "limit": "job_limit",
      "error": "error",
    }

    assignments = []
    values = []
    for key, value in fields.items():
      if key not in db_field_map:
        continue
      column = db_field_map[key]
      if key == "clear_existing" and value is not None:
        value = int(bool(value))
      assignments.append(f"{column} = ?")
      values.append(value)

    if not assignments:
      return

    _ensure_backfill_job_table()
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
      conn.execute(
        f"UPDATE backfill_jobs SET {', '.join(assignments)} WHERE job_id = ?",
        (*values, job_id),
      )

  def _get_backfill_job(job_id: str) -> dict | None:
    _ensure_backfill_job_table()
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
      conn.row_factory = sqlite3.Row
      row = conn.execute(
        "SELECT * FROM backfill_jobs WHERE job_id = ? LIMIT 1",
        (job_id,),
      ).fetchone()
    return _row_to_backfill_job(row)

  def _get_latest_backfill_job() -> dict | None:
    _ensure_backfill_job_table()
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
      conn.row_factory = sqlite3.Row
      row = conn.execute(
        "SELECT * FROM backfill_jobs ORDER BY created_at DESC LIMIT 1"
      ).fetchone()
    return _row_to_backfill_job(row)

  def _table_exists(table_name: str) -> bool:
    try:
      with sqlite3.connect(app.config["DB_PATH"]) as conn:
        row = conn.execute(
          "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
          (table_name,),
        ).fetchone()
      return bool(row)
    except Exception:
      return False

  def _normalize_admin_token(value: str) -> str:
    token = str(value or "").strip()
    # Azure/GitHub app settings sometimes end up wrapped in quotes.
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
      token = token[1:-1].strip()
    return token

  def _auth_admin_token() -> tuple[bool, str, tuple[dict, int] | None]:
    configured_tokens: list[tuple[str, str]] = []
    for env_name in ("ADMIN_BACKFILL_TOKEN", "BACKFILL_ADMIN_TOKEN", "ADMIN_TOKEN"):
      candidate = _normalize_admin_token(os.getenv(env_name, ""))
      if candidate:
        configured_tokens.append((env_name, candidate))

    if not configured_tokens:
      return False, "", ({
        "error": "Backfill endpoint disabled (set ADMIN_BACKFILL_TOKEN)",
        "status": 503,
      }, 503)

    # Accept multiple auth transports to tolerate proxy/header normalization.
    provided_token = _normalize_admin_token(request.headers.get("X-Admin-Token", ""))
    if not provided_token:
      auth_header = _normalize_admin_token(request.headers.get("Authorization", ""))
      if auth_header.lower().startswith("bearer "):
        provided_token = _normalize_admin_token(auth_header[7:])
    if not provided_token:
      provided_token = _normalize_admin_token(request.headers.get("X-API-Key", ""))
    if not provided_token and request.method in {"POST", "PUT", "PATCH"}:
      payload = request.get_json(silent=True) or {}
      provided_token = _normalize_admin_token(payload.get("admin_token", ""))

    if not provided_token:
      return False, "", ({"error": "Unauthorized: admin token missing", "status": 401}, 401)

    matched_source = ""
    for env_name, configured_token in configured_tokens:
      if secrets.compare_digest(provided_token, configured_token):
        matched_source = env_name
        break

    if not matched_source:
      logger.warning(
        "Admin backfill token mismatch | sources=%s provided_len=%s configured_lens=%s",
        ",".join(name for name, _ in configured_tokens),
        len(provided_token),
        ",".join(str(len(token)) for _, token in configured_tokens),
      )
      return False, "", ({"error": "Unauthorized: invalid admin token", "status": 401}, 401)

    # Optional hardening: enforce an admin user header.
    require_user_header = os.getenv("ADMIN_REQUIRE_USER_HEADER", "false").strip().lower() in {"1", "true", "yes"}
    if require_user_header and not request.headers.get("X-Admin-User", "").strip():
      return False, "", ({"error": "Missing X-Admin-User header", "status": 401}, 401)

    # Optional hardening: allow only known CIDRs.
    allow_cidrs = os.getenv("ADMIN_ALLOWLIST_CIDRS", "")
    ip = _client_ip()
    if not _is_ip_allowed(ip, allow_cidrs):
      return False, "", ({"error": "Forbidden: source IP not in allowlist", "status": 403}, 403)

    return True, provided_token, None

  def _get_backfill_payload() -> tuple[dict, tuple[dict, int] | None]:
    payload = request.get_json(silent=True) or {}
    clear_existing = payload.get("clear_existing", True)
    limit = payload.get("limit")

    if limit is not None:
      try:
        limit = int(limit)
        if limit <= 0:
          raise ValueError("limit must be > 0")
      except Exception:
        return {}, ({"error": "Field 'limit' must be a positive integer", "status": 400}, 400)

    return {
      "clear_existing": bool(clear_existing),
      "limit": limit,
    }, None

  def _run_backfill_job(job_id: str, clear_existing: bool, limit: int | None) -> None:
    loader: DataLoader = app.config["LOADER"]
    pipeline: NERPipeline = app.config["PIPELINE"]
    audit: AuditLogger = app.config["AUDIT"]
    jobs: dict = app.config["BACKFILL_JOBS"]
    lock: threading.Lock = app.config["BACKFILL_LOCK"]

    with lock:
      job = jobs.get(job_id, {"job_id": job_id})
      job["status"] = "running"
      job["started_at"] = _utc_now()
      jobs[job_id] = job

    _update_backfill_job(
      job_id,
      status="running",
      started_at=job["started_at"],
      error=None,
    )

    try:
      deleted_rows = 0
      if clear_existing:
        with sqlite3.connect(app.config["DB_PATH"]) as conn:
          try:
            cur = conn.execute("SELECT COUNT(*) FROM processed_notes")
            deleted_rows = int(cur.fetchone()[0] or 0)
            conn.execute("DELETE FROM processed_notes")
          except sqlite3.OperationalError:
            deleted_rows = 0

      sql = "SELECT note_id, transcription FROM clinical_notes ORDER BY note_id"
      if limit is not None:
        sql += f" LIMIT {limit}"

      notes_df = loader.sql_query(sql)
      total_notes = int(len(notes_df))
      if notes_df.empty:
        _update_backfill_job(
          job_id,
          status="failed",
          error="No clinical_notes found to process",
          finished_at=_utc_now(),
        )
        with lock:
          job["status"] = "failed"
          job["error"] = "No clinical_notes found to process"
          job["finished_at"] = _utc_now()
        return

      notes = notes_df.to_dict(orient="records")
      audit.log(
        EventType.PIPELINE_START,
        description=f"Admin backfill started | notes={len(notes)} | clear_existing={clear_existing}",
        metadata={"limit": limit, "clear_existing": clear_existing, "job_id": job_id},
      )

      processed = 0
      total_entities = 0
      notes_with_phi = 0

      for item in notes:
        text = str(item.get("transcription") or "")
        note_id = item.get("note_id")
        result = pipeline.process_note(text, note_id=note_id, save_to_db=True)

        count = int(result.get("entity_count", 0) or 0)
        total_entities += count
        if count > 0:
          notes_with_phi += 1

        processed += 1
        if processed % 10 == 0 or processed == total_notes:
          progress_pct = round((processed / max(total_notes, 1)) * 100, 1)
          _update_backfill_job(
            job_id,
            processed_notes=processed,
            total_notes=total_notes,
            total_entities=total_entities,
            notes_with_phi=notes_with_phi,
            progress_pct=progress_pct,
          )
          with lock:
            job["processed_notes"] = processed
            job["total_notes"] = total_notes
            job["total_entities"] = total_entities
            job["notes_with_phi"] = notes_with_phi
            job["progress_pct"] = progress_pct

      finished_at = _utc_now()
      _update_backfill_job(
        job_id,
        status="completed",
        processed_notes=processed,
        total_notes=total_notes,
        notes_with_phi=notes_with_phi,
        total_entities=total_entities,
        cleared_previous_rows=deleted_rows,
        progress_pct=100.0,
        finished_at=finished_at,
      )
      with lock:
        job["status"] = "completed"
        job["processed_notes"] = processed
        job["total_notes"] = total_notes
        job["notes_with_phi"] = notes_with_phi
        job["total_entities"] = total_entities
        job["cleared_previous_rows"] = deleted_rows
        job["progress_pct"] = 100.0
        job["finished_at"] = finished_at

      audit.log(
        EventType.PIPELINE_COMPLETE,
        description="Admin backfill completed",
        metadata={
          "job_id": job_id,
          "processed_notes": processed,
          "notes_with_phi": notes_with_phi,
          "total_entities": total_entities,
          "cleared_previous_rows": deleted_rows,
        },
      )
    except Exception as e:
      finished_at = _utc_now()
      _update_backfill_job(
        job_id,
        status="failed",
        error=str(e),
        finished_at=finished_at,
      )
      with lock:
        jobs.setdefault(job_id, {"job_id": job_id})
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["finished_at"] = finished_at
      audit.log(EventType.ERROR, description=f"Admin backfill failed [{job_id}]: {e}")
      logger.error("admin_backfill job failed [%s]: %s", job_id, e)
    finally:
      with lock:
        if app.config.get("BACKFILL_ACTIVE_JOB_ID") == job_id:
          app.config["BACKFILL_ACTIVE_JOB_ID"] = None

  @app.route("/health")
  def health():
      """Liveness probe â€” Docker HEALTHCHECK and cloud load balancers call this."""
      return jsonify({
        "status": "ok",
        "service": "ClinicalNER",
        "app_version": os.getenv("APP_VERSION", "unknown"),
        "started_at": app.config.get("APP_STARTED_AT"),
      })

  @app.route("/api/deidentify", methods=["POST"])
  def deidentify():
      """
      De-identify a clinical note.

      Request body (JSON):
        {
          "text":     "Patient James Smith...",   â† required
          "note_id":  42,                          â† optional
          "save":     true                         â† optional, default true
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

      ok, rate_error = _check_rate_limit("deidentify", limit=120, window_sec=60)
      if not ok:
        return jsonify(rate_error[0]), rate_error[1]

      # â”€â”€ Input validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

      # â”€â”€ Log API request â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      audit.log(
          EventType.API_REQUEST,
          description=f"POST /api/deidentify | note_id={note_id} | chars={len(text)}",
          note_id=note_id,
      )

      # â”€â”€ Pre-NER clean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      pre_result = cleaner.clean_pre_ner(text)

      # â”€â”€ NER + masking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      ner_result = pipeline.process_note(
          pre_result.cleaned_text,
          note_id=note_id,
          save_to_db=save,
      )

      # â”€â”€ Post-NER clean + validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      post_result = cleaner.clean_post_ner(ner_result["masked_text"])

      # â”€â”€ Log NER result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      audit.log_ner_result(ner_result)
      audit.log_cleaning_result(post_result, note_id=note_id)

      # â”€â”€ Build response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
          # Note counts â€” return 0 gracefully if tables don't exist yet
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

          # Risk and confidence telemetry for UI triage.
          low_confidence_count = 0
          high_risk_count = 0
          avg_confidence = 0.0

          try:
              risk_df = loader.sql_query(
                  "SELECT entity_count FROM processed_notes WHERE entity_count IS NOT NULL"
              )
              if not risk_df.empty:
                  high_risk_count = int((risk_df["entity_count"].astype(float) >= 8).sum())
          except Exception:
              pass

          # avg_confidence is optional depending on DB migration state.
          try:
              conf_df = loader.sql_query(
                  "SELECT avg_confidence FROM processed_notes WHERE avg_confidence IS NOT NULL"
              )
              if not conf_df.empty:
                  conf_series = conf_df["avg_confidence"].astype(float)
                  avg_confidence = float(round(conf_series.mean(), 4))
                  low_confidence_count = int((conf_series < 0.70).sum())
          except Exception:
              pass

          return jsonify({
              "note_count":       int(note_count),
              "processed_count":  int(processed_count),
              "entity_totals":    entity_totals,
              "specialty":        specialty,
              "phi_by_specialty": phi_by_spec,
              "audit_summary":    audit_summary,
              "total_audit_events": audit.total_events(),
              "avg_confidence":   avg_confidence,
              "low_confidence_count": low_confidence_count,
              "high_risk_count":  high_risk_count,
          }), 200

      except Exception as e:
          logger.error("stats error: %s", e)
          return jsonify({"error": str(e), "status": 500}), 500

  @app.route("/api/review/queue", methods=["GET"])
  def review_queue():
      """Return high-risk processed notes pending manual review."""
      loader: DataLoader = app.config["LOADER"]
      _ensure_review_table(loader)

      if not _table_exists("processed_notes"):
        return jsonify({
          "status": 200,
          "count": 0,
          "items": [],
          "message": "processed_notes table not found yet. Run processing/backfill first.",
        }), 200

      limit_raw = request.args.get("limit", "50")
      try:
          limit = max(1, min(int(limit_raw), 200))
      except Exception:
          limit = 50

      try:
          rows = loader.sql_query(
              f"""
              SELECT
                pn.note_id,
                pn.masked_text,
                pn.entity_count,
                pn.processed_at
              FROM processed_notes pn
              LEFT JOIN review_decisions rd ON rd.note_id = pn.note_id
              WHERE pn.entity_count >= 8
                AND rd.note_id IS NULL
              ORDER BY pn.entity_count DESC, pn.processed_at DESC
              LIMIT {limit}
              """
          )
      except Exception as e:
          return jsonify({"error": f"Unable to load review queue: {e}", "status": 500}), 500

      return jsonify({
          "status": 200,
          "count": int(len(rows)),
          "items": rows.to_dict(orient="records"),
      }), 200

  @app.route("/api/review/decision", methods=["POST"])
  def review_decision():
      """Submit a manual review decision for a note."""
      loader: DataLoader = app.config["LOADER"]
      _ensure_review_table(loader)

      if not request.is_json:
          return jsonify({"error": "Request must be JSON", "status": 400}), 400

      payload = request.get_json(silent=True) or {}
      note_id = payload.get("note_id")
      decision = str(payload.get("decision", "")).strip().lower()
      reviewer = str(payload.get("reviewer", "")).strip() or "anonymous"
      comments = str(payload.get("comments", "")).strip()

      if not note_id:
          return jsonify({"error": "Field 'note_id' is required", "status": 400}), 400
      if decision not in {"approved", "edited", "rejected"}:
          return jsonify({"error": "Field 'decision' must be approved|edited|rejected", "status": 400}), 400

      try:
          with sqlite3.connect(app.config["DB_PATH"]) as conn:
              conn.execute(
                  "INSERT INTO review_decisions (note_id, decision, reviewer, comments) VALUES (?, ?, ?, ?)",
                  (int(note_id), decision, reviewer, comments),
              )
      except Exception as e:
          return jsonify({"error": f"Unable to save review decision: {e}", "status": 500}), 500

      return jsonify({"status": 201, "message": "Review decision recorded"}), 201

  @app.route("/api/review/summary", methods=["GET"])
  def review_summary():
      """Aggregate review queue and decisions for dashboard cards."""
      loader: DataLoader = app.config["LOADER"]
      _ensure_review_table(loader)

      try:
        if _table_exists("processed_notes"):
          pending_df = loader.sql_query(
            """
            SELECT COUNT(*) AS n
            FROM processed_notes pn
            LEFT JOIN review_decisions rd ON rd.note_id = pn.note_id
            WHERE pn.entity_count >= 8 AND rd.note_id IS NULL
            """
          )
        else:
          pending_df = None
        decided_df = loader.sql_query(
            "SELECT decision, COUNT(*) AS n FROM review_decisions GROUP BY decision"
        )
      except Exception as e:
          return jsonify({"error": f"Unable to build review summary: {e}", "status": 500}), 500

      by_decision = {r["decision"]: int(r["n"]) for _, r in decided_df.iterrows()} if not decided_df.empty else {}
      return jsonify({
          "status": 200,
          "pending": int(pending_df.iloc[0]["n"] if pending_df is not None and not pending_df.empty else 0),
          "by_decision": by_decision,
      }), 200

  @app.route("/api/drift/summary", methods=["GET"])
  def drift_summary():
      """Simple drift monitor comparing recent vs baseline processed-note windows."""
      loader: DataLoader = app.config["LOADER"]
      window = 100

      if not _table_exists("processed_notes"):
        return jsonify({
          "status": 200,
          "message": "processed_notes table not found yet. Run processing/backfill first.",
          "window": window,
          "drift_pct": 0.0,
          "drift_state": "insufficient-data",
        }), 200

      try:
          recent = loader.sql_query(
              f"SELECT entity_count FROM processed_notes ORDER BY processed_at DESC LIMIT {window}"
          )
          baseline = loader.sql_query(
              f"SELECT entity_count FROM processed_notes ORDER BY processed_at DESC LIMIT {window} OFFSET {window}"
          )
      except Exception as e:
          return jsonify({"error": f"Unable to compute drift: {e}", "status": 500}), 500

      if recent.empty or baseline.empty:
          return jsonify({
              "status": 200,
              "message": "Insufficient processed notes for drift analysis",
              "window": window,
              "drift_pct": 0.0,
              "drift_state": "insufficient-data",
          }), 200

      recent_avg = float(recent["entity_count"].astype(float).mean())
      baseline_avg = float(baseline["entity_count"].astype(float).mean())
      drift_pct = ((recent_avg - baseline_avg) / max(baseline_avg, 1e-9)) * 100

      if abs(drift_pct) >= 20:
          state = "high"
      elif abs(drift_pct) >= 10:
          state = "medium"
      else:
          state = "normal"

      return jsonify({
          "status": 200,
          "window": window,
          "recent_avg_entities": round(recent_avg, 3),
          "baseline_avg_entities": round(baseline_avg, 3),
          "drift_pct": round(drift_pct, 3),
          "drift_state": state,
      }), 200

  @app.route("/api/export/compliance-pack", methods=["GET"])
  def export_compliance_pack():
      """One-shot JSON compliance export bundle for sponsor/CRO handoff."""
      loader: DataLoader = app.config["LOADER"]
      audit: AuditLogger = app.config["AUDIT"]

      report = _build_report_summary_payload(loader)

      _ensure_review_table(loader)
      try:
          if _table_exists("processed_notes"):
            pending_df = loader.sql_query(
              """
              SELECT COUNT(*) AS n
              FROM processed_notes pn
              LEFT JOIN review_decisions rd ON rd.note_id = pn.note_id
              WHERE pn.entity_count >= 8 AND rd.note_id IS NULL
              """
            )
          else:
            pending_df = None
          decided_df = loader.sql_query(
              "SELECT decision, COUNT(*) AS n FROM review_decisions GROUP BY decision"
          )
          review = {
            "pending": int(pending_df.iloc[0]["n"] if pending_df is not None and not pending_df.empty else 0),
              "by_decision": {r["decision"]: int(r["n"]) for _, r in decided_df.iterrows()} if not decided_df.empty else {},
          }
      except Exception:
          review = {"pending": 0, "by_decision": {}}

      payload = {
          "generated_at": datetime.now(timezone.utc).isoformat(),
          "compliance_frameworks": ["HIPAA Safe Harbor", "ICH E6 (R2)", "21 CFR Part 11 readiness"],
          "report_summary": report,
          "audit_total_events": int(audit.total_events()),
          "audit_summary": audit.get_summary().to_dict(orient="records"),
          "review_summary": review,
      }
      return jsonify(payload), 200

  @app.route("/api/admin/backfill-processed", methods=["POST"])
  def admin_backfill_processed():
      """
      Admin-only one-shot backfill for processed_notes.

      Auth:
        Header X-Admin-Token must match env ADMIN_BACKFILL_TOKEN.

      Request body (optional JSON):
        {
          "clear_existing": true,   # default true
          "limit": 500              # optional positive integer
        }
      """
      ok, _, auth_error = _auth_admin_token()
      if not ok:
        return jsonify(auth_error[0]), auth_error[1]

      payload, payload_error = _get_backfill_payload()
      if payload_error:
        return jsonify(payload_error[0]), payload_error[1]

      job, active_job = _create_backfill_job(
        clear_existing=payload["clear_existing"],
        limit=payload["limit"],
      )
      if active_job:
        return jsonify({
          "error": "A backfill job is already running",
          "status": 409,
          "job": active_job,
        }), 409
      if not job:
        return jsonify({"error": "Failed to create backfill job", "status": 500}), 500

      jobs: dict = app.config["BACKFILL_JOBS"]
      lock: threading.Lock = app.config["BACKFILL_LOCK"]
      with lock:
        jobs[job["job_id"]] = dict(job)
        app.config["BACKFILL_ACTIVE_JOB_ID"] = job["job_id"]

      worker = threading.Thread(
        target=_run_backfill_job,
        args=(job["job_id"], payload["clear_existing"], payload["limit"]),
        daemon=True,
        name=f"backfill-{job['job_id'][:8]}",
      )
      worker.start()

      return jsonify({
        "status": 202,
        "message": "Backfill job queued",
        "job": job,
      }), 202

  @app.route("/api/admin/backfill-status/<job_id>", methods=["GET"])
  def admin_backfill_status(job_id: str):
      """Return status for a previously created admin backfill job."""
      ok, rate_error = _check_rate_limit("admin_backfill_status", limit=240, window_sec=60)
      if not ok:
          return jsonify(rate_error[0]), rate_error[1]

      ok, _, auth_error = _auth_admin_token()
      if not ok:
          return jsonify(auth_error[0]), auth_error[1]

      job = _get_backfill_job(job_id)
      if not job:
          jobs: dict = app.config["BACKFILL_JOBS"]
          job = jobs.get(job_id)
      if not job:
          return jsonify({"error": "Backfill job not found", "status": 404}), 404
      return jsonify({"status": 200, "job": job}), 200

  @app.route("/api/admin/backfill-status", methods=["GET"])
  def admin_backfill_status_latest():
      """Return latest backfill job status if available."""
      ok, rate_error = _check_rate_limit("admin_backfill_status", limit=240, window_sec=60)
      if not ok:
          return jsonify(rate_error[0]), rate_error[1]

      ok, _, auth_error = _auth_admin_token()
      if not ok:
          return jsonify(auth_error[0]), auth_error[1]

      latest = _get_latest_backfill_job()
      if not latest:
          jobs: dict = app.config["BACKFILL_JOBS"]
          if not jobs:
              return jsonify({"status": 200, "job": None}), 200
          latest = max(jobs.values(), key=lambda j: j.get("created_at") or "")
      return jsonify({"status": 200, "job": latest}), 200

  @app.route("/api/predict-readmission", methods=["POST"])
  def predict_readmission():
      predictor: ReadmissionPredictor = app.config["PREDICTOR"]
      loader:    DataLoader           = app.config["LOADER"]

      ok, rate_error = _check_rate_limit("predict_readmission", limit=30, window_sec=60)
      if not ok:
          return jsonify(rate_error[0]), rate_error[1]

      if not request.is_json:
          return jsonify({"error": "Request must be JSON", "status": 400}), 400

      data = request.get_json()

      if not predictor.is_fitted:
          try:
              import json as _json
              df = loader.sql_query(
                  "SELECT note_id, masked_text, entity_types_json, avg_confidence "
                  "FROM processed_notes WHERE entity_types_json IS NOT NULL LIMIT 2000"
              )
              if len(df) < 50:
                  return jsonify({"error": "Need at least 50 processed notes. Run run_phase2.py first.", "status": 400}), 400

              corpus = []
              for _, row in df.iterrows():
                  try:
                      et = _json.loads(row.get("entity_types_json") or "{}")
                      entities = [{"label": k} for k, v in et.items() for _ in range(int(v))]
                  except Exception:
                      entities = []
                  corpus.append({
                      "id":       int(row["note_id"]),
                      "text":     str(row.get("masked_text") or ""),
                      "entities": entities,
                  })
              predictor.fit(corpus)
          except Exception as e:
              return jsonify({"error": f"Model training failed: {e}", "status": 500}), 500

      try:
          if "notes" in data:
              results = predictor.predict_batch(data["notes"])
              return jsonify({
                  "results":     [r.to_dict() for r in results],
                  "model_stats": predictor.model_stats(),
                  "count":       len(results),
              }), 200
          else:
              result = predictor.predict_one(data)
              return jsonify({**result.to_dict(), "model_stats": predictor.model_stats()}), 200
      except Exception as e:
          return jsonify({"error": str(e), "status": 500}), 500


  @app.route("/api/anomaly-scan", methods=["POST"])
  def anomaly_scan():
      """Fit IsolationForest on submitted notes, return anomaly scores."""
      detector: AnomalyDetector = app.config["DETECTOR"]
      audit:    AuditLogger     = app.config["AUDIT"]

      ok, rate_error = _check_rate_limit("anomaly_scan", limit=20, window_sec=60)
      if not ok:
          return jsonify(rate_error[0]), rate_error[1]

      if not request.is_json:
          return jsonify({"error": "Request must be JSON", "status": 400}), 400

      payload = request.get_json(silent=True) or {}
      notes_in = payload.get("notes", [])

      if not isinstance(notes_in, list):
          return jsonify({"error": "Field 'notes' must be a JSON array", "status": 400}), 400

      # Accept both list[str] and list[dict] to make the API Explorer
      # and external clients easier to use.
      notes: list[dict] = []
      for idx, item in enumerate(notes_in):
          if isinstance(item, str):
              notes.append({"id": idx + 1, "text": item, "entities": []})
              continue

          if isinstance(item, dict):
              note_text = (
                  item.get("text")
                  or item.get("transcription")
                  or item.get("masked_text")
                  or ""
              )
              entities = item.get("entities") if isinstance(item.get("entities"), list) else []
              notes.append({
                  "id": item.get("id", idx + 1),
                  "text": str(note_text),
                  "entities": entities,
              })
              continue

          return jsonify({
              "error": "Each note must be either a string or an object",
              "status": 400,
          }), 400

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


# â”€â”€ UI routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _register_ui_routes(app: Flask) -> None:

  @app.route("/")
  def home():
    """Primary frontend landing page."""
    return render_template("dashboard_new.html", active_page="dashboard")

  @app.route("/dashboard")
  def dashboard():
    """
    Live EDA dashboard â€” renders stats as interactive charts.
    Uses Chart.js loaded from CDN, data fetched from /api/stats.
    """
    return render_template("dashboard_new.html", active_page="dashboard")

  @app.route("/stats")
  def stats_page():
    """
    Stats page â€” displays JSON data in a readable format.
    """
    return render_template("stats.html", active_page="stats")

  @app.route("/system-status")
  def system_status():
    """Visual system status page."""
    return render_template("system_status.html", active_page="status")

  @app.route("/api-explorer")
  def api_explorer():
    """Interactive API explorer page."""
    return render_template("api_explorer.html", active_page="api")

  @app.route("/review-queue")
  def review_queue_page():
    """Human-in-the-loop review queue page for high-risk notes."""
    return render_template("review_queue.html", active_page="review_queue")

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

      if orig_df is None or orig_df.empty:
        return f"<h3>Note {note_id} not found</h3>", 404

      orig_text = orig_df.iloc[0].get("transcription", "")
      specialty = orig_df.iloc[0].get("medical_specialty", "Unknown")
      proc_text = "Not yet processed"
      entity_count = 0

      # processed_notes may not exist yet if phase 2 has not been run.
      try:
        proc_df = loader.sql_query(
          f"SELECT * FROM processed_notes WHERE note_id = {note_id} "
          f"ORDER BY id DESC LIMIT 1"
        )
        if not proc_df.empty:
          proc_text = proc_df.iloc[0].get("masked_text", "Not yet processed")
          entity_count = proc_df.iloc[0].get("entity_count", 0)
      except Exception:
        pass

      return render_template(
        "report.html",
        note_id=note_id,
        specialty=specialty,
        orig_text=orig_text,
        proc_text=proc_text,
        entity_count=entity_count,
        active_page="report",
      )
    except Exception as e:
      return f"<h3>Error: {e}</h3>", 500

  def _build_report_summary_payload(loader: DataLoader) -> dict:
    """Build summary payload shared by HTML and JSON report routes."""
    try:
      try:
        stats = loader.sql_query(
          """
          SELECT
            cn.medical_specialty,
            COUNT(cn.note_id)                           AS total_notes,
            COUNT(pn.id)                                AS processed_notes,
            ROUND(AVG(pn.entity_count), 1)              AS avg_phi_per_note,
            SUM(pn.entity_count)                        AS total_phi_found,
            ROUND(COUNT(pn.id) * 100.0 / COUNT(cn.note_id), 1) AS pct_processed
          FROM clinical_notes cn
          LEFT JOIN processed_notes pn ON cn.note_id = pn.note_id
          GROUP BY cn.medical_specialty
          ORDER BY total_notes DESC
          """
        ).to_dict(orient="records")
      except Exception:
        try:
          # Graceful fallback when processed_notes is not created yet.
          stats = loader.sql_query(
            """
            SELECT
              cn.medical_specialty,
              COUNT(cn.note_id) AS total_notes,
              0 AS processed_notes,
              0.0 AS avg_phi_per_note,
              0 AS total_phi_found,
              0.0 AS pct_processed
            FROM clinical_notes cn
            GROUP BY cn.medical_specialty
            ORDER BY total_notes DESC
            """
          ).to_dict(orient="records")
        except Exception:
          # If clinical_notes is also unavailable, return empty summary.
          stats = []
      return {
        "report_type": "Study Status Summary",
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "compliance": "ICH E6 (R2) GCP",
        "warning": "No source tables found. Run run_phase1.py (and run_phase2.py for processed stats)." if not stats else None,
        "specialties": stats,
      }
    except Exception as e:
      return {"error": str(e)}

  @app.route("/api/report/summary")
  def report_summary_api():
    """Study-level aggregate status report by medical specialty (JSON API)."""
    loader: DataLoader = app.config["LOADER"]
    payload = _build_report_summary_payload(loader)
    if "error" in payload:
      return jsonify(payload), 500
    return jsonify(payload), 200

  @app.route("/report/summary")
  def report_summary_page():
    """Human-readable study-level summary report page."""
    loader: DataLoader = app.config["LOADER"]
    payload = _build_report_summary_payload(loader)

    # Backward compatibility for callers that expect JSON on this path.
    wants_json = request.args.get("format") == "json"
    if wants_json:
      if "error" in payload:
        return jsonify(payload), 500
      return jsonify(payload), 200

    if "error" in payload:
      return f"<h3>Error: {payload['error']}</h3>", 500

    return render_template("report_summary.html", report=payload, active_page="report")


# â”€â”€ HTML Templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inline templates â€” no separate file needed for a portfolio project.
# In production these would be Jinja2 .html files in templates/.

# â”€â”€ HTML Templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClinicalNER Â· Intelligence Platform</title>
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

/* â”€â”€ Animated grid background â”€â”€ */
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

/* â”€â”€ Layout â”€â”€ */
.layout { display: flex; min-height: 100vh; position: relative; z-index: 1; }

/* â”€â”€ Sidebar â”€â”€ */
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

/* â”€â”€ Main content â”€â”€ */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

/* â”€â”€ Topbar â”€â”€ */
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

/* â”€â”€ Content â”€â”€ */
.content { padding: 32px 36px; flex: 1; }

/* â”€â”€ Section header â”€â”€ */
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

/* â”€â”€ Stat cards â”€â”€ */
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

/* â”€â”€ Charts grid â”€â”€ */
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

/* â”€â”€ Audit table â”€â”€ */
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

/* â”€â”€ Event badges â”€â”€ */
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

/* â”€â”€ Count col â”€â”€ */
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

/* â”€â”€ Loading shimmer â”€â”€ */
.shimmer {
  background: linear-gradient(90deg, var(--bg3) 25%, #FFFFFF 50%, var(--bg3) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 6px;
  height: 24px;
  width: 90px;
}
@keyframes shimmer { to { background-position: -200% 0; } }

/* â”€â”€ Error â”€â”€ */
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

  <!-- â”€â”€ Sidebar â”€â”€ -->
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

  <!-- â”€â”€ Main â”€â”€ -->
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
          <div class="stat-delta" id="s-notes-sub">Loadingâ€¦</div>
          <div class="stat-icon">ðŸ“„</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">De-identified</div>
          <div class="stat-value" id="s-processed"><div class="shimmer"></div></div>
          <div class="stat-delta" id="s-proc-sub">Loadingâ€¦</div>
          <div class="stat-icon">ðŸ”’</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">PHI Entities Found</div>
          <div class="stat-value amber" id="s-entities"><div class="shimmer"></div></div>
          <div class="stat-delta">across all processed notes</div>
          <div class="stat-icon">ðŸ”</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Audit Events</div>
          <div class="stat-value blue" id="s-audit"><div class="shimmer"></div></div>
          <div class="stat-delta">append-only trail</div>
          <div class="stat-icon">ðŸ“‹</div>
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
          <span class="tag tag-teal">Append-only Â· ICH E6 compliant</span>
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
            <tr><td colspan="4" style="text-align:center;padding:24px;font-family:'DM Mono',monospace;font-size:11px;color:var(--txt3)">Loading audit eventsâ€¦</td></tr>
          </tbody>
        </table>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /layout -->

<script>
// â”€â”€ Chart defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

// â”€â”€ Animated counter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

// â”€â”€ Event badge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function evBadge(type) {
  const map = {
    NER_COMPLETED: 'teal', DATA_CLEANED_POST: 'teal', PIPELINE_COMPLETE: 'blue',
    RESIDUAL_PHI_FOUND: 'red', API_REQUEST: 'amber', API_RESPONSE: 'amber',
    DATA_CLEANED_PRE: 'teal', PIPELINE_START: 'blue', DATA_INGESTED: 'blue', ERROR: 'red'
  };
  const cls = map[type] || 'blue';
  return `<span class="ev ev-${cls}"><span class="ev-dot"></span>${type}</span>`;
}

// â”€â”€ Load dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
      : 'â€”';
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
          <td><span class="ts">${r.first_seen ? r.first_seen.slice(0,19).replace('T',' ') : 'â€”'}</span></td>
          <td><span class="ts">${r.last_seen  ? r.last_seen.slice(0,19).replace('T',' ')  : 'â€”'}</span></td>
        </tr>`).join('');

  } catch(e) {
    document.getElementById('error-container').innerHTML =
      `<div class="error-banner">âš  Failed to load pipeline data: ${e.message}</div>`;
  }
}

load();
setInterval(load, 30000);
</script>

<!-- â”€â”€ Live De-identification Widget â”€â”€ -->
<div style="margin:0 32px 40px;opacity:0;animation:card-in .5s ease .7s forwards;">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px;">
    <span style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);">Live De-identification</span>
    <div style="flex:1;height:1px;background:var(--border);"></div>
    <span style="font-family:'DM Mono',monospace;font-size:9px;color:var(--txt3);letter-spacing:.1em;" id="widget-status">READY</span>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;overflow:hidden;">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--border);">
        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);display:flex;align-items:center;gap:8px;">
          <div style="width:6px;height:6px;border-radius:50%;background:var(--amber);"></div>
          Raw clinical note
        </div>
        <button onclick="loadSample()" style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.08em;padding:4px 12px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--txt3);cursor:pointer;" onmouseover="this.style.color='var(--teal)'" onmouseout="this.style.color='var(--txt3)'">Load sample</button>
      </div>
      <textarea id="deid-input" placeholder="Paste a clinical note here and watch it de-identify in real time..." style="width:100%;height:220px;background:transparent;border:none;outline:none;padding:16px 18px;font-family:'DM Mono',monospace;font-size:12px;color:var(--txt2);resize:none;line-height:1.7;"></textarea>
      <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 18px;border-top:1px solid var(--border);">
        <span style="font-family:'DM Mono',monospace;font-size:10px;color:var(--txt3);" id="char-count">0 chars</span>
        <div style="display:flex;gap:8px;">
          <button onclick="clearWidget()" style="font-family:'DM Mono',monospace;font-size:9px;padding:5px 12px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--txt3);cursor:pointer;">Clear</button>
          <button onclick="runDeid()" style="font-family:'DM Mono',monospace;font-size:9px;padding:5px 16px;border-radius:6px;border:1px solid rgba(0,229,180,.3);background:rgba(0,229,180,.1);color:var(--teal);cursor:pointer;">Run now</button>
        </div>
      </div>
    </div>

    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;overflow:hidden;">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--border);">
        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);display:flex;align-items:center;gap:8px;">
          <div style="width:6px;height:6px;border-radius:50%;background:var(--teal);"></div>
          De-identified output
        </div>
        <div id="entity-summary"></div>
      </div>
      <div id="deid-output" style="height:220px;padding:16px 18px;font-family:'DM Mono',monospace;font-size:12px;color:var(--txt2);line-height:1.7;overflow-y:auto;white-space:pre-wrap;word-break:break-word;">
        <span style="color:var(--txt3);font-style:italic;">Output will appear here...</span>
      </div>
      <div style="padding:10px 18px;border-top:1px solid var(--border);">
        <div id="entity-chips" style="display:flex;flex-wrap:wrap;gap:6px;min-height:22px;"></div>
      </div>
    </div>
  </div>

  <div id="metrics-row" style="display:none;margin-top:12px;grid-template-columns:repeat(4,1fr);gap:12px;">
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:6px;">Entities found</div>
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--teal);" id="m-count">â€”</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:6px;">Avg confidence</div>
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--txt);" id="m-conf">â€”</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:6px;">PHI types</div>
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--txt);" id="m-types">â€”</div>
    </div>
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:6px;">Valid output</div>
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;" id="m-valid">â€”</div>
    </div>
  </div>
</div>

<script>
const ENTITY_COLORS = {
  DATE:     {bg:'rgba(79,142,247,0.15)',  border:'rgba(79,142,247,0.35)',  text:'#4F8EF7'},
  DOB:      {bg:'rgba(79,142,247,0.1)',   border:'rgba(79,142,247,0.25)',  text:'#7EB0F9'},
  PHONE:    {bg:'rgba(0,229,180,0.12)',   border:'rgba(0,229,180,0.3)',    text:'#00E5B4'},
  MRN:      {bg:'rgba(255,181,71,0.12)',  border:'rgba(255,181,71,0.3)',   text:'#FFB547'},
  HOSPITAL: {bg:'rgba(181,116,247,0.12)', border:'rgba(181,116,247,0.3)', text:'#B574F7'},
  AGE:      {bg:'rgba(118,228,247,0.12)', border:'rgba(118,228,247,0.3)', text:'#76E4F7'},
  PERSON:   {bg:'rgba(255,92,106,0.12)',  border:'rgba(255,92,106,0.3)',   text:'#FF5C6A'},
  LOCATION: {bg:'rgba(0,229,180,0.08)',   border:'rgba(0,229,180,0.2)',    text:'#5DCAA5'},
};

const SAMPLE_NOTE = `Patient: James R. Smith, DOB: 06/15/1978\nMRN: MRN302145. Admitted to St. Mary's Hospital on 01/15/2024.\nContact: (415) 555-9876. Referred by Dr. Emily Chen.\nAge: 45-year-old male with Type 2 Diabetes.\nFollow-up at Memorial Medical Center on 03/01/2024.`;

let debounceTimer = null;
let lastText = '';

const deidInput  = document.getElementById('deid-input');
const deidOutput = document.getElementById('deid-output');
const wStatus    = document.getElementById('widget-status');
const metricsRow = document.getElementById('metrics-row');

function loadSample() {
  deidInput.value = SAMPLE_NOTE;
  deidInput.dispatchEvent(new Event('input'));
}

function clearWidget() {
  deidInput.value = '';
  deidOutput.innerHTML = '<span style="color:var(--txt3);font-style:italic;">Output will appear here...</span>';
  document.getElementById('entity-chips').innerHTML = '';
  document.getElementById('entity-summary').innerHTML = '';
  metricsRow.style.display = 'none';
  wStatus.textContent = 'READY';
  lastText = '';
}

function runDeid() {
  clearTimeout(debounceTimer);
  callAPI(deidInput.value.trim());
}

deidInput.addEventListener('input', () => {
  const text = deidInput.value;
  document.getElementById('char-count').textContent = text.length + ' chars';
  clearTimeout(debounceTimer);
  if (!text.trim()) return;
  wStatus.textContent = 'TYPING...';
  debounceTimer = setTimeout(() => callAPI(text.trim()), 600);
});

async function callAPI(text) {
  if (!text || text === lastText) return;
  lastText = text;
  wStatus.textContent = 'PROCESSING...';
  deidOutput.innerHTML = '<span style="color:var(--txt3);">Running NER pipeline...</span>';
  try {
    const res = await fetch('/api/deidentify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, save: false})
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'API error');
    renderOutput(d);
    renderMetrics(d);
    wStatus.textContent = 'COMPLETE';
  } catch(e) {
    deidOutput.innerHTML = `<span style="color:var(--red);">Error: ${e.message}</span>`;
    wStatus.textContent = 'ERROR';
  }
}

function renderOutput(d) {
  let html = d.masked_text;
  html = html.replace(/\[([A-Z]+)\]/g, (match, label) => {
    const c = ENTITY_COLORS[label] || {bg:'rgba(255,255,255,0.1)',border:'rgba(255,255,255,0.2)',text:'#aaa'};
    return `<span style="background:${c.bg};border:1px solid ${c.border};color:${c.text};border-radius:4px;padding:1px 6px;font-weight:500;font-size:11px;">${match}</span>`;
  });
  deidOutput.innerHTML = html;
}

function renderMetrics(d) {
  const counts = d.entity_types || {};
  document.getElementById('entity-chips').innerHTML = Object.entries(counts).map(([label, count]) => {
    const c = ENTITY_COLORS[label] || {bg:'rgba(255,255,255,0.08)',border:'rgba(255,255,255,0.15)',text:'#aaa'};
    return `<span style="font-family:'DM Mono',monospace;font-size:9px;padding:3px 9px;border-radius:20px;background:${c.bg};border:1px solid ${c.border};color:${c.text};">${label} Ã—${count}</span>`;
  }).join('');
  document.getElementById('entity-summary').innerHTML =
    `<span style="font-family:'DM Mono',monospace;font-size:10px;color:var(--txt3);">${d.entity_count} entities</span>`;
  document.getElementById('m-count').textContent = d.entity_count;
  document.getElementById('m-conf').textContent  = (((d.avg_confidence || 0) * 100).toFixed(0)) + '%';
  document.getElementById('m-types').textContent = Object.keys(counts).length;
  const validEl = document.getElementById('m-valid');
  validEl.textContent = d.is_valid ? 'Yes' : 'No';
  validEl.style.color = d.is_valid ? 'var(--teal)' : 'var(--red)';
  metricsRow.style.display = 'grid';
}
</script>
</body>
</html>"""

REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Note {{ note_id }} Â· De-identification Report</title>
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
    <a href="/dashboard">â† Dashboard</a>
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
          Original â€” contains PHI
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
<title>API Explorer Â· ClinicalNER</title>
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
  bodyEl.textContent = 'Loadingâ€¦';
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
<title>System Status Â· ClinicalNER</title>
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
    <span class="tag tag-teal" id="overall-status">Checkingâ€¦</span>
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
          <div class="check-row"><span class="check-label">Service</span><span class="check-value ok" id="service-name">â€”</span></div>
          <div class="check-row"><span class="check-label">Status</span><span class="check-value ok" id="service-status">â€”</span></div>
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
          <div class="check-row"><span class="check-label">Clinical notes</span><span class="check-value ok" id="note-count">â€”</span></div>
          <div class="check-row"><span class="check-label">Processed</span><span class="check-value ok" id="proc-count">â€”</span></div>
          <div class="check-row"><span class="check-label">Audit events</span><span class="check-value ok" id="audit-count">â€”</span></div>
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
            <td>Liveness probe Â· Docker HEALTHCHECK</td>
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
            <td>De-identify a clinical note Â· returns masked text + entities</td>
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
    document.getElementById('service-name').textContent  = health.service || 'â€”';
    document.getElementById('service-status').textContent = health.status  || 'â€”';
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
