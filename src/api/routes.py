"""
routes.py
─────────
Phase 4: API route definitions.

Endpoints:
  POST   /api/notes          - Upload clinical note
  GET    /api/notes/:id      - Retrieve note
  POST   /api/process        - Run NER pipeline
  GET    /api/dashboard      - EDA visualizations
  GET    /api/audit          - Audit log
  GET    /api/data-quality   - Live data quality report
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, has_request_context, jsonify

from src.utils.dq_vocab import load_specialty_vocab

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Resolve the database path relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "clinicalner.db"
_REGISTRY_PATH = _PROJECT_ROOT / "models" / "model_registry.json"
_BENCHMARK_PATH = _PROJECT_ROOT / "data" / "benchmark_results.json"


# ── Helper ─────────────────────────────────────────────────────────────────────

def _resolve_db_path() -> Path:
    """Use the live Flask ``DB_PATH`` when in a request (matches ``create_app`` tests)."""
    try:
        if has_request_context() and current_app:
            cfg = current_app.config.get("DB_PATH")
            if cfg is not None and str(cfg).strip():
                return Path(cfg)
    except RuntimeError:
        pass
    return _DB_PATH


def _get_db_conn() -> sqlite3.Connection | None:
    """SQLite connection for blueprint DQ/audit routes (read-only for on-disk files)."""
    db_path = _resolve_db_path()
    path_str = str(db_path)
    if path_str == ":memory:":
        try:
            conn = sqlite3.connect(path_str)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error:
            return None
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{path_str}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ── Existing stubs ─────────────────────────────────────────────────────────────

@api_bp.route("/notes", methods=["POST"])
def upload_note():
    """Upload a clinical note for processing."""
    raise NotImplementedError("Phase 4: Note upload to be implemented")


@api_bp.route("/notes/<int:note_id>", methods=["GET"])
def get_note(note_id: int):
    """Retrieve a stored clinical note by ID."""
    conn = _get_db_conn()
    if conn is None:
        return jsonify({"error": "Database not initialised"}), 503
    with conn:
        row = conn.execute(
            "SELECT note_id, medical_specialty, description, transcription "
            "FROM clinical_notes WHERE note_id = ?",
            (note_id,),
        ).fetchone()
    if row is None:
        return jsonify({"error": f"Note {note_id} not found"}), 404
    return jsonify(dict(row))


@api_bp.route("/process", methods=["POST"])
def process_note():
    """Process note through NER pipeline."""
    raise NotImplementedError("Phase 4: NER processing to be implemented")


@api_bp.route("/audit", methods=["GET"])
def get_audit_log():
    """Return recent audit log entries."""
    conn = _get_db_conn()
    if conn is None:
        return jsonify({"error": "Database not initialised"}), 503
    with conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ── /api/data-quality ──────────────────────────────────────────────────────────

@api_bp.route("/data-quality", methods=["GET"])
def data_quality_report():
    """
    GET /api/data-quality
    ─────────────────────
    Returns a live JSON data quality report covering five dimensions:

    - completeness  : null/missing field counts
    - conformity    : invalid dates, unrecognised specialties
    - consistency   : zero-PHI rate (likely-missed PHI)
    - accuracy      : precision / recall / F1 from latest benchmark run
    - timeliness    : 95th-percentile processing latency + last run timestamp

    If the database or benchmark file has not been created yet the endpoint
    returns best-effort partial data with an ``initialised: false`` flag.
    """
    _dp = _resolve_db_path()
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initialised": str(_dp) == ":memory:" or _dp.exists(),
    }

    # 1 · Completeness ────────────────────────────────────────────────────────
    completeness: dict = {"score": None, "null_counts": {}}
    conn = _get_db_conn()
    if conn is not None:
        try:
            with conn:
                total = conn.execute("SELECT COUNT(*) FROM clinical_notes").fetchone()[0]
                null_specialty = conn.execute(
                    "SELECT COUNT(*) FROM clinical_notes "
                    "WHERE medical_specialty IS NULL OR TRIM(medical_specialty) = ''"
                ).fetchone()[0]
                null_text_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM clinical_notes "
                        "WHERE transcription IS NULL OR TRIM(transcription) = ''"
                    ).fetchone()[0]
                )

            nonnull_total = max(1, total)
            completeness["score"] = round(1 - (null_specialty + null_text_count) / (nonnull_total * 2), 4)
            completeness["total_notes"] = total
            completeness["null_counts"] = {
                "medical_specialty": null_specialty,
                "transcription": null_text_count,
            }
        except Exception as exc:
            logger.warning("Completeness query failed: %s", exc)
            completeness["error"] = str(exc)
    report["completeness"] = completeness

    # 2 · Conformity ──────────────────────────────────────────────────────────
    conformity: dict = {"malformed_dates": None, "invalid_specialties": None}
    if conn is not None:
        try:
            with conn:
                try:
                    bad_dates = conn.execute(
                        "SELECT COUNT(*) FROM clinical_notes "
                        "WHERE note_date IS NOT NULL "
                        "  AND typeof(note_date) = 'text' "
                        "  AND date(note_date) IS NULL"
                    ).fetchone()[0]
                    conformity["malformed_dates"] = bad_dates
                except sqlite3.OperationalError:
                    conformity["malformed_dates"] = 0
                    conformity["malformed_dates_note"] = "note_date column absent in this DB build"

                vocab = load_specialty_vocab(_PROJECT_ROOT)
                if vocab:
                    rows = conn.execute(
                        "SELECT note_id, medical_specialty FROM clinical_notes"
                    ).fetchall()
                    invalid = sum(
                        1 for r in rows if r["medical_specialty"] and r["medical_specialty"] not in vocab
                    )
                    conformity["invalid_specialties"] = invalid
                    conformity["vocab_version"] = "data/specialty_vocab.json"
        except Exception as exc:
            logger.warning("Conformity query failed: %s", exc)
            conformity["error"] = str(exc)
    report["conformity"] = conformity

    # 3 · Consistency (zero-PHI rate) ─────────────────────────────────────────
    consistency: dict = {"zero_phi_rate": None}
    if conn is not None:
        try:
            with conn:
                # Notes that were processed but yielded 0 detected entities
                processed_total = conn.execute(
                    "SELECT COUNT(*) FROM processed_notes"
                ).fetchone()[0]
                zero_phi = conn.execute(
                    "SELECT COUNT(*) FROM processed_notes WHERE entity_count = 0"
                ).fetchone()[0]
            consistency["zero_phi_rate"] = round(zero_phi / max(1, processed_total), 4)
            consistency["zero_phi_notes"] = zero_phi
            consistency["processed_total"] = processed_total
        except Exception as exc:
            logger.warning("Consistency query failed: %s", exc)
            consistency["error"] = str(exc)
    report["consistency"] = consistency

    # 4 · Accuracy (latest benchmark) ─────────────────────────────────────────
    accuracy: dict = {"precision": None, "recall": None, "f1": None}
    if _BENCHMARK_PATH.exists():
        try:
            bm = json.loads(_BENCHMARK_PATH.read_text())
            if isinstance(bm, list):
                primary = None
                for row in bm:
                    if "spacy" in str(row.get("model_name", "")).lower():
                        primary = row
                        break
                primary = primary or (bm[-1] if bm else None)
                if primary:
                    accuracy.update({
                        "precision": primary.get("precision"),
                        "recall": primary.get("recall"),
                        "f1": primary.get("f1"),
                        "run_date": None,
                        "test_notes": primary.get("notes_tested"),
                        "schema": "legacy-array",
                    })
            else:
                accuracy.update({
                    "precision": bm.get("precision"),
                    "recall": bm.get("recall"),
                    "f1": bm.get("f1_score") or bm.get("f1"),
                    "run_date": bm.get("run_date") or bm.get("run_timestamp") or bm.get("timestamp"),
                    "test_notes": bm.get("test_notes") or bm.get("test_set_size"),
                    "schema": bm.get("schema_version", "benchmark/v2"),
                    "active_model_id": bm.get("active_model_id"),
                })
        except Exception as exc:
            logger.warning("Benchmark parse failed: %s", exc)
            accuracy["error"] = str(exc)
    else:
        accuracy["note"] = "No benchmark results found. Run `python run_benchmark.py` to populate."
    report["accuracy"] = accuracy

    # 5 · Timeliness ──────────────────────────────────────────────────────────
    timeliness: dict = {"p95_latency_ms": None, "last_run_utc": None}
    if conn is not None:
        try:
            with conn:
                last_run = conn.execute(
                    "SELECT MAX(processed_at) FROM processed_notes"
                ).fetchone()[0]
            timeliness["last_run_utc"] = last_run
            try:
                with conn:
                    lat_rows = conn.execute(
                        "SELECT processing_time_ms FROM processed_notes "
                        "WHERE processing_time_ms IS NOT NULL "
                        "ORDER BY processing_time_ms"
                    ).fetchall()
                if lat_rows:
                    vals = sorted(float(r[0]) for r in lat_rows)
                    idx = min(len(vals) - 1, max(0, int(0.95 * (len(vals) - 1))))
                    timeliness["p95_latency_ms"] = round(vals[idx], 2)
            except sqlite3.OperationalError:
                pass
        except Exception as exc:
            logger.warning("Timeliness query failed: %s", exc)
            timeliness["error"] = str(exc)
    if timeliness.get("p95_latency_ms") is None and _BENCHMARK_PATH.exists():
        try:
            bm = json.loads(_BENCHMARK_PATH.read_text())
            rows = bm.get("results") if isinstance(bm, dict) else bm
            if isinstance(rows, list):
                for row in rows:
                    if "spacy" in str(row.get("model_name", "")).lower():
                        timeliness["p95_latency_ms"] = row.get("latency_ms")
                        timeliness["latency_source"] = "benchmark_results.json (spaCy row)"
                        break
        except (json.JSONDecodeError, TypeError):
            pass
    report["timeliness"] = timeliness

    # 6 · Active model info ───────────────────────────────────────────────────
    if _REGISTRY_PATH.exists():
        try:
            registry = json.loads(_REGISTRY_PATH.read_text())
            active_id = registry.get("active_model")
            active = next(
                (m for m in registry.get("models", []) if m["id"] == active_id), None
            )
            report["active_model"] = {
                "id":      active_id,
                "name":    active.get("name") if active else None,
                "f1":      (active or {}).get("metrics", {}).get("f1"),
            }
        except Exception as exc:
            logger.warning("Registry parse failed: %s", exc)

    if conn is not None:
        conn.close()

    return jsonify(report)


@api_bp.route("/model-registry", methods=["GET"])
def model_registry():
    """
    GET /api/model-registry
    ───────────────────────
    Returns the full ``models/model_registry.json`` payload plus resolved
    ``active_model`` for UI and CI traceability.
    """
    if not _REGISTRY_PATH.exists():
        return jsonify(
            {"error": "model_registry.json not found", "_schema": "clinicalner-model-registry/v1"}
        ), 404
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Registry read failed: %s", exc)
        return jsonify({"error": str(exc)}), 500

    aid = data.get("active_model")
    active = next((m for m in data.get("models", []) if m.get("id") == aid), None)
    return jsonify(
        {
            "_schema": data.get("_schema"),
            "active_model_id": aid,
            "active_model": active,
            "models": data.get("models", []),
        }
    )
