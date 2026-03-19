"""
audit_logger.py
───────────────
AuditLogger: immutable, append-only audit trail for all data
transformations in the ClinicalNER pipeline.

Why this exists:
  Clinical data pipelines in regulated environments (GCP, ICH E6, HIPAA)
  require a tamper-evident record of every transformation applied to
  patient data. Without an audit trail you cannot prove de-identification
  was applied, reproduce a result from 6 months ago, or answer
  "who changed this and when?" in a regulatory audit.

Design decisions:
  - Append-only  : no UPDATE/DELETE on audit_log ever
  - Immutable    : AuditEntry is a frozen dataclass
  - Structured   : each event has a typed EventType enum
  - Queryable    : helper methods return Pandas DataFrames
  - Lightweight  : pure SQLite, no extra dependencies
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ── Event types ───────────────────────────────────────────────────────────────

class EventType(str, Enum):
    DATA_INGESTED      = "DATA_INGESTED"
    DATA_CLEANED_PRE   = "DATA_CLEANED_PRE"
    NER_COMPLETED      = "NER_COMPLETED"
    DATA_CLEANED_POST  = "DATA_CLEANED_POST"
    RESIDUAL_PHI_FOUND = "RESIDUAL_PHI_FOUND"
    NOTE_VALIDATED     = "NOTE_VALIDATED"
    NOTE_FLAGGED       = "NOTE_FLAGGED"
    API_REQUEST        = "API_REQUEST"
    API_RESPONSE       = "API_RESPONSE"
    PIPELINE_START     = "PIPELINE_START"
    PIPELINE_COMPLETE  = "PIPELINE_COMPLETE"
    ERROR              = "ERROR"


# ── AuditEntry dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEntry:
    event_type:  str
    note_id:     Optional[int]
    description: str
    metadata:    str
    timestamp:   str

    @classmethod
    def create(
        cls,
        event_type: EventType,
        description: str,
        note_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> "AuditEntry":
        return cls(
            event_type=event_type.value,
            note_id=note_id,
            description=description,
            metadata=json.dumps(metadata or {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ── AuditLogger class ─────────────────────────────────────────────────────────

class AuditLogger:
    """
    Append-only audit logger persisting to SQLite.

    Usage
    -----
    audit = AuditLogger(db_path="data/clinicalner.db")
    audit.log(EventType.NER_COMPLETED, "5 entities found", note_id=42,
              metadata={"entity_types": {"DATE": 2}})
    df = audit.get_log(note_id=42)
    """

    TABLE = "audit_log"

    CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT    NOT NULL,
            note_id     INTEGER,
            description TEXT    NOT NULL,
            metadata    TEXT    DEFAULT '{}',
            timestamp   TEXT    NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    INDEX_SQLS = [
        "CREATE INDEX IF NOT EXISTS idx_audit_note_id    ON audit_log (note_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log (event_type)",
        "CREATE INDEX IF NOT EXISTS idx_audit_timestamp  ON audit_log (timestamp)",
    ]

    def __init__(self, db_path: str = "data/clinicalner.db") -> None:
        self.db_path = str(db_path)
        # :memory: databases are per-connection in SQLite.
        # Hold one persistent connection so the schema survives across calls.
        self._mem_conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if self.db_path == ":memory:" else None
        )
        self._init_table()
        logger.info("AuditLogger ready | db=%s", self.db_path)

    # ── Context manager for connections ──────────────────────────────────────

    @contextmanager
    def _connect(self):
        """
        Yield the right SQLite connection:
          :memory: → persistent connection, never closes
          file     → fresh connection per call, auto-closes
        """
        if self._mem_conn is not None:
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(self.CREATE_SQL)
            for sql in self.INDEX_SQLS:
                conn.execute(sql)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        event_type: EventType,
        description: str,
        note_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Append one audit entry. Returns the new row id."""
        entry = AuditEntry.create(
            event_type=event_type,
            description=description,
            note_id=note_id,
            metadata=metadata,
        )
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO audit_log "
                "(event_type, note_id, description, metadata, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (entry.event_type, entry.note_id,
                 entry.description, entry.metadata, entry.timestamp)
            )
            row_id = cursor.lastrowid
        logger.debug("Audit | %s | note=%s | %s", event_type.value, note_id, description)
        return row_id

    def log_ner_result(self, ner_result: dict) -> int:
        """Convenience: log a NERPipeline.process_note() result."""
        return self.log(
            EventType.NER_COMPLETED,
            description=f"NER complete: {ner_result['entity_count']} entities found",
            note_id=ner_result.get("note_id"),
            metadata={
                "entity_count": ner_result["entity_count"],
                "entity_types": ner_result["entity_types"],
            },
        )

    def log_cleaning_result(self, clean_result, note_id: Optional[int] = None) -> int:
        """Convenience: log a DataCleaner CleaningResult."""
        event = EventType.DATA_CLEANED_POST
        if clean_result.residual_phi:
            event = EventType.RESIDUAL_PHI_FOUND
        return self.log(
            event,
            description=(
                f"Cleaning: {clean_result.change_count} change(s) | "
                f"valid={clean_result.is_valid}"
            ),
            note_id=note_id,
            metadata={
                "changes":      clean_result.changes,
                "is_valid":     clean_result.is_valid,
                "residual_phi": clean_result.residual_phi,
            },
        )

    def log_batch_pipeline(
        self,
        ner_results: list[dict],
        clean_stats: dict,
        pipeline_id: Optional[str] = None,
    ) -> None:
        self.log(
            EventType.PIPELINE_COMPLETE,
            description=f"Batch pipeline complete: {len(ner_results)} notes processed",
            metadata={
                "pipeline_id":     pipeline_id,
                "notes_processed": len(ner_results),
                "total_entities":  sum(r["entity_count"] for r in ner_results),
                "clean_stats":     clean_stats,
            },
        )

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_log(
        self,
        note_id: Optional[int] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        conditions, params = [], []
        if note_id is not None:
            conditions.append("note_id = ?")
            params.append(note_id)
        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT id, event_type, note_id, description, timestamp "
            f"FROM audit_log {where} ORDER BY id DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_summary(self) -> pd.DataFrame:
        sql = (
            "SELECT event_type, COUNT(*) as count, "
            "MIN(timestamp) as first_seen, MAX(timestamp) as last_seen "
            "FROM audit_log GROUP BY event_type ORDER BY count DESC"
        )
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn)

    def get_flagged_notes(self) -> pd.DataFrame:
        return self.get_log(event_type=EventType.RESIDUAL_PHI_FOUND, limit=500)

    def get_pipeline_history(self) -> pd.DataFrame:
        return self.get_log(event_type=EventType.PIPELINE_COMPLETE, limit=50)

    def total_events(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]