"""
data_loader.py
──────────────
Handles ingestion of clinical note datasets into the project.

Supports:
  - MTSamples CSV (Kaggle download)
  - Synthetic dataset (for dev/demo when Kaggle unavailable)
  - i2b2 XML (stub — Phase 2 extension)

Design decision: DataLoader is a class (not bare functions) so the Flask
app and CLI scripts both get consistent state (db_path, raw_dir) without
passing config everywhere.  OOP = direct JD requirement tick.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.dq_vocab import load_specialty_vocab

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ── Constants ────────────────────────────────────────────────────────────────

MTSAMPLES_URL = (
    "https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions/download"
)
REQUIRED_COLUMNS = {"transcription", "medical_specialty", "description"}


# ── DataLoader class ─────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and stores clinical notes into a SQLite database.

    Attributes
    ----------
    raw_dir   : Path to raw data directory
    db_path   : Path to SQLite database file
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        db_path: str = "data/clinicalner.db",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.db_path = Path(db_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialised | raw_dir=%s | db=%s", self.raw_dir, self.db_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_mtsamples(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load MTSamples from a local CSV (downloaded from Kaggle).

        How to get the CSV:
          1. Go to https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
          2. Download → place 'mtsamples.csv' in data/raw/
          3. Call: loader.load_mtsamples()

        Parameters
        ----------
        csv_path : override default data/raw/mtsamples.csv
        """
        path = Path(csv_path) if csv_path else self.raw_dir / "mtsamples.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"MTSamples CSV not found at {path}.\n"
                "Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
                "Or call: loader.generate_synthetic_dataset() for a demo dataset."
            )
        df = pd.read_csv(path)
        df = self._clean_column_names(df)
        self._validate_columns(df)
        df = self._basic_clean(df)
        logger.info("Loaded %d records from MTSamples", len(df))
        return df

    def generate_synthetic_dataset(self, n_records: int = 500) -> pd.DataFrame:
        """
        Generate a realistic synthetic clinical notes dataset for dev/demo.

        This mimics the MTSamples schema so all downstream code works
        identically.  Replace with real MTSamples before final demo.

        Key decision: synthetic data lets you build and test the full
        pipeline without waiting for dataset approval — common in
        regulated industries where real data access is gated.
        """
        import random

        random.seed(42)

        specialties = [
            "Cardiology", "Orthopedic", "Neurology", "Gastroenterology",
            "Pulmonology", "Nephrology", "Oncology", "Psychiatry",
            "Endocrinology", "Surgery",
        ]

        # PHI patterns that will become our NER targets
        first_names = ["James", "Maria", "Robert", "Linda", "Michael", "Patricia",
                        "David", "Jennifer", "John", "Elizabeth", "William", "Susan"]
        last_names  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                        "Miller", "Davis", "Wilson", "Taylor", "Anderson", "Thomas"]
        hospitals   = ["St. Mary's Hospital", "Memorial Medical Center",
                        "City General Hospital", "University Health System",
                        "Riverside Medical Center", "Lakeside Clinic"]
        ages        = list(range(25, 85))
        phone_area  = ["212", "415", "312", "713", "404", "602"]

        def rand_date():
            m = random.randint(1, 12)
            d = random.randint(1, 28)
            y = random.randint(2018, 2023)
            return f"{m:02d}/{d:02d}/{y}"

        def rand_phone(area):
            return f"({area}) {random.randint(200,999)}-{random.randint(1000,9999)}"

        templates = [
            (
                "Patient {first} {last}, a {age}-year-old {sex} presented to {hospital} on {date}. "
                "Chief complaint: chest pain radiating to the left arm for 3 days. "
                "Vital signs stable. ECG shows sinus rhythm. "
                "Contact: {phone}. Plan: admit for observation and serial troponins."
            ),
            (
                "This is a {age}-year-old {sex} referred by Dr. {ref_last} for evaluation. "
                "Patient {first} {last} was seen at {hospital} on {date}. "
                "History of hypertension and type 2 diabetes mellitus. "
                "HbA1c 8.2%. Current medications include metformin 1000mg BID. "
                "Follow-up scheduled for {follow_date}."
            ),
            (
                "PATIENT: {first} {last} | DOB: {dob} | MRN: {mrn}\n"
                "Date of service: {date} | Facility: {hospital}\n"
                "Phone: {phone}\n\n"
                "ASSESSMENT: {age}-year-old {sex} with progressive shortness of breath. "
                "Pulmonary function tests reveal moderate obstructive pattern. "
                "Initiating tiotropium 18mcg inhaled daily. "
                "Patient instructed to return to {hospital} if symptoms worsen."
            ),
            (
                "Operative report — {hospital}, {date}.\n"
                "Patient: {first} {last}, Age {age}, {sex}.\n"
                "Surgeon: Dr. {ref_last}. Procedure: Laparoscopic cholecystectomy.\n"
                "Findings: Acutely inflamed gallbladder with multiple calculi. "
                "Procedure completed without complications. "
                "Patient to follow up at {hospital} on {follow_date}."
            ),
        ]

        records = []
        for i in range(n_records):
            fn    = random.choice(first_names)
            ln    = random.choice(last_names)
            age   = random.choice(ages)
            sex   = random.choice(["male", "female"])
            hosp  = random.choice(hospitals)
            date  = rand_date()
            fdate = rand_date()
            phone = rand_phone(random.choice(phone_area))
            mrn   = f"MRN{random.randint(100000, 999999)}"
            dob   = rand_date()
            ref   = random.choice(last_names)
            spec  = random.choice(specialties)

            tmpl = random.choice(templates)
            note = tmpl.format(
                first=fn, last=ln, age=age, sex=sex, hospital=hosp,
                date=date, follow_date=fdate, phone=phone, mrn=mrn,
                dob=dob, ref_last=ref,
            )

            records.append({
                "note_id":           i + 1,
                "medical_specialty": spec,
                "description":       f"Clinical note — {spec}",
                "transcription":     note,
                "has_phi":           True,
            })

        df = pd.DataFrame(records)
        out = self.raw_dir / "synthetic_notes.csv"
        df.to_csv(out, index=False)
        logger.info("Generated %d synthetic records → %s", n_records, out)
        return df

    def save_to_db(self, df: pd.DataFrame, table: str = "clinical_notes") -> int:
        """
        Persist a DataFrame to the SQLite database.

        Schema (clinical_notes):
          note_id           INTEGER PRIMARY KEY
          medical_specialty TEXT
          description       TEXT
          transcription     TEXT    ← raw clinical note
          has_phi           INTEGER ← 1 = confirmed PHI present
          created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        Key decision: SQLite over Postgres for portability — the Docker
        compose in Phase 5 will add Postgres as an optional service, but
        SQLite lets any reviewer run the project with zero infrastructure.
        """
        with sqlite3.connect(self.db_path) as conn:
            df_db = df.copy()
            # Ensure has_phi is integer (SQLite has no bool)
            if "has_phi" in df_db.columns:
                df_db["has_phi"] = df_db["has_phi"].astype(int)
            df_db.to_sql(table, conn, if_exists="replace", index=False)
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("Saved %d rows to table '%s' in %s", count, table, self.db_path)
        return count

    def load_from_db(self, table: str = "clinical_notes", limit: Optional[int] = None) -> pd.DataFrame:
        """Load records back from SQLite."""
        query = f"SELECT * FROM {table}"
        if limit:
            query += f" LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        logger.info("Loaded %d rows from '%s'", len(df), table)
        return df

    def sql_query(self, query: str, params: tuple | list | None = None) -> pd.DataFrame:
        """
        Run an arbitrary SQL query against the project DB.

        Example:
            loader.sql_query(
                "SELECT medical_specialty, COUNT(*) as n "
                "FROM clinical_notes GROUP BY medical_specialty ORDER BY n DESC"
            )
            loader.sql_query("SELECT * FROM t WHERE id = ?", (note_id,))
        """
        with sqlite3.connect(self.db_path) as conn:
            if params is None:
                return pd.read_sql_query(query, conn)
            return pd.read_sql_query(query, conn, params=tuple(params))

    def quarantine_invalid_specialties(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split rows whose ``medical_specialty`` is not in ``data/specialty_vocab.json``.

        Quarantined rows are written under ``data/quarantine/`` as CSV (DQP §3.2).
        """
        root = self.db_path.resolve().parent.parent
        vocab = load_specialty_vocab(root)
        if not vocab or "medical_specialty" not in df.columns:
            return df, pd.DataFrame()

        ok_mask = df["medical_specialty"].astype(str).isin(vocab)
        good = df[ok_mask].copy()
        bad = df[~ok_mask].copy()
        if not bad.empty:
            qdir = root / "data" / "quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = qdir / f"quarantine_specialty_{ts}.csv"
            bad.to_csv(path, index=False)
            logger.warning("Quarantined %d rows → %s", len(bad), path)
        return good, bad

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop fully empty transcriptions; strip leading/trailing whitespace."""
        before = len(df)
        df = df.dropna(subset=["transcription"])
        df["transcription"] = df["transcription"].str.strip()
        df = df[df["transcription"].str.len() > 0]
        logger.info("Cleaned: %d → %d rows (dropped %d empty)", before, len(df), before - len(df))
        return df.reset_index(drop=True)
