"""
anomaly_detector.py
───────────────────
AnomalyDetector: flags statistically unusual clinical notes using
IsolationForest on 9 engineered features per note.

Why this matters for the JD:
  "Anomaly detection" appears EXPLICITLY in the JD. This class is the
  direct keyword match. IsolationForest is also an ML model — so it
  simultaneously satisfies "develop ML models and algorithms".

Feature engineering choices (9 features):
  1. phi_count         — total PHI entities
  2. phi_density       — entities per 100 characters
  3. text_length       — raw character count
  4. date_count        — DATE + DOB entity count
  5. person_count      — PERSON entity count
  6. phone_count       — PHONE entity count
  7. mrn_count         — MRN entity count
  8. hospital_count    — HOSPITAL entity count
  9. label_diversity   — number of distinct PHI types

IsolationForest is ideal here because:
  - No labelled anomaly data needed (unsupervised)
  - Scales well to high-dimensional feature spaces
  - contamination param maps to expected % of anomalous notes
  - Used in production clinical data QC pipelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

N_FEATURES = 9


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """
    Anomaly detection output for a single note.

    anomaly_score : higher = more anomalous (we flip IsolationForest's
                    sign so the number is intuitive)
    is_anomaly    : True if the model classified this as an outlier
    flags         : human-readable explanations of why it was flagged
    """
    note_id:       int
    anomaly_score: float
    is_anomaly:    bool
    flags:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "note_id":       self.note_id,
            "anomaly_score": self.anomaly_score,
            "is_anomaly":    self.is_anomaly,
            "flags":         self.flags,
        }


# ── AnomalyDetector class ─────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Unsupervised anomaly detection for clinical note quality control.

    Usage
    -----
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(notes)          # list of processed note dicts
    results = detector.predict(notes)

    Each note dict must have:
      id       : int
      text     : str (original or masked)
      entities : list of {"label": str, ...}
    """

    FEATURE_NAMES = [
        "phi_count", "phi_density", "text_length",
        "date_count", "person_count", "phone_count",
        "mrn_count", "hospital_count", "label_diversity",
    ]

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        """
        Parameters
        ----------
        contamination : expected proportion of anomalous notes (0.0–0.5).
                        0.05 = expect 5% of corpus to be flagged.
        random_state  : for reproducibility — same data always same result.
        """
        self.contamination = contamination
        self.model  = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=random_state,
        )
        self.scaler    = StandardScaler()
        self.is_fitted = False
        logger.info(
            "AnomalyDetector ready | contamination=%.2f | n_estimators=100",
            contamination
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, notes: list[dict]) -> "AnomalyDetector":
        """
        Fit IsolationForest and StandardScaler on the full corpus.

        Call this once on your full training set before calling predict().
        Fitting on fewer than 20 notes gives unreliable anomaly boundaries.
        """
        if len(notes) < 10:
            raise ValueError(
                f"Need at least 10 notes to fit, got {len(notes)}. "
                "IsolationForest requires enough variety to learn 'normal'."
            )
        X = self._build_feature_matrix(notes)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("AnomalyDetector fitted on %d notes", len(notes))
        return self

    def predict(self, notes: list[dict]) -> list[AnomalyResult]:
        """
        Score notes against the fitted model.

        Returns a list of AnomalyResult, one per input note.
        Raises RuntimeError if fit() hasn't been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X        = self._build_feature_matrix(notes)
        X_scaled = self.scaler.transform(X)
        scores   = self.model.decision_function(X_scaled)  # lower = more anomalous
        labels   = self.model.predict(X_scaled)             # -1 = anomaly, 1 = normal

        results = []
        for i, note in enumerate(notes):
            anomaly_score = round(max(0.0, float(-scores[i])), 4)  # flip: higher = worse, clamp to 0
            is_anomaly    = bool(labels[i] == -1)
            flags         = self._generate_flags(note, X[i])

            results.append(AnomalyResult(
                note_id=note.get("id", i),
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                flags=flags,
            ))
        return results

    def fit_predict(self, notes: list[dict]) -> list[AnomalyResult]:
        """Convenience: fit and predict in one call."""
        return self.fit(notes).predict(notes)

    def summary(self, results: list[AnomalyResult]) -> dict:
        """Aggregate stats across a batch — used by Flask /api/anomaly-scan."""
        anomalies = [r for r in results if r.is_anomaly]
        return {
            "total_notes":    len(results),
            "anomalies_found": len(anomalies),
            "anomaly_rate":   round(len(anomalies) / max(len(results), 1), 3),
            "avg_score":      round(
                sum(r.anomaly_score for r in results) / max(len(results), 1), 4
            ),
            "top_flags": self._top_flags(results),
        }

    # ── Private: feature engineering ─────────────────────────────────────────

    def _build_feature_matrix(self, notes: list[dict]) -> np.ndarray:
        """Convert list of note dicts to (n_notes, N_FEATURES) numpy array."""
        return np.array([self._extract_features(n) for n in notes])

    def _extract_features(self, note: dict) -> np.ndarray:
        """
        Extract 9 numerical features from a single note dict.

        Key decision: we engineer domain-specific features rather than
        using raw TF-IDF or embeddings. This is faster, interpretable,
        and directly tied to clinical data quality signals that a CDM
        team would care about.
        """
        entities  = note.get("entities", [])
        text      = note.get("text", "")
        text_len  = max(len(text), 1)
        phi_count = len(entities)

        # Aggregate label counts
        label_counts: dict[str, int] = {}
        for e in entities:
            lbl = e.get("label", "") if isinstance(e, dict) else getattr(e, "label", "")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return np.array([
            phi_count,
            phi_count / (text_len / 100),          # phi_density per 100 chars
            text_len,
            label_counts.get("DATE", 0) + label_counts.get("DOB", 0),
            label_counts.get("PERSON", 0),
            label_counts.get("PHONE", 0),
            label_counts.get("MRN", 0),
            label_counts.get("HOSPITAL", 0),
            len(set(label_counts.keys())),          # label diversity
        ], dtype=float)

    # ── Private: flag generation ──────────────────────────────────────────────

    def _generate_flags(self, note: dict, features: np.ndarray) -> list[str]:
        """
        Produce human-readable flags explaining the anomaly.
        These appear in the Flask API response and audit log.
        """
        flags     = []
        entities  = note.get("entities", [])
        text      = note.get("text", "")
        phi_count = int(features[0])
        text_len  = int(features[2])

        if phi_count == 0:
            flags.append("No PHI detected — possible data quality issue")
        if phi_count > 15:
            flags.append(f"Unusually high PHI density: {phi_count} entities")
        if text_len < 50:
            flags.append("Note suspiciously short")
        if text_len > 5000:
            flags.append("Note unusually long — check for concatenation error")

        labels = {
            e.get("label", "") if isinstance(e, dict) else getattr(e, "label", "")
            for e in entities
        }
        if phi_count > 3 and "MRN" not in labels:
            flags.append("Multiple PHI entities but no MRN — possible legacy note")
        if phi_count > 0 and "DATE" not in labels and "DOB" not in labels:
            flags.append("PHI present but no dates — check temporal completeness")

        return flags

    def _top_flags(self, results: list[AnomalyResult], top_n: int = 5) -> list[str]:
        from collections import Counter
        all_flags = [f for r in results for f in r.flags]
        return [flag for flag, _ in Counter(all_flags).most_common(top_n)]