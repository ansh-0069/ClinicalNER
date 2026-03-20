"""
quality_scorer.py
─────────────────
DataQualityScorer: computes a 0-100 data quality score for each
de-identified clinical note.

Why this matters for the JD:
  Activity 5 — "Identify and resolve data inconsistencies and errors
  in alignment with data quality standards and regulatory requirements."

  A numeric quality score with letter grade is exactly what CDM teams
  produce in Data Quality Plans (DQP). This class makes that concrete.

Scoring dimensions (5 components, weighted):
  1. PHI completeness   (30 pts) — does the note have MRN + date?
  2. NER confidence     (25 pts) — avg entity confidence score
  3. Text adequacy      (20 pts) — note is long enough to be meaningful
  4. Validation pass    (15 pts) — DataCleaner found no residual PHI
  5. Entity diversity   (10 pts) — at least 2 different PHI types found

Grade mapping:
  A  90-100   Production-ready, no review needed
  B  75-89    Acceptable, spot-check recommended
  C  60-74    Marginal, manual review advised
  D  < 60     Flag for QA team
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class QualityResult:
    """Immutable quality assessment for a single note."""
    note_id:        int
    score:          int          # 0-100
    grade:          str          # A / B / C / D
    label:          str          # human-readable label
    breakdown:      dict         # component scores
    flags:          list[str]    # specific quality issues
    recommendation: str          # single-line action

    def to_dict(self) -> dict:
        return {
            "note_id":        self.note_id,
            "score":          self.score,
            "grade":          self.grade,
            "label":          self.label,
            "breakdown":      self.breakdown,
            "flags":          self.flags,
            "recommendation": self.recommendation,
        }


class DataQualityScorer:
    """
    Scores clinical notes on 5 weighted dimensions.

    Usage
    -----
    scorer = DataQualityScorer()
    result = scorer.score(
        note_id    = 42,
        text       = "Patient DOB: 04/12/1985...",
        entities   = [{"label": "DOB"}, {"label": "MRN"}],
        is_valid   = True,
        avg_confidence = 0.95,
    )
    print(result.grade, result.score)   # A  94
    """

    # Minimum text length to be considered a meaningful clinical note
    MIN_TEXT_LENGTH = 80

    # Core PHI that a complete note should always have
    CORE_PHI = {"MRN", "DATE", "DOB"}

    def score(
        self,
        note_id:        int,
        text:           str,
        entities:       list,
        is_valid:       bool  = True,
        avg_confidence: float = 0.0,
    ) -> QualityResult:
        """
        Compute quality score for one note.

        Parameters
        ----------
        note_id        : database note ID
        text           : original or masked text (used for length check)
        entities       : list of entity dicts with 'label' key
        is_valid       : DataCleaner validation result
        avg_confidence : mean NER confidence across all entities
        """
        labels = {
            e.get("label", "") if isinstance(e, dict) else getattr(e, "label", "")
            for e in entities
        }

        # ── Component 1: PHI completeness (30 pts) ───────────────────────────
        # A clinically useful de-identified note needs at least MRN + a date
        core_found  = len(labels & self.CORE_PHI)
        completeness = min(30, core_found * 10 + (10 if len(entities) >= 2 else 0))

        # ── Component 2: NER confidence (25 pts) ─────────────────────────────
        # Scale 0.0-1.0 confidence linearly to 0-25 pts
        confidence_score = round(avg_confidence * 25)

        # ── Component 3: Text adequacy (20 pts) ──────────────────────────────
        text_len = len(text.strip())
        if text_len >= 400:
            adequacy = 20
        elif text_len >= 200:
            adequacy = 15
        elif text_len >= self.MIN_TEXT_LENGTH:
            adequacy = 10
        else:
            adequacy = 0

        # ── Component 4: Validation pass (15 pts) ────────────────────────────
        validation_score = 15 if is_valid else 0

        # ── Component 5: Entity diversity (10 pts) ───────────────────────────
        diversity = min(10, len(labels) * 3)

        total = completeness + confidence_score + adequacy + validation_score + diversity
        total = max(0, min(100, total))

        grade, label, recommendation = self._grade(total, is_valid, labels, entities)
        flags = self._generate_flags(
            text_len, labels, entities, is_valid, avg_confidence
        )

        return QualityResult(
            note_id  = note_id,
            score    = total,
            grade    = grade,
            label    = label,
            breakdown = {
                "phi_completeness":  completeness,
                "ner_confidence":    confidence_score,
                "text_adequacy":     adequacy,
                "validation_pass":   validation_score,
                "entity_diversity":  diversity,
            },
            flags          = flags,
            recommendation = recommendation,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _grade(
        self,
        score:    int,
        is_valid: bool,
        labels:   set,
        entities: list,
    ) -> tuple[str, str, str]:
        if not is_valid:
            return "D", "Failed validation", "Residual PHI detected — manual review required before use"
        if score >= 90:
            return "A", "Production-ready", "No action required — cleared for downstream processing"
        if score >= 75:
            return "B", "Acceptable", "Spot-check recommended — review entity coverage"
        if score >= 60:
            return "C", "Marginal", "Manual review advised — check completeness and confidence"
        return "D", "Flagged for QA", "Hold for QA team review — does not meet DQP standards"

    def _generate_flags(
        self,
        text_len:       int,
        labels:         set,
        entities:       list,
        is_valid:       bool,
        avg_confidence: float,
    ) -> list[str]:
        flags = []
        if not is_valid:
            flags.append("Residual PHI detected by DataCleaner — failed validation")
        if text_len < self.MIN_TEXT_LENGTH:
            flags.append(f"Note too short ({text_len} chars) — minimum {self.MIN_TEXT_LENGTH} recommended")
        if len(entities) == 0:
            flags.append("No PHI entities detected — note may be already clean or data quality issue")
        if "MRN" not in labels:
            flags.append("Missing MRN — patient linkage may not be possible")
        if "DATE" not in labels and "DOB" not in labels:
            flags.append("No temporal PHI (DATE/DOB) — check temporal completeness")
        if avg_confidence < 0.70 and len(entities) > 0:
            flags.append(f"Low NER confidence ({avg_confidence:.0%}) — entity extraction uncertain")
        if len(labels) == 1:
            flags.append("Only one PHI type found — note may be incomplete")
        return flags