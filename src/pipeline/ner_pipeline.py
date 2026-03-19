"""
ner_pipeline.py
───────────────
NERPipeline: core PHI extraction and masking class for ClinicalNER.

Design: Hybrid NER approach
  - Regex layer  : structured PHI — DATE, PHONE, AGE, MRN (high precision,
                   zero false negatives on well-formed patterns)
  - spaCy layer  : unstructured PHI — PERSON, LOCATION, ORG/HOSPITAL
                   (ML-based, handles name variation and context)

Why hybrid and not pure ML?
  Clinical PHI has two distinct subtypes:
    1. Structured  → "04/12/2022", "(415) 555-1234", "MRN902341"
       These follow strict formats. Regex gets 100% recall here.
       A neural model wastes capacity learning what a regex already knows.
    2. Unstructured → "James Smith", "Riverside Medical Center"
       Context-dependent. "Smith" is a name; "Smith procedure" is not.
       ML (spaCy) handles this better than regex.

  This separation is standard in production clinical NLP pipelines
  (e.g., AWS Comprehend Medical uses the same two-layer strategy).

Install spaCy model on your machine:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ── Entity dataclass ──────────────────────────────────────────────────────────

@dataclass
class PHIEntity:
    """
    Represents a single detected PHI span.

    Attributes
    ----------
    label    : entity type (PERSON, DATE, LOCATION, HOSPITAL, AGE, PHONE, MRN)
    text     : original matched text in the note
    start    : character offset start
    end      : character offset end
    source   : 'regex' or 'spacy' — which layer detected it
    """
    label:  str
    text:   str
    start:  int
    end:    int
    source: str = "regex"

    def __repr__(self) -> str:
        return f"PHIEntity({self.label}, '{self.text}', [{self.start}:{self.end}], src={self.source})"


# ── Regex patterns ────────────────────────────────────────────────────────────

# Key decision: patterns ordered from MOST to LEAST specific.
# Overlapping matches are resolved by taking the longest span (see _resolve_overlaps).

PHI_PATTERNS: dict[str, list[str]] = {

    "MRN": [
        r"\bMRN[-\s]?\d{4,10}\b",              # MRN123456, MRN-123456
        r"\bMedical\s+Record\s+(?:Number|No\.?|#)\s*:?\s*\d{4,10}\b",
    ],

    "PHONE": [
        r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}",    # (415) 555-1234 / 415.555.1234
        r"\b\d{10}\b",                               # 4155551234 (10 digits, no separator)
        r"\+1[\s\-]?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}",  # +1 (415) 555-1234
    ],

    "DATE": [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",             # 04/12/2022 or 4/12/22
        r"\b\d{4}-\d{2}-\d{2}\b",                   # 2022-04-12  (ISO)
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
    ],

    "AGE": [
        r"\b\d{1,3}[\s\-]year[\s\-]old\b",          # 45-year-old / 45 year old
        r"\bage[d]?\s*:?\s*\d{1,3}\b",              # age: 45 / aged 45
        r"\b\d{1,3}[yY][oO]\b",                     # 45yo / 45YO
    ],

    "HOSPITAL": [
        # Named clinical facilities — order matters: longer patterns first
        r"\b[A-Z][a-zA-Z\s\'\.]+(?:Hospital|Medical\s+Center|Health\s+System|"
        r"Clinic|Healthcare|Medical\s+Group|Infirmary|Institute|"
        r"Health\s+Center|Urgent\s+Care)\b",
        r"\b(?:St\.|Saint|Mount|Mt\.)\s+[A-Z][a-zA-Z]+(?:\s+Hospital|\s+Medical)?\b",
    ],

    "DOB": [
        r"\bDOB\s*:?\s*\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\bDate\s+of\s+Birth\s*:?\s*\d{1,2}/\d{1,2}/\d{2,4}\b",
    ],
}

# spaCy label → our PHI label mapping
SPACY_LABEL_MAP = {
    "PERSON":  "PERSON",
    "ORG":     "HOSPITAL",    # spaCy uses ORG for medical orgs
    "GPE":     "LOCATION",    # Geo-political entities: cities, states
    "LOC":     "LOCATION",
    "FAC":     "HOSPITAL",    # Facilities (spaCy)
}


# ── NERPipeline class ─────────────────────────────────────────────────────────

class NERPipeline:
    """
    Hybrid PHI Named Entity Recognition and de-identification pipeline.

    Usage
    -----
    pipeline = NERPipeline(db_path="data/clinicalner.db")
    result   = pipeline.process_note(note_text, note_id=1)

    print(result["masked_text"])
    print(result["entities"])
    """

    def __init__(
        self,
        db_path: str = "data/clinicalner.db",
        use_spacy: bool = True,
        mask_style: str = "bracket",   # 'bracket' → [PERSON] | 'tag' → <PERSON/>
    ) -> None:
        self.db_path    = Path(db_path)
        self.mask_style = mask_style
        self._nlp       = None          # lazy-loaded spaCy model

        # Compile all regex patterns once at init (big performance win)
        self._compiled: dict[str, list[re.Pattern]] = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in PHI_PATTERNS.items()
        }

        if use_spacy:
            self._load_spacy()

        logger.info(
            "NERPipeline ready | spaCy=%s | mask_style=%s",
            "loaded" if self._nlp else "unavailable (regex-only mode)",
            self.mask_style,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_note(
        self,
        text: str,
        note_id: Optional[int] = None,
        save_to_db: bool = True,
    ) -> dict:
        """
        Full pipeline: extract entities → resolve overlaps → mask text.

        Parameters
        ----------
        text       : raw clinical note string
        note_id    : foreign key linking to clinical_notes table
        save_to_db : persist result to processed_notes table

        Returns
        -------
        {
          "note_id":        int | None,
          "original_text":  str,
          "masked_text":    str,
          "entities":       list[dict],   ← serialised PHIEntity list
          "entity_count":   int,
          "entity_types":   dict[str,int] ← label → count
        }
        """
        if not text or not text.strip():
            return self._empty_result(note_id, text)

        # 1. Regex layer — structured PHI
        entities = self._extract_regex(text)

        # 2. spaCy layer — unstructured PHI (names, locations, orgs)
        if self._nlp:
            entities += self._extract_spacy(text)

        # 3. Resolve overlaps (longest span wins)
        entities = self._resolve_overlaps(entities)

        # 4. Mask text (replace spans with [LABEL] tokens)
        masked_text = self._mask(text, entities)

        # 5. Build result dict
        entity_dicts = [
            {"label": e.label, "text": e.text, "start": e.start,
             "end": e.end, "source": e.source}
            for e in entities
        ]
        entity_types: dict[str, int] = {}
        for e in entities:
            entity_types[e.label] = entity_types.get(e.label, 0) + 1

        result = {
            "note_id":       note_id,
            "original_text": text,
            "masked_text":   masked_text,
            "entities":      entity_dicts,
            "entity_count":  len(entities),
            "entity_types":  entity_types,
        }

        # 6. Persist to DB
        if save_to_db and note_id is not None:
            self._save_processed(result)

        return result

    def process_batch(
        self,
        notes: list[dict],
        text_col: str = "transcription",
        id_col: str = "note_id",
    ) -> list[dict]:
        """
        Process a list of note dicts (e.g. from DataLoader.load_from_db()).

        Parameters
        ----------
        notes    : list of dicts with at least text_col and id_col keys
        text_col : column name for the note text
        id_col   : column name for the note ID

        Returns list of result dicts.
        """
        results = []
        total = len(notes)
        for i, note in enumerate(notes, 1):
            text    = note.get(text_col, "")
            note_id = note.get(id_col)
            result  = self.process_note(text, note_id=note_id, save_to_db=True)
            results.append(result)
            if i % 50 == 0 or i == total:
                logger.info("Processed %d / %d notes", i, total)
        logger.info(
            "Batch complete | %d notes | %d total entities found",
            total, sum(r["entity_count"] for r in results)
        )
        return results

    def evaluate(self, results: list[dict]) -> dict:
        """
        Aggregate statistics across a batch of processed results.
        Useful for the Flask /dashboard route and README metrics.
        """
        total_notes    = len(results)
        total_entities = sum(r["entity_count"] for r in results)
        notes_with_phi = sum(1 for r in results if r["entity_count"] > 0)

        # Entity type breakdown
        type_counts: dict[str, int] = {}
        for r in results:
            for label, count in r["entity_types"].items():
                type_counts[label] = type_counts.get(label, 0) + count

        # Avg entities per note (only notes that had any)
        phi_notes = [r for r in results if r["entity_count"] > 0]
        avg_per_phi_note = (
            sum(r["entity_count"] for r in phi_notes) / len(phi_notes)
            if phi_notes else 0
        )

        stats = {
            "total_notes":       total_notes,
            "notes_with_phi":    notes_with_phi,
            "phi_rate":          f"{notes_with_phi/total_notes*100:.1f}%" if total_notes else "0%",
            "total_entities":    total_entities,
            "avg_per_phi_note":  round(avg_per_phi_note, 1),
            "entity_breakdown":  dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        }
        logger.info("Evaluation: %s", stats)
        return stats

    # ── Private: extraction ───────────────────────────────────────────────────

    def _extract_regex(self, text: str) -> list[PHIEntity]:
        """Run all compiled regex patterns and return PHIEntity list."""
        found = []
        for label, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    found.append(PHIEntity(
                        label=label,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        source="regex",
                    ))
        return found

    def _extract_spacy(self, text: str) -> list[PHIEntity]:
        """Run spaCy NER and map standard labels to our PHI taxonomy."""
        doc = self._nlp(text)
        found = []
        for ent in doc.ents:
            phi_label = SPACY_LABEL_MAP.get(ent.label_)
            if phi_label is None:
                continue    # ignore non-PHI entity types (e.g. MONEY, PRODUCT)

            # Filter false positives: skip very short tokens and pure numbers
            if len(ent.text.strip()) < 3 or ent.text.strip().isdigit():
                continue

            found.append(PHIEntity(
                label=phi_label,
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                source="spacy",
            ))
        return found

    # ── Private: overlap resolution ───────────────────────────────────────────

    def _resolve_overlaps(self, entities: list[PHIEntity]) -> list[PHIEntity]:
        """
        When two detected spans overlap, keep the LONGER one.

        Strategy:
          - Sort by start offset, then by span length descending
          - Iterate: accept a span only if it doesn't overlap with
            the last accepted span
          - This is O(n log n) — fast enough for any note length

        Key decision: we prefer LONGER spans because they're usually
        more informative. "St. Mary's Hospital" beats "Mary" (PERSON
        false positive inside a hospital name).
        """
        if not entities:
            return []

        # Sort: primary = start, secondary = length DESC (longer first)
        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

        resolved = []
        last_end = -1

        for ent in sorted_ents:
            if ent.start >= last_end:
                resolved.append(ent)
                last_end = ent.end
            else:
                # Overlap — check if this span is longer than the last accepted
                if resolved and ent.end > resolved[-1].end:
                    resolved[-1] = ent     # replace with longer span
                    last_end = ent.end

        return resolved

    # ── Private: masking ──────────────────────────────────────────────────────

    def _mask(self, text: str, entities: list[PHIEntity]) -> str:
        """
        Replace each PHI span with a mask token.

        Key decision: we rebuild the string from right to left (reverse
        order of offsets) so replacing one span doesn't shift the character
        offsets of spans that come before it.
        """
        result = text
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            mask = self._make_mask(ent.label)
            result = result[:ent.start] + mask + result[ent.end:]
        return result

    def _make_mask(self, label: str) -> str:
        """Return the mask token for a given entity label."""
        if self.mask_style == "tag":
            return f"<{label}/>"
        return f"[{label}]"     # default bracket style

    # ── Private: database ─────────────────────────────────────────────────────

    def _save_processed(self, result: dict) -> None:
        """
        Persist de-identified note to processed_notes table.

        Schema:
          note_id           INTEGER  ← FK to clinical_notes
          masked_text       TEXT     ← de-identified note
          entity_count      INTEGER
          entity_types_json TEXT     ← JSON dict of label→count
          processed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """
        import json
        sql_create = """
            CREATE TABLE IF NOT EXISTS processed_notes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id           INTEGER,
                masked_text       TEXT,
                entity_count      INTEGER,
                entity_types_json TEXT,
                processed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        sql_migrate = """
            ALTER TABLE processed_notes ADD COLUMN masked_text TEXT
        """
        sql_insert = """
            INSERT INTO processed_notes
                (note_id, masked_text, entity_count, entity_types_json)
            VALUES (?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql_create)
            try:
                conn.execute(sql_migrate)
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute(sql_insert, (
                result["note_id"],
                result["masked_text"],
                result["entity_count"],
                json.dumps(result["entity_types"]),
            ))

    # ── Private: spaCy loader ─────────────────────────────────────────────────

    def _load_spacy(self) -> None:
        """
        Attempt to load spaCy model. Fails gracefully to regex-only mode.

        On your local machine:
            pip install spacy
            python -m spacy download en_core_web_sm
        """
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to regex-only NER (structured PHI still detected)."
            )
        except ImportError:
            logger.warning("spaCy not installed. Run: pip install spacy")

    @staticmethod
    def _empty_result(note_id: Optional[int], text: str) -> dict:
        return {
            "note_id": note_id, "original_text": text, "masked_text": text,
            "entities": [], "entity_count": 0, "entity_types": {},
        }
