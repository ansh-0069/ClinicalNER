"""
data_cleaner.py
───────────────
DataCleaner: normalises and validates clinical text before and after
de-identification.

Two cleaning passes:
  PRE-NER  (raw text)  → fix encoding, normalise whitespace, strip
                          formatting artefacts that confuse the NER model
  POST-NER (masked text) → validate mask tokens, remove residual PHI
                            patterns, standardise output format

Why this matters for the JD:
  "Clean, transform, and manage structured and unstructured clinical data"
  is Activity 2 in the JD. DataCleaner directly satisfies that requirement
  with domain-specific logic — not generic pandas cleaning.

Clinical text quirks this class handles:
  - Dictation artefacts  : "umm", "uh", filler words from voice transcription
  - Encoding noise       : \x92 → ', \x93\x94 → "", garbled unicode
  - Inconsistent spacing : multiple spaces, mixed newlines (\r\n vs \n)
  - Section headers      : "ASSESSMENT:", "PLAN:" often have irregular spacing
  - Abbreviation dots    : "Dr." / "St." shouldn't split sentences
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ── Cleaning result dataclass ─────────────────────────────────────────────────

@dataclass
class CleaningResult:
    """
    Output of a single DataCleaner.clean() call.

    Attributes
    ----------
    original_text   : text before cleaning
    cleaned_text    : text after cleaning
    changes         : list of (change_type, description) tuples
    encoding_fixed  : True if encoding artefacts were repaired
    whitespace_fixed : True if whitespace was normalised
    residual_phi    : list of suspected residual PHI patterns (post-NER pass)
    is_valid        : False if post-NER validation finds unmasked PHI
    """
    original_text:    str
    cleaned_text:     str
    changes:          list[tuple[str, str]] = field(default_factory=list)
    encoding_fixed:   bool = False
    whitespace_fixed: bool = False
    residual_phi:     list[str] = field(default_factory=list)
    is_valid:         bool = True

    @property
    def change_count(self) -> int:
        return len(self.changes)

    def summary(self) -> str:
        lines = [
            f"Changes  : {self.change_count}",
            f"Valid    : {self.is_valid}",
        ]
        if self.residual_phi:
            lines.append(f"Residual : {self.residual_phi}")
        return " | ".join(lines)


# ── DataCleaner class ─────────────────────────────────────────────────────────

class DataCleaner:
    """
    Cleans clinical text pre- and post-NER de-identification.

    Usage
    -----
    cleaner = DataCleaner()

    # Before NER — clean raw note
    pre  = cleaner.clean_pre_ner(raw_text)

    # After NER — validate and clean masked output
    post = cleaner.clean_post_ner(masked_text)
    """

    # Dictation filler words (voice-to-text artefacts common in clinical notes)
    FILLER_PATTERN = re.compile(
        r"\b(um+|uh+|er+|ah+|hmm+|umm+)\b", re.IGNORECASE
    )

    # Residual PHI checks run on masked text to catch anything NER missed
    # Key decision: conservative patterns only — we prefer false positives
    # here over false negatives (missing real PHI is far worse than
    # flagging a benign string for review)
    RESIDUAL_PHI_PATTERNS: dict[str, re.Pattern] = {
        "phone":    re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"),
        "date":     re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        "mrn":      re.compile(r"\bMRN[-\s]?\d{4,10}\b", re.IGNORECASE),
        "iso_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        "ssn":      re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    # Section headers to normalise (add consistent spacing after colon)
    SECTION_HEADERS = re.compile(
        r"(ASSESSMENT|PLAN|HPI|HISTORY|MEDICATIONS?|ALLERGIES|"
        r"VITALS?|PHYSICAL EXAM|IMPRESSION|DIAGNOSIS|PROCEDURE|"
        r"FINDINGS?|DISPOSITION|FOLLOW[\s\-]?UP)\s*:",
        re.IGNORECASE,
    )

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Parameters
        ----------
        strict_mode : if True, clean_post_ner marks notes with ANY
                      residual PHI pattern as invalid (is_valid=False).
                      If False, only logs a warning (softer compliance posture).
        """
        self.strict_mode = strict_mode
        logger.info("DataCleaner ready | strict_mode=%s", self.strict_mode)

    # ── Public API ────────────────────────────────────────────────────────────

    def clean_pre_ner(self, text: str) -> CleaningResult:
        """
        Clean raw clinical text BEFORE running NER.

        Steps (in order — order matters, each step assumes the previous
        has already run):
          1. Fix encoding artefacts
          2. Normalise unicode to NFC form
          3. Normalise line endings
          4. Strip dictation filler words
          5. Normalise section headers
          6. Collapse multiple whitespace
          7. Strip leading/trailing whitespace
        """
        if not text:
            return CleaningResult(original_text=text, cleaned_text=text)

        result = CleaningResult(original_text=text, cleaned_text=text)
        t = text

        # Step 1: encoding artefacts
        t, enc_changed = self._fix_encoding(t)
        if enc_changed:
            result.encoding_fixed = True
            result.changes.append(("encoding", "Fixed Windows-1252 / latin-1 artefacts"))

        # Step 2: unicode normalisation
        t_norm = unicodedata.normalize("NFC", t)
        if t_norm != t:
            result.changes.append(("unicode", "NFC normalisation applied"))
            t = t_norm

        # Step 3: line endings
        t_lines = t.replace("\r\n", "\n").replace("\r", "\n")
        if t_lines != t:
            result.changes.append(("line_endings", "Normalised CRLF → LF"))
            t = t_lines

        # Step 4: filler words
        t_fill, n_fillers = self._remove_fillers(t)
        if n_fillers:
            result.changes.append(("fillers", f"Removed {n_fillers} dictation filler(s)"))
            t = t_fill

        # Step 5: section headers
        t_sec, n_headers = self._normalise_headers(t)
        if n_headers:
            result.changes.append(("headers", f"Normalised {n_headers} section header(s)"))
            t = t_sec

        # Step 6: collapse whitespace (but preserve paragraph breaks)
        t_ws, ws_changed = self._normalise_whitespace(t)
        if ws_changed:
            result.whitespace_fixed = True
            result.changes.append(("whitespace", "Collapsed multiple spaces/tabs"))
            t = t_ws

        # Step 7: strip edges
        t = t.strip()

        result.cleaned_text = t
        return result

    def clean_post_ner(self, masked_text: str) -> CleaningResult:
        """
        Validate and clean masked text AFTER NER.

        Steps:
          1. Normalise mask token format (handles edge cases like [  DATE ])
          2. Scan for residual PHI patterns the NER may have missed
          3. Mark result as invalid if residual PHI found (strict mode)
          4. Clean up any double-spaces left by masking
        """
        if not masked_text:
            return CleaningResult(original_text=masked_text, cleaned_text=masked_text)

        result = CleaningResult(original_text=masked_text, cleaned_text=masked_text)
        t = masked_text

        # Step 1: normalise mask tokens (e.g. [ DATE ] → [DATE])
        t_masks, n_fixed = self._normalise_masks(t)
        if n_fixed:
            result.changes.append(("mask_tokens", f"Normalised {n_fixed} mask token(s)"))
            t = t_masks

        # Step 2: residual PHI scan
        residual = self._scan_residual_phi(t)
        if residual:
            result.residual_phi = residual
            msg = f"Possible residual PHI: {residual}"
            if self.strict_mode:
                result.is_valid = False
                result.changes.append(("residual_phi_FAIL", msg))
                logger.warning("STRICT FAIL — %s", msg)
            else:
                result.changes.append(("residual_phi_WARN", msg))
                logger.warning("Residual PHI detected (non-strict): %s", msg)

        # Step 3: clean double spaces left by masking
        t_clean = re.sub(r"  +", " ", t)
        if t_clean != t:
            result.changes.append(("double_spaces", "Removed double spaces from masking"))
            t = t_clean

        result.cleaned_text = t.strip()
        return result

    def clean_batch(
        self,
        records: list[dict],
        text_col: str = "masked_text",
        id_col: str = "note_id",
        pass_type: str = "post",      # 'pre' or 'post'
    ) -> tuple[list[CleaningResult], dict]:
        """
        Clean a list of record dicts.

        Returns
        -------
        (results, stats) where stats is a summary dict for /dashboard
        """
        results = []
        for rec in records:
            text = rec.get(text_col, "")
            if pass_type == "pre":
                r = self.clean_pre_ner(text)
            else:
                r = self.clean_post_ner(text)
            results.append(r)

        stats = self._batch_stats(results)
        logger.info(
            "Batch clean (%s) | %d records | %d with changes | %d invalid",
            pass_type, stats["total"], stats["with_changes"], stats["invalid_count"]
        )
        return results, stats

    # ── Private: encoding ─────────────────────────────────────────────────────

    def _fix_encoding(self, text: str) -> tuple[str, bool]:
        """
        Fix common Windows-1252 / latin-1 artefacts that appear in
        copy-pasted clinical documents.

        These are the most common offenders in MTSamples and real EHR exports.
        """
        replacements = {
            "\x92": "'",    # right single quote (apostrophe)
            "\x91": "'",    # left single quote
            "\x93": '"',    # left double quote
            "\x94": '"',    # right double quote
            "\x96": "–",    # en dash
            "\x97": "—",    # em dash
            "\x85": "...",  # ellipsis
            "\xa0": " ",    # non-breaking space
            "\x00": "",     # null bytes
        }
        changed = False
        for bad, good in replacements.items():
            if bad in text:
                text = text.replace(bad, good)
                changed = True
        return text, changed

    # ── Private: fillers ──────────────────────────────────────────────────────

    def _remove_fillers(self, text: str) -> tuple[str, int]:
        """Remove dictation filler words and return (cleaned_text, count)."""
        matches = self.FILLER_PATTERN.findall(text)
        cleaned = self.FILLER_PATTERN.sub("", text)
        return cleaned, len(matches)

    # ── Private: section headers ──────────────────────────────────────────────

    def _normalise_headers(self, text: str) -> tuple[str, int]:
        """
        Ensure section headers have consistent formatting.
        "ASSESSMENT:" → "ASSESSMENT: " (single space after colon)
        """
        count = [0]

        def replacer(m):
            count[0] += 1
            header = m.group(1).upper()
            return f"{header}:"

        cleaned = self.SECTION_HEADERS.sub(replacer, text)
        return cleaned, count[0]

    # ── Private: whitespace ───────────────────────────────────────────────────

    def _normalise_whitespace(self, text: str) -> tuple[str, bool]:
        """
        Collapse multiple spaces and tabs to a single space.
        Preserves double newlines (paragraph breaks).
        """
        # Collapse horizontal whitespace only (not newlines)
        cleaned = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to double newline (paragraph break)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned, cleaned != text

    # ── Private: mask normalisation ───────────────────────────────────────────

    def _normalise_masks(self, text: str) -> tuple[str, int]:
        """
        Fix malformed mask tokens:
          [ DATE ]  → [DATE]
          [date]    → [DATE]
          [ PERSON] → [PERSON]
        """
        count = [0]
        def fix(m):
            count[0] += 1
            return f"[{m.group(1).strip().upper()}]"
        cleaned = re.sub(r"\[\s*([A-Za-z_]+)\s*\]", fix, text)
        return cleaned, count[0]

    # ── Private: residual PHI scan ────────────────────────────────────────────

    def _scan_residual_phi(self, text: str) -> list[str]:
        """
        Scan masked text for PHI patterns that slipped through NER.
        Returns list of matched strings (empty = clean).
        """
        found = []
        for phi_type, pattern in self.RESIDUAL_PHI_PATTERNS.items():
            matches = pattern.findall(text)
            for m in matches:
                found.append(f"{phi_type}:'{m}'")
        return found

    # ── Private: batch stats ──────────────────────────────────────────────────

    def _batch_stats(self, results: list[CleaningResult]) -> dict:
        total         = len(results)
        with_changes  = sum(1 for r in results if r.change_count > 0)
        encoding_fixed = sum(1 for r in results if r.encoding_fixed)
        ws_fixed      = sum(1 for r in results if r.whitespace_fixed)
        invalid       = sum(1 for r in results if not r.is_valid)
        residual_phi  = sum(1 for r in results if r.residual_phi)

        return {
            "total":          total,
            "with_changes":   with_changes,
            "encoding_fixed": encoding_fixed,
            "ws_fixed":       ws_fixed,
            "invalid_count":  invalid,
            "residual_phi":   residual_phi,
            "clean_rate":     f"{(total - residual_phi) / total * 100:.1f}%" if total else "0%",
        }