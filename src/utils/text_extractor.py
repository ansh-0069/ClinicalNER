"""
text_extractor.py
─────────────────
Utility for extracting plain text from unstructured clinical documents.

Supported input types
─────────────────────
  - Plain text  (.txt, raw strings)
  - PDF         (.pdf)  — via pdfminer.six (optional dependency)
  - DOCX        (.docx) — via python-docx (optional dependency)
  - HTML        (.html, .htm) — via built-in html.parser
  - JSON / JSONL — extracts known text fields (HL7/FHIR note fragments)

Design decisions
────────────────
  - Each extractor is a small strategy function — easy to add new formats.
  - Missing optional dependencies degrade gracefully with an ImportError hint.
  - Output is always a normalised UTF-8 string suitable for NERPipeline.process().
  - The public API is a single `extract(source, ...)` function that dispatches
    on file extension or explicit `format` kwarg.
"""

from __future__ import annotations

import html as html_module
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

# HL7/FHIR JSON keys that typically contain clinical free text
_FHIR_TEXT_KEYS = ("text", "div", "note", "description", "reasonCode", "conclusion")

# Regex to collapse whitespace and control characters
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE    = re.compile(r"\n{3,}")
_CTRL_RE       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Public API ─────────────────────────────────────────────────────────────────

def extract(source: str | Path, *, fmt: str | None = None) -> str:
    """
    Extract clean plain text from a file path or raw string.

    Parameters
    ----------
    source : str | Path
        Either a file path or a raw text/HTML/JSON string.
    fmt : 'txt' | 'pdf' | 'docx' | 'html' | 'json' | 'jsonl' | None
        Override auto-detection from file extension.

    Returns
    -------
    str
        Normalised UTF-8 text ready for the NER pipeline.

    Examples
    --------
    >>> extract("patient.txt")
    'Patient presents with chest pain...'
    >>> extract("<p>Patient <b>John Doe</b></p>", fmt="html")
    'Patient John Doe'
    """
    path = Path(source) if isinstance(source, (str, Path)) and Path(source).exists() else None

    if fmt is None:
        if path is not None:
            fmt = path.suffix.lstrip(".").lower()
        else:
            # Heuristic: try to detect from content
            src_str = str(source)
            if src_str.lstrip().startswith("<"):
                fmt = "html"
            elif src_str.lstrip().startswith(("{", "[")):
                fmt = "json"
            else:
                fmt = "txt"

    if fmt == "pdf":
        raw = _extract_pdf(path or source)
    elif fmt == "docx":
        raw = _extract_docx(path or source)
    elif fmt in ("html", "htm"):
        content = path.read_text(encoding="utf-8", errors="replace") if path else str(source)
        raw = _extract_html(content)
    elif fmt in ("json",):
        content = path.read_text(encoding="utf-8", errors="replace") if path else str(source)
        raw = _extract_json(content)
    elif fmt == "jsonl":
        content = path.read_text(encoding="utf-8", errors="replace") if path else str(source)
        raw = _extract_jsonl(content)
    else:  # txt or unknown — treat as plain text
        raw = path.read_text(encoding="utf-8", errors="replace") if path else str(source)

    return _normalise(raw)


def extract_many(sources: list[str | Path], *, fmt: str | None = None) -> list[str]:
    """Batch extraction — returns one cleaned string per source."""
    return [extract(s, fmt=fmt) for s in sources]


# ── Normalisation ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Remove control chars, collapse whitespace, strip leading/trailing space."""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ── Extractors ─────────────────────────────────────────────────────────────────

def _extract_html(content: str) -> str:
    """Strip HTML tags via stdlib html.parser; decode HTML entities."""
    from html.parser import HTMLParser

    class _TextCollector(HTMLParser):
        SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in self.SKIP_TAGS:
                self._skip += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in self.SKIP_TAGS:
                self._skip = max(0, self._skip - 1)
            if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3"):
                self._parts.append("\n")

        def handle_data(self, data: str) -> None:
            if self._skip == 0:
                self._parts.append(data)

    parser = _TextCollector()
    parser.feed(content)
    return " ".join(parser._parts)


def _extract_json(content: str) -> str:
    """Recursively extract text values from known clinical JSON keys."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON content: {exc}") from exc
    return "\n".join(_walk_json(obj))


def _extract_jsonl(content: str) -> str:
    """Extract text from a newline-delimited JSON (JSONL) file."""
    parts: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parts.extend(_walk_json(obj))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
    return "\n".join(parts)


def _walk_json(obj: Any, depth: int = 0) -> list[str]:
    """DFS through a JSON object collecting text fields."""
    if depth > 20:
        return []
    if isinstance(obj, str):
        # If it looks like HTML div (FHIR narrative), strip tags
        clean = _extract_html(obj) if obj.lstrip().startswith("<") else obj
        return [clean] if clean.strip() else []
    if isinstance(obj, dict):
        results: list[str] = []
        for k, v in obj.items():
            if k.lower() in _FHIR_TEXT_KEYS:
                results.extend(_walk_json(v, depth + 1))
            elif isinstance(v, (dict, list)):
                results.extend(_walk_json(v, depth + 1))
        return results
    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            out.extend(_walk_json(item, depth + 1))
        return out
    return []


def _extract_pdf(source: str | Path) -> str:
    """Extract text from a PDF using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text as _pdf_extract
    except ImportError as exc:
        raise ImportError(
            "pdfminer.six is required for PDF extraction. "
            "Install it with: pip install pdfminer.six"
        ) from exc

    path = Path(source) if not isinstance(source, Path) else source
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    return _pdf_extract(str(path)) or ""


def _extract_docx(source: str | Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        ) from exc

    path = Path(source) if not isinstance(source, Path) else source
    if not path.is_file():
        raise FileNotFoundError(f"DOCX not found: {path}")

    document = docx.Document(str(path))
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)
