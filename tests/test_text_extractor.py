"""Tests for text extraction utilities."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.text_extractor import extract, extract_many, _normalise


def test_extract_plain_string():
    assert "hello" in extract("hello world", fmt="txt").lower()


def test_extract_html_string():
    out = extract("<p>Hello <b>World</b></p>", fmt="html")
    assert "Hello" in out and "World" in out


def test_extract_json_fhir_like():
    payload = json.dumps({"text": {"div": "<div>Clinical note text</div>"}})
    out = extract(payload, fmt="json")
    assert "Clinical" in out or "note" in out.lower()


def test_extract_jsonl():
    lines = json.dumps({"note": "alpha"}) + "\n" + json.dumps({"note": "beta"})
    out = extract(lines, fmt="jsonl")
    assert "alpha" in out or "beta" in out


def test_extract_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        extract("{not json", fmt="json")


def test_extract_jsonl_bad_line_raises():
    with pytest.raises(ValueError, match="Invalid JSON on line"):
        extract('{"a":1}\n{broken', fmt="jsonl")


def test_extract_many():
    outs = extract_many(["a", "<p>b</p>"], fmt="txt")
    assert len(outs) == 2


def test_normalise_strips_control_chars():
    t = _normalise("hello\x00world")
    assert "\x00" not in t


def test_extract_from_txt_file(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("Plain text content here.", encoding="utf-8")
    assert "Plain text" in extract(p)


def test_extract_html_file(tmp_path):
    p = tmp_path / "x.html"
    p.write_text("<html><body><p>File HTML</p></body></html>", encoding="utf-8")
    out = extract(p)
    assert "File HTML" in out


def test_extract_auto_detect_html_from_string():
    out = extract("  <div>Auto</div>")
    assert "Auto" in out


def test_extract_auto_detect_json_from_string():
    out = extract('  {"text": "JSON content"}')
    assert "JSON" in out or "content" in out


def test_extract_pdf_invalid_or_unreadable(tmp_path):
    p = tmp_path / "f.pdf"
    p.write_bytes(b"%PDF-1.4")
    try:
        from pdfminer.pdfparser import PDFSyntaxError
    except ImportError:
        pytest.skip("pdfminer not installed")
    with pytest.raises(Exception):
        extract(p, fmt="pdf")


def test_extract_docx_invalid_package(tmp_path):
    p = tmp_path / "f.docx"
    p.write_bytes(b"PK\x03\x04")
    try:
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError:
        pytest.skip("python-docx not installed")
    with pytest.raises(PackageNotFoundError):
        extract(p, fmt="docx")
