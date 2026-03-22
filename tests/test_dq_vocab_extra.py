"""Edge cases for dq_vocab."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.dq_vocab import load_specialty_vocab, specialty_vocab_path


def test_missing_vocab_file_returns_empty(tmp_path):
    assert load_specialty_vocab(tmp_path) == frozenset()


def test_specialty_vocab_path():
    p = specialty_vocab_path()
    assert p.name == "specialty_vocab.json"


def test_corrupt_vocab_json_returns_empty(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "specialty_vocab.json").write_text("{not valid json", encoding="utf-8")
    assert load_specialty_vocab(tmp_path) == frozenset()
