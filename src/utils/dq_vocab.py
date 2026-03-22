"""
dq_vocab.py
───────────
Data-quality vocabulary loaders for DQP conformity checks (specialty codelist).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import FrozenSet

logger = logging.getLogger(__name__)


def specialty_vocab_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "data" / "specialty_vocab.json"


def load_specialty_vocab(project_root: Path | None = None) -> FrozenSet[str]:
    """Return frozen set of allowed medical_specialty values (case-sensitive)."""
    path = specialty_vocab_path(project_root)
    if not path.exists():
        logger.warning("specialty_vocab.json missing at %s — conformity check skipped", path)
        return frozenset()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("allowed_specialties", [])
        return frozenset(str(x).strip() for x in items if x)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load specialty vocab: %s", exc)
        return frozenset()
