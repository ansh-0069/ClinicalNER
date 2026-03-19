"""
routes.py
─────────
Phase 4: API route definitions.

Endpoints:
  POST   /api/notes          - Upload clinical note
  GET    /api/notes/:id      - Retrieve note
  POST   /api/process        - Run NER pipeline
  GET    /api/dashboard      - EDA visualizations
  GET    /api/audit          - Audit log
"""

from __future__ import annotations

import logging
from flask import Blueprint

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/notes", methods=["POST"])
def upload_note():
    """Upload a clinical note for processing."""
    raise NotImplementedError("Phase 4: Note upload to be implemented")


@api_bp.route("/process", methods=["POST"])
def process_note():
    """Process note through NER pipeline."""
    raise NotImplementedError("Phase 4: NER processing to be implemented")
