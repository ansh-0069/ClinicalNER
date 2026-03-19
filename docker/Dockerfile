# Dockerfile
# Phase 5 — ClinicalNER production container
#
# Design decisions:
#   - python:3.11-slim  : 50% smaller than the full image, no GUI libs needed
#   - Multi-step RUN    : each layer cached independently → fast rebuilds when
#                         only code changes (requirements layer stays cached)
#   - spaCy model baked : downloaded at BUILD time, not boot time — container
#                         starts in ~1 s, not 30 s
#   - HEALTHCHECK       : Docker restarts the container automatically if /health
#                         stops returning 200 — same signal used by AWS ECS / GCP
#                         Cloud Run load balancers
#   - Non-root user     : industry standard for container security hardening
#   - PYTHONDONTWRITEBYTECODE / PYTHONUNBUFFERED : keep logs clean and immediate

FROM python:3.11-slim

# ── OS packages ───────────────────────────────────────────────────────────────
# gcc  : needed to compile some Python C extensions
# curl : used in HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python env flags ──────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
# Copy requirements FIRST so pip layer is rebuilt only when deps change,
# not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model (cached layer) ────────────────────────────────────────────────
# Install model via direct wheel URL — avoids the `python -m spacy download`
# CLI resolver bug that produces a malformed "-en_core_web_sm" URL in Docker.
# Pinned to en_core_web_sm-3.7.1 (compatible with spacy==3.7.4).
# This layer is cached and only rebuilt when the spaCy version changes.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN useradd -m -u 1000 clinicalner && chown -R clinicalner /app
USER clinicalner

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Health check (called by Docker daemon every 30 s) ─────────────────────────
# --fail : return non-zero on HTTP error codes (triggers unhealthy status)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# entrypoint.sh seeds the DB on first boot, then starts Flask
COPY docker/entrypoint.sh /entrypoint.sh
# Switch to root briefly to chmod, then back to app user
USER root
RUN chmod +x /entrypoint.sh
USER clinicalner

ENTRYPOINT ["/entrypoint.sh"]
