# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install deps in a separate stage so the final image stays lean.
# python:3.11-slim is ~50MB vs python:3.11 at ~900MB.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some Python packages (numpy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — Docker cache will reuse this layer
# on rebuilds as long as requirements.txt hasn't changed.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install --no-deps \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder — no build tools in final image
COPY --from=builder /install /usr/local

# Copy application source
COPY src/       ./src/
COPY run_phase1.py run_phase2.py run_phase3.py run_phase4.py ./
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# Create data directory for the SQLite database
RUN mkdir -p data

# Run as non-root user — security best practice for containers
RUN useradd --no-create-home --shell /bin/false appuser && \
    chmod +x /app/docker/entrypoint.sh && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV DB_PATH=/app/data/clinicalner.db

# Expose Flask port
EXPOSE 5000

# Health check — Docker will mark container unhealthy if this fails
# Uses Render's dynamic PORT and allows slower first boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD sh -c "python -c \"import os,urllib.request; p=os.getenv('PORT','5000'); urllib.request.urlopen(f'http://localhost:{p}/health')\""

# Entrypoint: seed database on first boot (without EDA) and start app
ENTRYPOINT ["/app/docker/entrypoint.sh"]
