# ==========================================================================
# GenAI Research Platform — Multi-stage Dockerfile
# Works on: Docker, Railway, Render, Fly.io, Google Cloud Run, AWS ECS,
#           DigitalOcean App Platform, any Docker host
# ==========================================================================

# ---------- Stage 1: Builder ----------
FROM python:3.13-slim AS builder

# System deps needed to build Python C extensions (rdkit, PyMuPDF, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.13-slim AS runtime

# System deps needed at runtime
# poppler-utils  -> pdf2image
# libgomp1       -> numpy / scikit-learn OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY . .

# Create upload directory (writable by appuser)
RUN mkdir -p backend/uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Use shell form so ${VAR:-default} works in CMD
SHELL ["/bin/sh", "-c"]
CMD gunicorn wsgi:application \
    --bind "0.0.0.0:${PORT:-5000}" \
    --workers "${GUNICORN_WORKERS:-2}" \
    --timeout "${GUNICORN_TIMEOUT:-120}" \
    --access-logfile - \
    --error-logfile -
