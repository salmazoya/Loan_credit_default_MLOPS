# ─────────────────────────────────────────────────────────────────────────────
# Loan Credit Default Risk — FastAPI Inference Container
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t loan-default-api .
# Run:    docker run -p 8080:8080 \
#           -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp_key.json \
#           -v $(pwd)/credentials:/app/credentials \
#           loan-default-api
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# libgomp1 is required by LightGBM (OpenMP)

# ── Create non-root user for security ────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY config/     ./config/
COPY src/        ./src/
COPY utils/      ./utils/
COPY main.py     .
COPY setup.py    .

# ── Create artifact directories ───────────────────────────────────────────────
# Model pkl files are NOT baked into the image — they are pulled from GCP
# at container startup via the entrypoint script
RUN mkdir -p artifacts/raw \
             artifacts/processed \
             artifacts/models \
             artifacts/plots \
             artifacts/monitoring \
             logs

# ── Entrypoint script — pulls models from GCP then starts API ─────────────────
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# ── Ownership ─────────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# ── Start ─────────────────────────────────────────────────────────────────────
ENTRYPOINT ["./docker-entrypoint.sh"]
