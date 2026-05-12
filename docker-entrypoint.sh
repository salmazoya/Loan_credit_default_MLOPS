#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# docker-entrypoint.sh
# Pulls the latest model artifacts from GCP then starts the FastAPI server.
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "========================================"
echo "  Loan Default Risk API — Starting Up"
echo "========================================"

# ── Pull model artifacts from GCP ────────────────────────────────────────────
# Only pull models (not raw/processed data — not needed for inference)
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "[1/2] Pulling model artifacts from GCP..."
    python -m src.gcp_sync --pull --group models
    echo "      Model artifacts ready."
else
    echo "[1/2] No GCP credentials found — using local model artifacts."
    echo "      Set GOOGLE_APPLICATION_CREDENTIALS to pull from GCP."
fi

# ── Verify model artifacts exist ─────────────────────────────────────────────
if [ ! -f "artifacts/models/lgbm_model.pkl" ]; then
    echo "ERROR: lgbm_model.pkl not found in artifacts/models/"
    echo "       Run 'python -m src.train' first, or provide GCP credentials."
    exit 1
fi

# ── Start FastAPI server ──────────────────────────────────────────────────────
echo "[2/2] Starting FastAPI server on port 8080..."
echo "========================================"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 2 \
    --log-level info
