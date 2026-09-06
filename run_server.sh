#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_server.sh  —  Start the Privacy-Safe OCR + PII Anonymisation Service
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# ── Activate virtual environment ──────────────────────────────────────────────
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "[run_server] Activating virtual environment: $VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
else
    echo "[run_server] WARNING: No venv found at $VENV_DIR"
    echo "[run_server] Ensure you have run: python3.11 -m venv venv && pip install -r requirements.txt"
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "[run_server] Python: $(python --version)"
echo "[run_server] Tesseract: $(tesseract --version 2>&1 | head -1)"

# ── Start server ──────────────────────────────────────────────────────────────
echo "[run_server] Starting backend on http://127.0.0.1:5005"
exec python "$SCRIPT_DIR/backend/app.py"
