#!/usr/bin/env bash
# Frontend: sets up its own virtualenv on first run, then runs the Streamlit UI.
# It only talks to the backend over HTTP, so it needs none of the backend's
# heavy dependencies (PaddlePaddle, PyMuPDF).
#
#   ./scripts/frontend.sh                           UI on http://localhost:8501
#   IDP_API=http://host:9000 ./scripts/frontend.sh  backend somewhere else
#   PORT=8600 ./scripts/frontend.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
ROOT=$PWD

VENV=$ROOT/.venv/frontend
PORT=${PORT:-8501}
export IDP_API=${IDP_API:-http://localhost:8000}

# ui.py calls st.image(use_column_width=...), which exists in this version.
DEPS=("streamlit==1.37.1" requests pandas)

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY=$candidate; break; fi
  done
fi
[ -n "$PY" ] || { echo "No Python found. Install 3.12, or set PYTHON=/path/to/python." >&2; exit 1; }

if [ ! -d "$VENV" ]; then
  echo "Creating $VENV with $("$PY" --version 2>&1)"
  "$PY" -m venv "$VENV"
fi

STAMP=$VENV/.deps-installed
if [ ! -f "$STAMP" ] || [ scripts/frontend.sh -nt "$STAMP" ]; then
  echo "Installing frontend dependencies..."
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "${DEPS[@]}"
  touch "$STAMP"
fi

curl -sf "$IDP_API/health" >/dev/null 2>&1 \
  || echo "warning: backend not reachable at $IDP_API - start ./scripts/backend.sh (the UI will say so too)" >&2

echo "UI: http://localhost:$PORT  (backend: $IDP_API)"
cd frontend  # Streamlit reads .streamlit/config.toml from here
exec "$VENV/bin/streamlit" run ui.py --server.port "$PORT"
