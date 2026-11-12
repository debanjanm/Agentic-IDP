#!/usr/bin/env bash
# Backend: sets up its own virtualenv on first run, then runs the API and the
# worker together. Ctrl-C stops both.
#
#   ./backend.sh                    API on http://127.0.0.1:8000
#   PORT=9000 ./backend.sh          another port
#   PYTHON=/path/to/python3.12 ./backend.sh
#
# Needs Ollama running with qwen2.5vl:3b and qwen3:4b pulled (see README).
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv/backend
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

# Tested on Python 3.12. paddlepaddle is the fussy one about interpreter versions.
DEPS=(
  fastapi "uvicorn[standard]" python-multipart httpx jsonschema
  pymupdf==1.26.7 paddlepaddle==3.3.1 paddleocr==3.7.0
)

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

# The dependency list above is the source of truth: reinstall only when this
# script is newer than the last successful install.
STAMP=$VENV/.deps-installed
if [ ! -f "$STAMP" ] || [ "$(basename "$0")" -nt "$STAMP" ]; then
  echo "Installing backend dependencies (first run downloads PaddlePaddle, a few minutes)..."
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "${DEPS[@]}"
  touch "$STAMP"
fi

# Say now, not mid-job, if the model server isn't ready.
"$VENV/bin/python" -c 'from app import llm; ok, why = llm.available(); not ok and print("warning:", why)' >&2 || true

"$VENV/bin/python" -m app.worker &
WORKER=$!
"$VENV/bin/python" -m uvicorn app.main:app --host "$HOST" --port "$PORT" &
API=$!

# Both children belong to this script: however it ends (Ctrl-C, kill, the API
# crashing) neither is left running. Without this, `kill <script>` orphans uvicorn.
trap 'kill "$WORKER" "$API" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM

echo "API: http://$HOST:$PORT   (api pid $API, worker pid $WORKER)"
wait "$API"
