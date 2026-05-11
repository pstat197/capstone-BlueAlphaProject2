#!/usr/bin/env bash
# Run the React UI dev stack: FastAPI backend (port 8000) + Vite frontend (port 5173).
# Use Ctrl-C to stop both. The existing Streamlit app is unaffected.
#
# Usage (from repo root):
#   ./scripts/run_dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PY="$ROOT/.venv/bin/python"
UVICORN="$ROOT/.venv/bin/uvicorn"

if [[ ! -x "$UVICORN" ]]; then
  echo "Missing $UVICORN — install the API extras first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -e \".[api]\""
  exit 1
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Missing frontend/node_modules — install first:"
  echo "  npm install --prefix frontend"
  exit 1
fi

cleanup() {
  trap - INT TERM
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting FastAPI on http://127.0.0.1:8000"
"$UVICORN" server.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Vite on  http://localhost:5173 (proxies /api -> 127.0.0.1:8000)"
( cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1 --port 5173 ) &
FRONTEND_PID=$!

wait
