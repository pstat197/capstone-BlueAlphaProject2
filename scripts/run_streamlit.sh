#!/usr/bin/env bash
# Run Streamlit with this repo's .venv so the Bayesian MMM tab sees google-meridian / TensorFlow.
# Usage (from repo root): ./scripts/run_streamlit.sh   [-- script args passed to streamlit]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_PY="$ROOT/.venv/bin/python"
ST="$ROOT/.venv/bin/streamlit"
if [[ ! -x "$ST" ]]; then
  echo "Missing $ST — create the venv and install deps first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -e \".\" && .venv/bin/pip install -r requirements-meridian.txt"
  echo "  # or: .venv/bin/pip install -e \".[mmm]\""
  exit 1
fi
exec "$ST" run app/streamlit_app.py "$@"
