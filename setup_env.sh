#!/usr/bin/env bash
# One-shot environment setup (Linux / macOS).
#
#   ./setup_env.sh          # runtime only  → .venv
#   ./setup_env.sh --dev    # + pytest
#
# Creates .venv with Python 3.12 (.python-version), installs the exact pinned
# versions from requirements-lock.txt, then verifies them with tools/check_env.py.
# Afterwards ALWAYS run the project with this interpreter:
#   .venv/bin/python main.py ...
#   .venv/bin/python -m streamlit run ui/app.py
set -euo pipefail
cd "$(dirname "$0")"

PY_VER="$(tr -d '[:space:]' < .python-version)"
if [ ! -x .venv/bin/python ]; then
  command -v "python${PY_VER}" >/dev/null || { echo "python${PY_VER} not found — install it first"; exit 1; }
  echo "Creating .venv with Python ${PY_VER} ..."
  "python${PY_VER}" -m venv .venv
fi
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
# The lock was produced on Windows; its only Windows-specific entry (colorama) is pure Python
# and harmless here. If a Linux-only transitive dep is reported missing, add it and re-freeze.
"$PY" -m pip install -r requirements-lock.txt
if [ "${1:-}" = "--dev" ]; then
  "$PY" -m pip install -r requirements-dev.txt
  "$PY" tools/check_env.py --dev
else
  "$PY" tools/check_env.py
fi
echo "Done. Use: $PY -m streamlit run ui/app.py"
