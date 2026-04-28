#!/usr/bin/env bash
set -euo pipefail

echo "DMD Agent 4 All installer"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required." >&2
  exit 1
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Error: Python 3.11 or newer is required.")
PY

echo "Python version: ${PYTHON_VERSION}"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment: .venv"
  python3 -m venv .venv
fi

echo "Installing package in editable mode"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

echo ""
echo "Initializing local data"
.venv/bin/dmdagent init

echo ""
.venv/bin/dmdagent wizard

echo ""
echo "Install complete."
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Start the local API with:"
echo "  dmdagent serve"
