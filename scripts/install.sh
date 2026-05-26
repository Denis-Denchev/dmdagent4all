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
.venv/bin/dmdcore init

echo ""
.venv/bin/dmdcore wizard

echo ""
echo "Install complete."
echo "Start terminal chat from this folder with:"
echo "  ./start session"
echo ""
echo "First start asks for agent name, your name, language, and model/provider."
echo ""
echo "Install the optional shell shortcut with:"
echo "  ./start install-command"
echo ""
echo "Then open a new terminal and run:"
echo "  start session"
