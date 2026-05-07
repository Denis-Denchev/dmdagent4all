#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${DMDAGENT_REPO_URL:-https://github.com/Denis-Denchev/dmdagent4all.git}"
INSTALL_DIR="${DMDAGENT_INSTALL_DIR:-dmdagent4all}"
USE_DOCKER=0
DETACHED=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --docker)
      USE_DOCKER=1
      ;;
    --detached|-d)
      DETACHED=1
      ;;
    --dir)
      shift
      INSTALL_DIR="${1:-$INSTALL_DIR}"
      ;;
    --help|-h)
      cat <<'EOF'
DMD Agent bootstrap

Usage:
  bash bootstrap.sh                 Native local install
  bash bootstrap.sh --docker        Docker install, app + Ollama + model
  bash bootstrap.sh --docker -d     Docker install in background
  bash bootstrap.sh --dir my-agent  Install into custom folder
EOF
      exit 0
      ;;
    *)
      INSTALL_DIR="$1"
      ;;
  esac
  shift
done

info() {
  printf '%s\n' "$*"
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [ -f "pyproject.toml" ] && [ -f "start" ] && [ -d "src/dmdagent4all" ]; then
  PROJECT_DIR="$(pwd)"
else
  if ! command -v git >/dev/null 2>&1; then
    fail "git is required to clone the project. Install Git, then run this command again."
  fi
  if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing checkout: $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --ff-only
  elif [ -e "$INSTALL_DIR" ]; then
    fail "$INSTALL_DIR already exists and is not a git checkout. Choose another folder with DMDAGENT_INSTALL_DIR."
  else
    info "Cloning $REPO_URL into $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
  PROJECT_DIR="$(cd "$INSTALL_DIR" && pwd)"
fi

cd "$PROJECT_DIR"

if [ "$USE_DOCKER" = "1" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is required for --docker. Install Docker Desktop or Docker Engine, then run this again."
  fi
  mkdir -p workspace
  info "Starting Docker stack: DMD Agent + Ollama + local model"
  if docker compose version >/dev/null 2>&1; then
    if [ "$DETACHED" = "1" ]; then
      docker compose up --build -d
      info "Dashboard: http://127.0.0.1:8765"
      exit 0
    fi
    docker compose up --build
    exit $?
  fi
  fail "docker compose is required. Install the Docker Compose plugin, then run this again."
fi

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  fail "Python 3.11+ is required. Install Python from https://www.python.org/downloads/ and run this again."
fi

info "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

if [ ! -x ".venv/bin/python" ]; then
  info "Creating local Python environment: .venv"
  "$PYTHON_BIN" -m venv .venv
fi

info "Installing Python package dependencies"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

info "Initializing local app data"
.venv/bin/dmdagent init

if [ -d "frontend" ]; then
  if command -v npm >/dev/null 2>&1; then
    info "Installing dashboard packages"
    (cd frontend && npm install)
  else
    info "npm was not found. Terminal chat will work; dashboard needs Node.js/npm from https://nodejs.org/"
  fi
fi

if [ "${DMDAGENT_SKIP_WIZARD:-0}" != "1" ]; then
  info "Running first-time setup wizard"
  .venv/bin/dmdagent wizard
fi

if [ "${DMDAGENT_INSTALL_SHORTCUT:-1}" != "0" ] && [ -x "./start" ]; then
  info "Installing optional shell shortcut: start"
  ./start install-command || true
fi

info ""
info "Install complete."
info "Project folder:"
info "  $PROJECT_DIR"
info ""
info "Start from this folder:"
info "  ./start session"
info "  ./start web"
info ""
info "If you installed the shell shortcut, open a new terminal and run from this folder:"
info "  start session"
info "  start web"
