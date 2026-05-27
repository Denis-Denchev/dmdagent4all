#!/usr/bin/env bash
# DMD Sentinel one-line installer for macOS and Linux.
# (Formerly DMD Core — Python package and env var names kept as `dmdcore` / `DMDCORE_*` for backwards compatibility.)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Denis-Denchev/dmd-sentinel/main/install.sh | bash
#
# Optional environment variables:
#   DMDCORE_INSTALL_DIR   Target install folder (default: $HOME/dmd-sentinel)
#   DMDCORE_REPO_URL      Source git URL (default: official repo)
#   DMDCORE_SKIP_OLLAMA   Set to 1 to skip Ollama installation
#   DMDCORE_SKIP_LAUNCH   Set to 1 to skip auto-launching the dashboard
#   DMDCORE_SKIP_WIZARD   Set to 1 to skip the interactive first-time wizard
#   DMDCORE_NO_SUDO       Set to 1 to never attempt sudo (will fail on linux if root needed)
set -euo pipefail

REPO_URL="${DMDCORE_REPO_URL:-https://github.com/Denis-Denchev/dmd-sentinel.git}"
INSTALL_DIR="${DMDCORE_INSTALL_DIR:-$HOME/dmd-sentinel}"
SKIP_OLLAMA="${DMDCORE_SKIP_OLLAMA:-0}"
SKIP_LAUNCH="${DMDCORE_SKIP_LAUNCH:-0}"
NO_SUDO="${DMDCORE_NO_SUDO:-0}"

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[1;33m   warn:\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

OS_KIND=""
case "$(uname -s)" in
  Darwin*) OS_KIND=mac ;;
  Linux*)  OS_KIND=linux ;;
  *) fail "Unsupported OS: $(uname -s). Use install.ps1 on Windows." ;;
esac

run_priv() {
  if [ "$(id -u)" = "0" ]; then
    "$@"
    return $?
  fi
  if [ "$NO_SUDO" = "1" ]; then
    fail "Need root for: $*. Re-run as root or unset DMDCORE_NO_SUDO."
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo is required to install system packages. Install sudo or set DMDCORE_NO_SUDO=1 with a different plan."
  fi
  sudo "$@"
}

detect_linux_pkg() {
  if command -v apt-get >/dev/null 2>&1; then echo apt; return; fi
  if command -v dnf     >/dev/null 2>&1; then echo dnf; return; fi
  if command -v yum     >/dev/null 2>&1; then echo yum; return; fi
  if command -v pacman  >/dev/null 2>&1; then echo pacman; return; fi
  if command -v zypper  >/dev/null 2>&1; then echo zypper; return; fi
  if command -v apk     >/dev/null 2>&1; then echo apk; return; fi
  echo unknown
}

pkg_refresh_done=0
pkg_refresh() {
  [ "$pkg_refresh_done" = "1" ] && return 0
  case "$LINUX_PKG" in
    apt)    run_priv apt-get update -y ;;
    dnf)    run_priv dnf -y check-update || true ;;
    yum)    run_priv yum -y check-update || true ;;
    pacman) run_priv pacman -Sy --noconfirm ;;
    zypper) run_priv zypper refresh ;;
    apk)    run_priv apk update ;;
  esac
  pkg_refresh_done=1
}

pkg_install() {
  if [ "$OS_KIND" = "mac" ]; then
    brew install "$@"
    return $?
  fi
  pkg_refresh
  case "$LINUX_PKG" in
    apt)    run_priv apt-get install -y "$@" ;;
    dnf)    run_priv dnf install -y "$@" ;;
    yum)    run_priv yum install -y "$@" ;;
    pacman) run_priv pacman -S --noconfirm --needed "$@" ;;
    zypper) run_priv zypper --non-interactive install "$@" ;;
    apk)    run_priv apk add --no-cache "$@" ;;
    *)      fail "No supported package manager. Install $* manually and re-run." ;;
  esac
}

ensure_homebrew_mac() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi
  step "Installing Homebrew (package manager for macOS)"
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

python_ok() {
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null
}

ensure_python() {
  for cand in python3.13 python3.12 python3.11 python3; do
    if python_ok "$cand"; then
      info "Python found: $cand ($("$cand" -c 'import sys; print(sys.version.split()[0])'))"
      return 0
    fi
  done
  step "Installing Python 3.13"
  if [ "$OS_KIND" = "mac" ]; then
    pkg_install python@3.13
    if [ -x /opt/homebrew/opt/python@3.13/bin/python3.13 ]; then
      ln -sf /opt/homebrew/opt/python@3.13/bin/python3.13 /opt/homebrew/bin/python3.13 2>/dev/null || true
    fi
  else
    case "$LINUX_PKG" in
      apt)    pkg_install python3 python3-venv python3-pip ;;
      dnf|yum) pkg_install python3 python3-pip ;;
      pacman) pkg_install python python-pip ;;
      zypper) pkg_install python3 python3-pip python3-venv ;;
      apk)    pkg_install python3 py3-pip ;;
      *)      fail "Cannot auto-install Python on this system. Install Python 3.11+ manually." ;;
    esac
  fi
  for cand in python3.13 python3.12 python3.11 python3; do
    if python_ok "$cand"; then return 0; fi
  done
  fail "Python 3.11+ install completed but the binary was not found on PATH."
}

ensure_git() {
  if command -v git >/dev/null 2>&1; then return 0; fi
  step "Installing Git"
  pkg_install git
}

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    info "Node.js found: $(node --version)"
    return 0
  fi
  step "Installing Node.js (for the dashboard UI)"
  if [ "$OS_KIND" = "mac" ]; then
    pkg_install node
  else
    case "$LINUX_PKG" in
      apt)
        curl -fsSL https://deb.nodesource.com/setup_20.x | run_priv -E bash -
        pkg_install nodejs
        ;;
      dnf|yum)
        run_priv bash -c "curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -"
        pkg_install nodejs
        ;;
      pacman) pkg_install nodejs npm ;;
      zypper) pkg_install nodejs20 npm20 ;;
      apk)    pkg_install nodejs npm ;;
      *)      warn "Could not auto-install Node. Dashboard UI will be skipped. Terminal chat will still work." ;;
    esac
  fi
}

ensure_ollama() {
  [ "$SKIP_OLLAMA" = "1" ] && { info "Skipping Ollama (DMDCORE_SKIP_OLLAMA=1)"; return 0; }
  if command -v ollama >/dev/null 2>&1; then
    info "Ollama found: $(ollama --version 2>/dev/null | head -1)"
    return 0
  fi
  step "Installing Ollama (local LLM runtime)"
  if [ "$OS_KIND" = "mac" ]; then
    if brew list ollama >/dev/null 2>&1; then return 0; fi
    if ! brew install ollama 2>/dev/null; then
      info "brew install failed; falling back to the official Ollama installer"
      curl -fsSL https://ollama.com/install.sh | sh
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
}

clone_or_update_repo() {
  if [ -f "$INSTALL_DIR/pyproject.toml" ] && [ -d "$INSTALL_DIR/src/dmdcore" ]; then
    step "Updating existing checkout at $INSTALL_DIR"
    if [ -d "$INSTALL_DIR/.git" ]; then
      git -C "$INSTALL_DIR" pull --ff-only || warn "git pull failed; continuing with current files"
    fi
  elif [ -e "$INSTALL_DIR" ]; then
    fail "$INSTALL_DIR exists but is not a DMD Core checkout. Move it or set DMDCORE_INSTALL_DIR."
  else
    step "Cloning DMD Core into $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
}

open_browser_async() {
  local url="$1"
  if [ "$OS_KIND" = "mac" ]; then
    ( sleep 3 && open "$url" >/dev/null 2>&1 ) &
  else
    if command -v xdg-open >/dev/null 2>&1; then
      ( sleep 3 && xdg-open "$url" >/dev/null 2>&1 ) &
    fi
  fi
}

step "DMD Core installer ($OS_KIND)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -f "$SCRIPT_DIR/pyproject.toml" ] && [ -d "$SCRIPT_DIR/src/dmdcore" ]; then
  info "Detected existing DMD Core checkout at $SCRIPT_DIR; using it instead of cloning."
  INSTALL_DIR="$SCRIPT_DIR"
fi

if [ "$OS_KIND" = "mac" ]; then
  ensure_homebrew_mac
else
  LINUX_PKG="$(detect_linux_pkg)"
  if [ "$LINUX_PKG" = "unknown" ]; then
    fail "Unknown Linux package manager. Install Python 3.11+, Node 20+, git, curl manually then re-run."
  fi
  info "Linux package manager: $LINUX_PKG"
  pkg_install ca-certificates curl
fi

ensure_git
ensure_python
ensure_node
ensure_ollama
clone_or_update_repo

cd "$INSTALL_DIR"

step "Running project bootstrap"
DMDCORE_INSTALL_SHORTCUT=0 bash scripts/bootstrap.sh

if [ "$SKIP_LAUNCH" != "1" ]; then
  step "Launching DMD Core"
  open_browser_async "http://127.0.0.1:5174/"
  info "If the browser does not open automatically, visit http://127.0.0.1:5174"
  info "Press Ctrl+C to stop."
  exec ./start web
fi

cat <<EOF

DMD Core install finished.

Project folder: $INSTALL_DIR

Next steps:
  cd "$INSTALL_DIR"
  ./start web         # dashboard + API
  ./start session     # terminal chat

EOF
