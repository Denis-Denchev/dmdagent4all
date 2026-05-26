#!/usr/bin/env bash
set -euo pipefail

MODEL="${DMDCORE_OLLAMA_MODEL:-qwen3:4b}"
BASE_URL="${DMDCORE_OLLAMA_BASE_URL:-http://ollama:11434}"
WORKSPACE="${DMDCORE_WORKSPACE:-/workspace}"

dmdcore init

python - <<'PY'
import os
from pathlib import Path

from dmdcore.app_paths import AppPaths
from dmdcore.config import load_config, save_config

paths = AppPaths.default()
config = load_config(paths.config)

workspace = os.environ.get("DMDCORE_WORKSPACE", "/workspace")
model = os.environ.get("DMDCORE_OLLAMA_MODEL", "qwen3:4b")
base_url = os.environ.get("DMDCORE_OLLAMA_BASE_URL", "http://ollama:11434")

llm = config.setdefault("llm", {})
llm["provider"] = "ollama"
llm["model"] = model
llm["base_url"] = base_url
llm.setdefault("mode", "light")

workspace_config = config.setdefault("workspace", {})
workspace_config["default_path"] = workspace
workspace_config["current_path"] = workspace
workspace_config["allowed_roots"] = [workspace]

terminal = config.setdefault("terminal", {})
terminal["workspace_root"] = workspace

Path(workspace).mkdir(parents=True, exist_ok=True)
save_config(config, paths.config)
PY

if [ "${DMDCORE_PULL_MODEL:-1}" = "1" ]; then
  python - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

base_url = os.environ.get("DMDCORE_OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
model = os.environ.get("DMDCORE_OLLAMA_MODEL", "qwen3:4b")

deadline = time.time() + int(os.environ.get("DMDCORE_OLLAMA_WAIT_SECONDS", "300"))
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5).read()
        break
    except (OSError, urllib.error.URLError):
        time.sleep(2)
else:
    print(f"Ollama was not reachable at {base_url}; skipping model pull.", flush=True)
    raise SystemExit(0)

print(f"Ensuring Ollama model is available: {model}", flush=True)
payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
request = urllib.request.Request(
    f"{base_url}/api/pull",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(request, timeout=int(os.environ.get("DMDCORE_OLLAMA_PULL_TIMEOUT", "1800"))).read()
except (OSError, urllib.error.URLError) as exc:
    print(f"Model pull failed: {exc}", flush=True)
PY
fi

case "${1:-serve}" in
  serve)
    exec uvicorn dmdcore.server:app --host 0.0.0.0 --port "${DMDCORE_PORT:-8765}"
    ;;
  *)
    exec "$@"
    ;;
esac
