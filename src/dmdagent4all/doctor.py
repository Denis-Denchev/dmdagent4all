from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse

from dmdagent4all.app_paths import AppPaths
from dmdagent4all.permissions import RiskLevel
from dmdagent4all.sandbox.terminal import ALWAYS_BLOCKED, BLOCKED_ARG_FRAGMENTS, TerminalPolicy
from dmdagent4all.tools import load_builtin_manifests


@dataclass(frozen=True)
class DoctorCheck:
    status: str
    area: str
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_doctor(
    *,
    paths: AppPaths,
    config: dict[str, Any],
    check_network: bool = True,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    checks.extend(_check_paths(paths))
    checks.extend(_check_secret_config(config))
    checks.extend(_check_llm(config, check_network=check_network))
    checks.extend(_check_terminal(config))
    checks.extend(_check_tools(config))
    checks.extend(_check_telegram(config))
    checks.extend(_check_frontend(Path.cwd()))
    return checks


def doctor_summary(checks: list[DoctorCheck]) -> dict[str, int]:
    return {
        "ok": sum(1 for check in checks if check.status == "ok"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "fail": sum(1 for check in checks if check.status == "fail"),
    }


def checks_to_json(checks: list[DoctorCheck]) -> str:
    return json.dumps(
        {
            "summary": doctor_summary(checks),
            "checks": [check.to_dict() for check in checks],
        },
        indent=2,
        sort_keys=True,
    )


def _check_paths(paths: AppPaths) -> list[DoctorCheck]:
    checks = []
    for label, path in (
        ("data", paths.root),
        ("memory", paths.memory),
        ("workspace", paths.workspace),
        ("logs", paths.logs),
    ):
        if path.exists() and path.is_dir() and os.access(path, os.W_OK):
            checks.append(DoctorCheck("ok", "paths", f"{label} directory is writable: {path}"))
        else:
            checks.append(
                DoctorCheck(
                    "fail",
                    "paths",
                    f"{label} directory is not writable: {path}",
                    "Run `dmdagent init` and check filesystem permissions.",
                )
            )
    return checks


def _check_secret_config(config: dict[str, Any]) -> list[DoctorCheck]:
    checks = []
    llm = config.get("llm", {})
    if any(key in llm for key in {"api_key", "token", "password"}):
        checks.append(
            DoctorCheck(
                "fail",
                "secrets",
                "LLM config contains a direct secret field.",
                "Store only an env var name such as api_key_env; never store API keys in config.yaml.",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "ok",
                "secrets",
                "No direct LLM API key fields found in config.",
            )
        )
    telegram = config.get("interfaces", {}).get("telegram", {})
    if any(key in telegram for key in {"bot_token", "token"}):
        checks.append(
            DoctorCheck(
                "fail",
                "secrets",
                "Telegram config contains a direct token field.",
                "Use bot_token_env and load the token into the process environment.",
            )
        )
    return checks


def _check_llm(config: dict[str, Any], *, check_network: bool) -> list[DoctorCheck]:
    llm = config.get("llm", {})
    provider = str(llm.get("provider", "ollama")).lower()
    model = str(llm.get("planner_model") or llm.get("model") or "").strip()
    base_url = str(llm.get("base_url") or _default_base_url(provider))
    checks = [
        DoctorCheck("ok", "llm", f"Provider configured: {provider}"),
        DoctorCheck("ok", "llm", f"Planner model configured: {model or '-'}"),
    ]

    if provider in {"ollama", "local"}:
        if not check_network:
            checks.append(DoctorCheck("warn", "llm", "Skipped Ollama reachability check."))
            return checks
        checks.append(_check_ollama(base_url, model))
        return checks

    api_key_env = llm.get("api_key_env") or _default_api_key_env(provider)
    if _is_local_base_url(base_url):
        checks.append(
            DoctorCheck(
                "ok",
                "llm",
                f"OpenAI-compatible local endpoint configured: {base_url}",
            )
        )
    elif api_key_env and os.environ.get(str(api_key_env)):
        checks.append(
            DoctorCheck(
                "ok",
                "llm",
                f"API key env var is available: {api_key_env}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "fail",
                "llm",
                f"API key env var is missing for {provider}: {api_key_env or '-'}",
                "Export the key in the environment; do not write the key into config.yaml.",
            )
        )

    privacy = config.get("privacy", {})
    if privacy.get("require_approval_for_cloud_context", True):
        checks.append(
            DoctorCheck(
                "ok",
                "privacy",
                "Cloud context approval policy is enabled.",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "fail",
                "privacy",
                "Cloud context approval policy is disabled.",
                "Set privacy.require_approval_for_cloud_context=true.",
            )
        )
    return checks


def _check_ollama(base_url: str, model: str) -> DoctorCheck:
    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return DoctorCheck(
            "warn",
            "llm",
            f"Ollama is not reachable at {base_url}: {exc}",
            "Start Ollama with `ollama serve` and pull the configured model.",
        )

    names = {
        str(item.get("name", ""))
        for item in data.get("models", [])
        if isinstance(item, dict)
    }
    if model and model in names:
        return DoctorCheck("ok", "llm", f"Ollama model is installed: {model}")
    return DoctorCheck(
        "warn",
        "llm",
        f"Ollama is reachable, but model is not listed: {model or '-'}",
        f"Run `ollama pull {model}`." if model else "Set a local model with `dmdagent models set <model>`.",
    )


def _check_terminal(config: dict[str, Any]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    terminal = config.get("terminal", {})
    policy = TerminalPolicy.from_config(config)
    tool_enabled = bool(config.get("tools", {}).get("terminal.run", {}).get("enabled", False))
    permissions = set(config.get("permissions", {}).get("granted", []))

    if policy.enabled:
        checks.append(DoctorCheck("ok", "terminal", "Terminal policy is enabled."))
        if not tool_enabled:
            checks.append(
                DoctorCheck(
                    "warn",
                    "terminal",
                    "Terminal policy is enabled, but terminal.run tool is disabled.",
                    "Run `dmdagent tools enable terminal.run` when you are ready.",
                )
            )
        if "terminal.run" not in permissions:
            checks.append(
                DoctorCheck(
                    "warn",
                    "terminal",
                    "terminal.run permission is not granted.",
                    "Run `dmdagent permissions grant terminal.run` after reviewing the allowlist.",
                )
            )
    else:
        checks.append(DoctorCheck("ok", "terminal", "Terminal policy is disabled by default."))
        if tool_enabled:
            checks.append(
                DoctorCheck(
                    "warn",
                    "terminal",
                    "terminal.run tool is enabled while terminal policy is disabled.",
                    "Run `dmdagent terminal enable` or `dmdagent tools disable terminal.run`.",
                )
            )

    if terminal.get("workspace_only", True):
        checks.append(DoctorCheck("ok", "terminal", "Terminal cwd is restricted to workspace."))
    else:
        checks.append(
            DoctorCheck(
                "fail",
                "terminal",
                "Terminal workspace_only is disabled.",
                "Set terminal.workspace_only=true.",
            )
        )

    checks.extend(_check_allowed_commands(policy.allowed_commands))
    return checks


def _check_allowed_commands(commands: tuple[tuple[str, ...], ...]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for command in commands:
        if not command:
            checks.append(DoctorCheck("fail", "terminal", "Empty command in terminal allowlist."))
            continue
        binary = Path(command[0]).name
        if binary in ALWAYS_BLOCKED:
            checks.append(
                DoctorCheck(
                    "fail",
                    "terminal",
                    f"Always-blocked command appears in allowlist: {' '.join(command)}",
                    "Remove it with `dmdagent terminal remove ...`.",
                )
            )
            continue
        lowered = [part.casefold() for part in command]
        if any(fragment.casefold() in part for part in lowered for fragment in BLOCKED_ARG_FRAGMENTS):
            checks.append(
                DoctorCheck(
                    "fail",
                    "terminal",
                    f"Allowlist command references a blocked secret/system path: {' '.join(command)}",
                    "Keep .env, SSH keys, and system secrets outside terminal allowlists.",
                )
            )
            continue
        checks.append(
            DoctorCheck("ok", "terminal", f"Allowlisted exact command: {' '.join(command)}")
        )
    return checks


def _check_tools(config: dict[str, Any]) -> list[DoctorCheck]:
    manifests = load_builtin_manifests()
    overrides = config.get("tools", {})
    checks = []
    for name, manifest in manifests.items():
        enabled = bool(overrides.get(name, {}).get("enabled", manifest.default_enabled))
        if enabled and int(manifest.risk) >= int(RiskLevel.MODIFY_EXTERNAL) and not manifest.approval_required:
            checks.append(
                DoctorCheck(
                    "fail",
                    "tools",
                    f"High-risk tool lacks manifest approval requirement: {name}",
                    "Set approval_required=true in the tool manifest.",
                )
            )
    if not checks:
        checks.append(DoctorCheck("ok", "tools", "No enabled high-risk tool bypasses approvals."))
    return checks


def _check_telegram(config: dict[str, Any]) -> list[DoctorCheck]:
    telegram = config.get("interfaces", {}).get("telegram", {})
    if not telegram.get("enabled", False):
        return [DoctorCheck("ok", "telegram", "Telegram interface is disabled by default.")]
    checks = [DoctorCheck("ok", "telegram", "Telegram interface is enabled.")]
    allowed = telegram.get("allowed_user_ids", [])
    if allowed:
        checks.append(DoctorCheck("ok", "telegram", "Telegram allowlist has at least one user."))
    else:
        checks.append(
            DoctorCheck(
                "fail",
                "telegram",
                "Telegram is enabled without allowed user IDs.",
                "Run `dmdagent telegram allow <telegram_user_id>`.",
            )
        )
    token_env = str(telegram.get("bot_token_env", "DMDAGENT_TELEGRAM_BOT_TOKEN"))
    if os.environ.get(token_env):
        checks.append(DoctorCheck("ok", "telegram", f"Telegram token env is available: {token_env}"))
    else:
        checks.append(
            DoctorCheck(
                "fail",
                "telegram",
                f"Telegram token env is missing: {token_env}",
                "Export the token in the environment; do not store it in config.yaml.",
            )
        )
    return checks


def _check_frontend(cwd: Path) -> list[DoctorCheck]:
    frontend = cwd / "frontend"
    package_json = frontend / "package.json"
    src = frontend / "src" / "App.tsx"
    if package_json.exists() and src.exists():
        return [DoctorCheck("ok", "frontend", "Dashboard source is present.")]
    return [
        DoctorCheck(
            "warn",
            "frontend",
            "Dashboard source was not found from the current working directory.",
            "Run doctor from the repository root.",
        )
    ]


def _default_base_url(provider: str) -> str:
    if provider in {"ollama", "local"}:
        return "http://localhost:11434"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider == "lmstudio":
        return "http://localhost:1234/v1"
    if provider == "vllm":
        return "http://localhost:8000/v1"
    return "https://api.openai.com/v1"


def _default_api_key_env(provider: str) -> str | None:
    if provider == "openai":
        return "DMDAGENT_OPENAI_API_KEY"
    if provider == "openrouter":
        return "DMDAGENT_OPENROUTER_API_KEY"
    if provider == "deepseek":
        return "DMDAGENT_DEEPSEEK_API_KEY"
    if provider in {"lmstudio", "vllm"}:
        return None
    return "DMDAGENT_OPENAI_COMPATIBLE_API_KEY"


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}
