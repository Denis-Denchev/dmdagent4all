from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import URLError

import yaml

from dmdcore import __version__
from dmdcore.app_paths import AppPaths
from dmdcore.audit import AuditStore
from dmdcore.config import load_config, save_config, write_default_config
from dmdcore.doctor import checks_to_json, doctor_summary, run_doctor
from dmdcore.interfaces.telegram import (
    DEFAULT_TOKEN_ENV,
    TelegramConfigError,
    TelegramError,
    TelegramInterface,
    load_telegram_token,
    settings_from_config,
    store_telegram_token,
    telegram_token_available,
)
from dmdcore.memory import MemoryManager
from dmdcore.model_presets import (
    MAC_MINI_RECOMMENDATIONS,
    MODEL_MODES,
    detect_memory_gb,
    recommend_for_memory,
)
from dmdcore.permissions import ToolRequest
from dmdcore.runtime import build_agent_core, permission_context_from_config
from dmdcore.sandbox import TerminalPolicy
from dmdcore.tools import load_builtin_manifests


@dataclass
class BackgroundService:
    name: str
    thread: threading.Thread
    stop_event: threading.Event


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        return command_chat("", start_ollama=True)

    parser = argparse.ArgumentParser(
        prog="dmdcore",
        description="DMD Agent 4 All local control center.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("status", help="Show local agent status.")
    subcommands.add_parser("wizard", help="Run the local install wizard.")
    subcommands.add_parser("init", help="Create local data directories and config.")
    start_parser = subcommands.add_parser("start", help="Start API and dashboard in one terminal.")
    start_parser.add_argument("--host", default="127.0.0.1")
    start_parser.add_argument("--api-port", default=8765, type=int)
    start_parser.add_argument("--ui-port", default=5174, type=int)
    start_parser.add_argument("--no-open", action="store_true", help="Do not open the browser.")
    start_parser.add_argument(
        "--pull-model",
        action="store_true",
        help="Pull the configured Ollama model before starting.",
    )
    start_parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Do not try to start Ollama automatically.",
    )
    start_parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Do not start Telegram polling with the control center.",
    )
    start_parser.add_argument("--verbose", action="store_true", help="Show child process logs.")
    open_parser = subcommands.add_parser("open", help="Open the local dashboard.")
    open_parser.add_argument("--ui-port", default=5174, type=int)
    doctor_parser = subcommands.add_parser("doctor", help="Run local readiness and security checks.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor_parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip local model reachability checks.",
    )
    chat_parser = subcommands.add_parser("chat", help="Send a message to the local agent.")
    chat_parser.add_argument("--no-ollama", action="store_true", help="Do not auto-start Ollama.")
    chat_parser.add_argument("message", nargs="*", help="Message text. Omit for interactive mode.")
    ask_parser = subcommands.add_parser("ask", help="Ask one question in the terminal.")
    ask_parser.add_argument("--no-ollama", action="store_true", help="Do not auto-start Ollama.")
    ask_parser.add_argument("message", nargs="+")

    model_parser = subcommands.add_parser("model", help="Simple model setup.")
    model_parser.add_argument("args", nargs="*", help="Examples: fast, light --pull, use qwen3:8b")
    model_parser.add_argument("--pull", action="store_true", help="Pull Ollama model after selecting it.")

    models_parser = subcommands.add_parser("models", help="Manage model settings.")
    models_subcommands = models_parser.add_subparsers(dest="models_command", required=True)
    models_subcommands.add_parser("list", help="List recommended model modes.")
    models_set_mode = models_subcommands.add_parser("set-mode", help="Set a recommended model mode.")
    models_set_mode.add_argument("mode", choices=[mode.key for mode in MODEL_MODES])
    models_set_mode.add_argument("--pull", action="store_true", help="Pull the selected Ollama model.")
    models_set = models_subcommands.add_parser("set", help="Set an explicit Ollama model name.")
    models_set.add_argument("model")
    models_set.add_argument("--pull", action="store_true", help="Pull the selected Ollama model.")
    models_provider = models_subcommands.add_parser(
        "provider",
        help="Select ollama/local or an OpenAI-compatible API provider.",
    )
    models_provider.add_argument(
        "provider",
        choices=["ollama", "local", "openai", "deepseek", "openai-compatible", "openrouter", "lmstudio", "vllm"],
    )
    models_provider.add_argument("--base-url")
    models_provider.add_argument("--api-key-env")
    models_provider.add_argument("--model", help="Model name to use with the selected provider.")

    serve_parser = subcommands.add_parser("serve", help="Start the local API server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8765, type=int)
    serve_parser.add_argument("--reload", action="store_true")

    tools_parser = subcommands.add_parser("tools", help="Manage tools.")
    tools_subcommands = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_subcommands.add_parser("list", help="List tools.")
    tools_enable = tools_subcommands.add_parser("enable", help="Enable a tool.")
    tools_enable.add_argument("tool")
    tools_disable = tools_subcommands.add_parser("disable", help="Disable a tool.")
    tools_disable.add_argument("tool")

    permissions_parser = subcommands.add_parser("permissions", help="Manage permissions.")
    permissions_subcommands = permissions_parser.add_subparsers(
        dest="permissions_command",
        required=True,
    )
    permissions_subcommands.add_parser("list", help="List granted permissions.")
    permissions_grant = permissions_subcommands.add_parser("grant", help="Grant a permission.")
    permissions_grant.add_argument("permission")
    permissions_revoke = permissions_subcommands.add_parser("revoke", help="Revoke a permission.")
    permissions_revoke.add_argument("permission")

    terminal_parser = subcommands.add_parser("terminal", help="Manage safe terminal execution.")
    terminal_subcommands = terminal_parser.add_subparsers(dest="terminal_command", required=True)
    terminal_subcommands.add_parser("status", help="Show terminal policy status.")
    terminal_enable = terminal_subcommands.add_parser("enable", help="Enable terminal policy.")
    terminal_enable.add_argument(
        "--tool",
        action="store_true",
        help="Also enable the terminal.run tool manifest.",
    )
    terminal_enable.add_argument(
        "--grant-permission",
        action="store_true",
        help="Also grant the terminal.run permission.",
    )
    terminal_subcommands.add_parser("disable", help="Disable terminal policy and tool.")
    terminal_auto_approve = terminal_subcommands.add_parser(
        "auto-approve",
        help="Toggle automatic approval for exact allowlisted commands.",
    )
    terminal_auto_approve.add_argument("state", choices=("on", "off"))
    terminal_workspace = terminal_subcommands.add_parser(
        "workspace",
        help="Set the root directory used by terminal.run.",
    )
    terminal_workspace.add_argument("path", help="Absolute path, or '-' to reset to the private workspace.")
    terminal_allow = terminal_subcommands.add_parser("allow", help="Add an exact command allowlist entry.")
    terminal_allow.add_argument("command", nargs=argparse.REMAINDER)
    terminal_remove = terminal_subcommands.add_parser("remove", help="Remove an exact command allowlist entry.")
    terminal_remove.add_argument("command", nargs=argparse.REMAINDER)
    terminal_run = terminal_subcommands.add_parser(
        "run",
        help="Request an approved terminal.run tool call.",
    )
    terminal_run.add_argument("--cwd", help="Workspace-relative cwd for the command.")
    terminal_run.add_argument("command", nargs=argparse.REMAINDER)

    memory_parser = subcommands.add_parser("memory", help="Manage local Markdown memory.")
    memory_subcommands = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subcommands.add_parser("path", help="Show memory directory.")
    memory_subcommands.add_parser("list", help="List memory files.")
    memory_read = memory_subcommands.add_parser("read", help="Read a memory file.")
    memory_read.add_argument("path")
    memory_subcommands.add_parser("open", help="Open the memory directory.")

    approvals_parser = subcommands.add_parser("approvals", help="Inspect pending approvals.")
    approvals_subcommands = approvals_parser.add_subparsers(
        dest="approvals_command",
        required=True,
    )
    approvals_subcommands.add_parser("list", help="List recent approval events.")
    approvals_approve = approvals_subcommands.add_parser("approve", help="Approve a pending item.")
    approvals_approve.add_argument("approval_id", type=int)
    approvals_deny = approvals_subcommands.add_parser("deny", help="Deny a pending item.")
    approvals_deny.add_argument("approval_id", type=int)

    telegram_parser = subcommands.add_parser("telegram", help="Manage Telegram remote access.")
    telegram_subcommands = telegram_parser.add_subparsers(
        dest="telegram_command",
        required=True,
    )
    telegram_subcommands.add_parser("status", help="Show Telegram interface status.")
    telegram_subcommands.add_parser("enable", help="Enable Telegram interface locally.")
    telegram_subcommands.add_parser("disable", help="Disable Telegram interface locally.")
    telegram_allow = telegram_subcommands.add_parser("allow", help="Allow a Telegram user ID.")
    telegram_allow.add_argument("user_id", type=int)
    telegram_remove = telegram_subcommands.add_parser(
        "remove",
        help="Remove a Telegram user ID from the allowlist.",
    )
    telegram_remove.add_argument("user_id", type=int)
    telegram_run = telegram_subcommands.add_parser("run", help="Run Telegram Bot API polling.")
    telegram_run.add_argument("--once", action="store_true", help="Poll once and exit.")
    telegram_run.add_argument("--timeout", type=int, help="Long-poll timeout in seconds.")

    args = parser.parse_args(raw_argv)

    if args.command == "init":
        return command_init()
    if args.command == "status":
        return command_status()
    if args.command == "start":
        return command_start(
            host=args.host,
            api_port=args.api_port,
            ui_port=args.ui_port,
            open_browser=not args.no_open,
            pull_model=args.pull_model,
            start_ollama=not args.no_ollama,
            start_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    if args.command == "open":
        _open_url(f"http://127.0.0.1:{args.ui_port}")
        return 0
    if args.command == "doctor":
        return command_doctor(json_output=args.json, check_network=not args.skip_network)
    if args.command == "wizard":
        return command_wizard()
    if args.command == "serve":
        return command_serve(args.host, args.port, args.reload)
    if args.command == "chat":
        return command_chat(" ".join(args.message), start_ollama=not args.no_ollama)
    if args.command == "ask":
        return command_chat(" ".join(args.message), start_ollama=not args.no_ollama)
    if args.command == "model":
        return command_model(args)
    if args.command == "models":
        return command_models(args)
    if args.command == "tools":
        return command_tools(args)
    if args.command == "permissions":
        return command_permissions(args)
    if args.command == "terminal":
        return command_terminal(args)
    if args.command == "memory":
        return command_memory(args)
    if args.command == "approvals":
        return command_approvals(args)
    if args.command == "telegram":
        return command_telegram(args)

    parser.print_help()
    return 2


def command_init() -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    MemoryManager(paths.memory).bootstrap()
    AuditStore(paths.audit_db)
    print("DMD Agent 4 All local data initialized.")
    print(f"Config: {config_path}")
    print(f"Memory: {paths.memory}")
    print(f"Workspace: {paths.workspace}")
    print(f"Audit DB: {paths.audit_db}")
    return 0


def command_status() -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    llm = config["llm"]
    api_key_env = llm.get("api_key_env")
    _print_title("DMD Agent 4 All status")
    _print_rows(
        ["Setting", "Value"],
        [
            ["Version", __version__],
            ["Data directory", str(paths.root)],
            ["Config", str(config_path)],
            ["Memory", str(paths.memory)],
            ["Workspace", str(paths.workspace)],
            ["Audit DB", str(paths.audit_db)],
            ["LLM provider", str(llm.get("provider", "-"))],
            ["LLM model", str(llm.get("model", "-"))],
            ["LLM base URL", str(llm.get("base_url", "-"))],
            [
                "API key env",
                f"{api_key_env} ({'set' if api_key_env and os.environ.get(str(api_key_env)) else 'not set'})"
                if api_key_env
                else "-",
            ],
            ["UI language", str(config["interfaces"]["ui_language"])],
            ["Assistant language", str(llm.get("response_language", "auto"))],
            ["Terminal enabled", _yes_no(bool(config["terminal"]["enabled"]))],
            ["Browser enabled", _yes_no(bool(config["browser"]["enabled"]))],
        ],
    )
    return 0


def command_doctor(*, json_output: bool, check_network: bool) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    checks = run_doctor(paths=paths, config=config, check_network=check_network)
    if json_output:
        print(checks_to_json(checks))
    else:
        _print_doctor_report(checks)
    return 1 if doctor_summary(checks)["fail"] else 0


def command_start(
    *,
    host: str,
    api_port: int,
    ui_port: int,
    open_browser: bool,
    pull_model: bool,
    start_ollama: bool,
    start_telegram: bool,
    verbose: bool,
) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    MemoryManager(paths.memory).bootstrap()
    AuditStore(paths.audit_db)
    config = load_config(config_path)

    _print_title("DMD Agent 4 All")
    print("Starting local control center.")
    print(f"Model: {config.get('llm', {}).get('model', '-')}")
    if str(config.get("llm", {}).get("model", "")).strip().lower() == "dumb":
        print("Model looks invalid. Recommended fix: dmdcore model fast --pull")
    print("")

    spawned: list[tuple[str, subprocess.Popen[Any]]] = []
    api_url = f"http://{host}:{api_port}"
    ui_url = f"http://{host}:{ui_port}"

    try:
        if start_ollama and str(config.get("llm", {}).get("provider", "ollama")).lower() in {"ollama", "local"}:
            ollama_process = _ensure_ollama(config, verbose=verbose)
            if ollama_process is not None:
                spawned.append(("Ollama", ollama_process))

        if pull_model:
            _pull_ollama_model(str(config.get("llm", {}).get("model", "")))

        api_process = _ensure_api(
            host=host,
            port=api_port,
            verbose=verbose,
            start_telegram=start_telegram,
        )
        if api_process is not None:
            spawned.append(("API", api_process))

        ui_process = _ensure_dashboard(
            host=host,
            ui_port=ui_port,
            api_url=api_url,
            verbose=verbose,
        )
        if ui_process is not None:
            spawned.append(("Dashboard", ui_process))

        print("")
        print(_style("Ready", "1;32"))
        print(f"Dashboard: {ui_url}")
        print(f"API:       {api_url}")
        if start_telegram:
            telegram_status = _http_json(f"{api_url}/v1/telegram")
            telegram_running = bool(telegram_status.get("polling")) if telegram_status else False
            print(f"Telegram: {'running' if telegram_running else 'not running'}")
        print("")
        print("Controls:")
        print("- Stop this session: Ctrl+C")
        print("- Change model:      dmdcore model fast --pull")
        print("- Check health:      dmdcore doctor")
        print("")
        if open_browser:
            _open_url(ui_url)

        if not spawned:
            print("Services are already running.")
            return 0

        while True:
            for name, process in spawned:
                returncode = process.poll()
                if returncode is not None:
                    print(f"{name} stopped with exit code {returncode}.")
                    _stop_processes(spawned)
                    return int(returncode)
            time.sleep(0.5)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        _stop_processes(spawned)
        return 1
    except KeyboardInterrupt:
        print("")
        print("Stopping DMD Agent...")
        _stop_processes(spawned)
        return 0


def command_wizard() -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    MemoryManager(paths.memory).bootstrap()

    ram_gb = detect_memory_gb()
    recommendation = recommend_for_memory(ram_gb)

    print("DMD Agent 4 All install wizard")
    print("")
    print("Language policy:")
    print("- Install wizard, terminal output, docs, and UI labels are English by default.")
    print("- The assistant response language is Auto by default and depends on the selected model.")
    print("")
    print("Local paths:")
    print(f"- Config: {config_path}")
    print(f"- Memory: {paths.memory}")
    print(f"- Workspace: {paths.workspace}")
    print("")
    print("Detected hardware:")
    print(f"- RAM: {ram_gb if ram_gb is not None else 'unknown'} GB")
    print("")
    print("Recommended default:")
    print(f"- Machine tier: {recommendation.machine_label}")
    print(f"- Default model: {recommendation.default_model}")
    print(f"- Better mode: {recommendation.better_mode}")
    print(f"- Note: {recommendation.note}")
    print("")
    print("Model modes:")
    for mode in MODEL_MODES:
        alternatives = f" or {', '.join(mode.alternatives)}" if mode.alternatives else ""
        print(f"- {mode.label}: {mode.default_model}{alternatives}")
        print(f"  {mode.description}")
    print("")
    print("Mac mini recommendations:")
    _print_rows(
        ["Mac mini", "Default model", "Better mode", "Note"],
        [
            [
                item.machine_label,
                item.default_model,
                item.better_mode,
                item.note,
            ]
            for item in MAC_MINI_RECOMMENDATIONS
        ],
    )
    print("")
    first_model = recommendation.default_model.split(" / ")[0]
    print("Next steps:")
    print("- Install Ollama if it is not installed: https://ollama.com/download")
    print(f"- Pull a local model: ollama pull {first_model}")
    print("- Start the local API: dmdcore serve")
    return 0


def command_serve(host: str, port: int, reload: bool) -> int:
    import uvicorn

    uvicorn.run(
        "dmdcore.server:app",
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def command_chat(message: str, *, start_ollama: bool = True) -> int:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    MemoryManager(paths.memory).bootstrap()
    AuditStore(paths.audit_db)
    config = load_config(paths.config)
    if not message.strip() and _needs_terminal_onboarding(config):
        config = _run_terminal_onboarding(paths)

    if message.strip() and _is_telegram_setup_request(message):
        _print_telegram_setup_guide()
        return 0

    spawned: list[tuple[str, subprocess.Popen[Any]]] = []
    background_services: list[BackgroundService] = []

    if start_ollama and str(config.get("llm", {}).get("provider", "ollama")).lower() in {"ollama", "local"}:
        ollama_process = _ensure_ollama(config, verbose=False)
        if ollama_process is not None:
            spawned.append(("Ollama", ollama_process))

    local_provider = str(config.get("llm", {}).get("provider", "ollama")).lower() in {"ollama", "local"}
    core = build_agent_core(planner_enabled=not (message.strip() and not start_ollama and local_provider))
    if not message.strip():
        _start_telegram_background_if_ready(config, background_services, explicit=False)

    try:
        if message.strip():
            response = core.handle_text(message)
            _print_chat_response(response.status, response.message, response.data)
            return 0 if response.status in {"ok", "approval_required", "denied"} else 1

        _print_chat_header(config)
        while True:
            try:
                line = input(_style("you> ", "1;36"))
            except EOFError:
                print("")
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            command_result = _handle_chat_command(stripped, core, background_services)
            if command_result == "exit":
                return 0
            if command_result == "handled":
                continue
            if _is_telegram_setup_request(stripped):
                _run_telegram_setup_wizard(background_services)
                continue
            response = core.handle_text(stripped)
            _print_chat_response(response.status, response.message, response.data)
    finally:
        _stop_background_services(background_services)
        _stop_processes(spawned)


def command_model(args: argparse.Namespace) -> int:
    raw_args = list(args.args)
    pull = bool(args.pull)
    if "--pull" in raw_args:
        pull = True
        raw_args = [item for item in raw_args if item != "--pull"]

    if not raw_args or raw_args[0] in {"list", "status"}:
        return command_models(
            argparse.Namespace(models_command="list")
        )

    selection = raw_args[0].lower()
    if selection in {mode.key for mode in MODEL_MODES}:
        return _set_model_mode(selection, pull=pull)

    if selection in {"use", "set"}:
        if len(raw_args) < 2:
            print("Model name is required. Example: dmdcore model use qwen3:8b", file=sys.stderr)
            return 1
        return _set_model_name(raw_args[1], pull=pull)

    print("Unknown model command.", file=sys.stderr)
    print("Examples:")
    print("  dmdcore model")
    print("  dmdcore model fast --pull")
    print("  dmdcore model light --pull")
    print("  dmdcore model use qwen3:8b --pull")
    return 1


def _needs_terminal_onboarding(config: dict[str, Any]) -> bool:
    return not bool(config.get("setup", {}).get("completed", False))


def _run_terminal_onboarding(paths: AppPaths) -> dict[str, Any]:
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    _print_title("Welcome to DMD Agent")
    print("First-time setup. Press Enter to use the recommended answer.")
    print("")

    setup = config.setdefault("setup", {})
    agent_name = _ask_text("Agent name", str(setup.get("agent_name") or "DMD Agent"))
    user_name = _ask_text("Your name", str(setup.get("user_name") or ""))
    language = _ask_choice(
        "Response language",
        [
            ("auto", "Auto-detect from your message"),
            ("en", "English"),
            ("bg", "Bulgarian"),
        ],
        default="auto",
    )
    model_choice = _ask_choice(
        "How should the agent think?",
        [
            ("local-fast", "Local Ollama, recommended qwen3:8b"),
            ("local-light", "Local Ollama, smaller qwen3:4b"),
            ("api-openai", "OpenAI/DeepSeek-compatible API key"),
            ("skip", "Skip model setup for now"),
        ],
        default="local-fast",
    )

    llm = config.setdefault("llm", {})
    if model_choice == "local-fast":
        llm["provider"] = "ollama"
        llm["model"] = "qwen3:8b"
        llm["planner_model"] = "qwen3:8b"
        llm["mode"] = "fast"
        llm["base_url"] = "http://localhost:11434"
        llm["api_key_env"] = None
    elif model_choice == "local-light":
        llm["provider"] = "ollama"
        llm["model"] = "qwen3:4b"
        llm["planner_model"] = "qwen3:4b"
        llm["mode"] = "light"
        llm["base_url"] = "http://localhost:11434"
        llm["api_key_env"] = None
    elif model_choice == "api-openai":
        provider = _ask_choice(
            "API provider",
            [
                ("openai", "OpenAI"),
                ("deepseek", "DeepSeek"),
                ("openrouter", "OpenRouter"),
                ("openai-compatible", "Other OpenAI-compatible endpoint"),
            ],
            default="openai",
        )
        default_model = "deepseek-v4-flash" if provider == "deepseek" else "gpt-4o-mini"
        model = _ask_text("Model name", default_model)
        base_url = _default_provider_base_url(provider)
        if provider == "openai-compatible":
            base_url = _ask_text("Base URL", "http://localhost:1234/v1")
        api_key_env = _ask_text(
            "API key environment variable name",
            _default_provider_api_key_env(provider) or "DMDCORE_OPENAI_COMPATIBLE_API_KEY",
        )
        llm["provider"] = provider
        llm["model"] = model
        llm["planner_model"] = model
        llm["mode"] = "custom"
        llm["base_url"] = base_url
        llm["api_key_env"] = api_key_env

    llm["response_language"] = language
    setup["completed"] = True
    setup["agent_name"] = agent_name
    setup["user_name"] = user_name
    setup["preferred_language"] = language
    save_config(config, config_path)
    _write_onboarding_memory(paths, agent_name=agent_name, user_name=user_name, language=language)

    print("")
    print(_style("Setup saved.", "1;32"))
    if model_choice in {"local-fast", "local-light"}:
        print(f"To pull the model now: /model {llm['mode']} --pull")
        print("You can change later with: /model light --pull or /model fast --pull")
    elif model_choice == "api-openai":
        print(f"Before using the API provider, export your key: export {llm['api_key_env']}=<api-key>")
        print("The key value was not saved in config.")
    print("")
    return config


def _ask_text(label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _ask_choice(
    label: str,
    choices: list[tuple[str, str]],
    *,
    default: str,
) -> str:
    print(label + ":")
    keys = [key for key, _description in choices]
    for index, (key, description) in enumerate(choices, start=1):
        marker = " default" if key == default else ""
        print(f"  {index}. {description} ({key}){marker}")
    while True:
        raw = input(f"Choose [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in keys:
            return raw
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(choices):
                return choices[index][0]
        print("Please choose one of the listed options.")


def _write_onboarding_memory(
    paths: AppPaths,
    *,
    agent_name: str,
    user_name: str,
    language: str,
) -> None:
    manager = MemoryManager(paths.memory)
    manager.write(
        "long-term/profile.md",
        "\n".join(
            [
                f"User name: {user_name or 'not set'}",
                f"Assistant name: {agent_name}",
                f"Preferred response language: {language}",
            ]
        ),
        metadata={
            "type": "profile",
            "memory_scope": "long-term",
            "source": "terminal_onboarding",
            "confidence": "high",
        },
    )
    manager.write(
        "long-term/preferences.md",
        "\n".join(
            [
                "The assistant should keep terminal interaction simple.",
                "The assistant should explain setup steps clearly before asking the user to act.",
                "The assistant should answer in the user's language when possible.",
            ]
        ),
        metadata={
            "type": "preferences",
            "memory_scope": "long-term",
            "source": "terminal_onboarding",
            "confidence": "high",
        },
    )


def command_models(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)

    if args.models_command == "list":
        rows = [
            [
                mode.key,
                mode.label,
                mode.default_model,
                ", ".join(mode.alternatives) or "-",
                mode.description,
            ]
            for mode in MODEL_MODES
        ]
        _print_rows(["Mode", "Label", "Default", "Alternatives", "Use case"], rows)
        print("")
        print(f"Current mode: {config['llm'].get('mode', '-')}")
        print(f"Current provider: {config['llm'].get('provider', '-')}")
        print(f"Current model: {config['llm'].get('model', '-')}")
        print(f"Current planner model: {config['llm'].get('planner_model') or config['llm'].get('model', '-')}")
        print(f"Current base URL: {config['llm'].get('base_url', '-')}")
        print(f"Current API key env: {config['llm'].get('api_key_env') or '-'}")
        return 0

    if args.models_command == "set-mode":
        return _set_model_mode(args.mode, pull=bool(args.pull))

    if args.models_command == "set":
        return _set_model_name(args.model, pull=bool(args.pull))

    if args.models_command == "provider":
        provider = str(args.provider)
        llm = _set_provider_config(
            config,
            provider=provider,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            model=args.model,
        )
        save_config(config, config_path)
        print(f"Provider set to {provider}.")
        print(f"Base URL: {llm['base_url']}")
        if llm.get("api_key_env"):
            print(f"API key env: {llm['api_key_env']}")
            print("The API key value is not stored in config.")
        else:
            print("API key env: -")
        _print_model_hint(llm)
        return 0

    return 2


def _set_model_mode(mode_key: str, *, pull: bool = False) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    selected = next(mode for mode in MODEL_MODES if mode.key == mode_key)
    config.setdefault("llm", {})["mode"] = selected.key
    config["llm"]["provider"] = "ollama"
    config["llm"]["model"] = selected.default_model
    config["llm"]["planner_model"] = selected.default_model
    config["llm"]["base_url"] = "http://localhost:11434"
    config["llm"]["api_key_env"] = None
    save_config(config, config_path)
    print(f"Model mode set to {selected.label}.")
    print(f"Model: {selected.default_model}")
    print(f"Planner model: {selected.default_model}")
    if pull:
        return _pull_ollama_model(selected.default_model)
    _print_model_hint(config["llm"])
    return 0


def _set_model_name(model: str, *, pull: bool = False) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    llm = config.setdefault("llm", {})
    llm["provider"] = "ollama"
    llm["model"] = model
    llm["planner_model"] = model
    llm["mode"] = "custom"
    llm["base_url"] = "http://localhost:11434"
    llm["api_key_env"] = None
    save_config(config, config_path)
    print(f"Model set to {model}.")
    print(f"Planner model set to {model}.")
    if pull:
        return _pull_ollama_model(model)
    _print_model_hint(config["llm"])
    return 0


def _set_provider_config(
    config: dict[str, Any],
    *,
    provider: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    llm = config.setdefault("llm", {})
    previous_model = str(llm.get("model") or "").strip()
    requested_model = str(model or "").strip()
    llm["provider"] = provider
    llm["base_url"] = base_url or _default_provider_base_url(provider)
    llm["api_key_env"] = api_key_env or _default_provider_api_key_env(provider)

    if provider in {"ollama", "local"}:
        selected_model = (
            requested_model
            or (previous_model if previous_model and not _looks_like_cloud_model(previous_model) else "qwen3:8b")
        )
        llm["mode"] = "fast" if selected_model == "qwen3:8b" else "custom"
        llm["model"] = selected_model
        llm["planner_model"] = selected_model
        llm["api_key_env"] = None
    elif provider == "openai":
        selected_model = requested_model or (
            previous_model if _looks_like_openai_chat_model(previous_model) else "gpt-4o-mini"
        )
        llm["mode"] = "openai"
        llm["model"] = selected_model
        llm["planner_model"] = selected_model
    elif provider == "deepseek":
        selected_model = requested_model or (
            previous_model if _looks_like_deepseek_model(previous_model) else "deepseek-v4-flash"
        )
        llm["mode"] = "deepseek"
        llm["model"] = selected_model
        llm["planner_model"] = selected_model
    else:
        llm["mode"] = provider
        if requested_model or not previous_model:
            selected_model = requested_model or provider
            llm["model"] = selected_model
            llm["planner_model"] = selected_model
    return llm


def command_tools(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    manifests = load_builtin_manifests()

    if args.tools_command == "list":
        rows = []
        overrides = config.get("tools", {})
        for name, manifest in manifests.items():
            override = overrides.get(name, {})
            enabled = bool(override.get("enabled", manifest.default_enabled))
            rows.append(
                [
                    name,
                    str(int(manifest.risk)),
                    "yes" if enabled else "no",
                    "yes" if manifest.approval_required else "no",
                    ", ".join(manifest.permissions) or "-",
                ]
            )
        _print_rows(["Tool", "Risk", "Enabled", "Approval", "Permissions"], rows)
        return 0

    if args.tool not in manifests:
        print(f"Unknown tool: {args.tool}", file=sys.stderr)
        return 1

    config.setdefault("tools", {}).setdefault(args.tool, {})["enabled"] = (
        args.tools_command == "enable"
    )
    save_config(config, config_path)
    print(f"Tool {args.tool} {'enabled' if args.tools_command == 'enable' else 'disabled'}.")
    return 0


def command_permissions(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    permissions = set(config.setdefault("permissions", {}).setdefault("granted", []))
    manifests = load_builtin_manifests()

    if args.permissions_command == "list":
        if not permissions:
            print("No permissions granted.")
            return 0
        for permission in sorted(permissions):
            print(permission)
        return 0

    target_permissions = _permissions_for_target(args.permission, manifests)
    if args.permissions_command == "grant":
        for permission in target_permissions:
            permissions.add(permission)
        print(f"Permission granted: {', '.join(target_permissions)}")
    elif args.permissions_command == "revoke":
        for permission in target_permissions:
            permissions.discard(permission)
        print(f"Permission revoked: {', '.join(target_permissions)}")

    config["permissions"]["granted"] = sorted(permissions)
    save_config(config, config_path)
    return 0


def command_terminal(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    terminal = _terminal_config_section(config)

    if args.terminal_command == "status":
        tool_enabled = bool(config.get("tools", {}).get("terminal.run", {}).get("enabled", False))
        permissions = set(config.get("permissions", {}).get("granted", []))
        _print_title("Terminal policy")
        _print_rows(
            ["Setting", "Value"],
            [
                ["Policy enabled", _yes_no(bool(terminal.get("enabled", False)))],
                ["terminal.run tool", _yes_no(tool_enabled)],
                ["terminal.run permission", _yes_no("terminal.run" in permissions)],
                ["Workspace only", _yes_no(bool(terminal.get("workspace_only", True)))],
                ["Workspace root", _terminal_workspace_root(terminal)],
                ["Timeout", f"{terminal.get('timeout_seconds', 30)}s"],
                ["Max output", str(terminal.get("max_output_chars", 20000))],
                ["Auto-approve allowlist", _yes_no(bool(terminal.get("auto_approve_allowlisted", False)))],
            ],
        )
        print("")
        _print_rows(
            ["Allowed command"],
            [[" ".join(command)] for command in _terminal_allowed_commands(terminal)],
        )
        return 0

    if args.terminal_command == "enable":
        terminal["enabled"] = True
        if args.tool:
            config.setdefault("tools", {}).setdefault("terminal.run", {})["enabled"] = True
        if args.grant_permission:
            permissions = set(config.setdefault("permissions", {}).setdefault("granted", []))
            permissions.add("terminal.run")
            config["permissions"]["granted"] = sorted(permissions)
        save_config(config, config_path)
        print("Terminal policy enabled.")
        if not args.tool:
            print("Next: dmdcore tools enable terminal.run")
        if not args.grant_permission:
            print("Next: dmdcore permissions grant terminal.run")
        print("Every terminal.run request still requires approval unless exact allowlist auto-approve is enabled.")
        return 0

    if args.terminal_command == "disable":
        terminal["enabled"] = False
        config.setdefault("tools", {}).setdefault("terminal.run", {})["enabled"] = False
        save_config(config, config_path)
        print("Terminal policy and terminal.run tool disabled.")
        return 0

    if args.terminal_command == "auto-approve":
        terminal["auto_approve_allowlisted"] = args.state == "on"
        save_config(config, config_path)
        print(f"Auto-approve exact allowlist: {args.state}.")
        return 0

    if args.terminal_command == "workspace":
        if args.path == "-":
            terminal["workspace_root"] = ""
            print("Terminal workspace reset to the private app workspace.")
        else:
            root = Path(args.path).expanduser()
            if not root.is_absolute():
                print("Workspace path must be absolute.", file=sys.stderr)
                return 1
            terminal["workspace_root"] = str(root)
            print(f"Terminal workspace root: {root}")
        save_config(config, config_path)
        return 0

    if args.terminal_command in {"allow", "remove"}:
        command = _terminal_command_from_args(args.command)
        if not command:
            print("Command is required.", file=sys.stderr)
            return 1
        allowed = _terminal_allowed_commands(terminal)
        command_tuple = tuple(command)
        if args.terminal_command == "allow":
            try:
                _validate_terminal_allowlist_command(command_tuple)
            except PermissionError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if command_tuple not in allowed:
                allowed.append(command_tuple)
            print(f"Allowlisted exact command: {' '.join(command_tuple)}")
        else:
            allowed = [item for item in allowed if item != command_tuple]
            print(f"Removed exact command if present: {' '.join(command_tuple)}")
        terminal["allowed_commands"] = [list(item) for item in allowed]
        save_config(config, config_path)
        return 0

    if args.terminal_command == "run":
        command = _terminal_command_from_args(args.command)
        if not command:
            print("Command is required.", file=sys.stderr)
            return 1
        response = build_agent_core().handle_tool_request(
            ToolRequest(
                tool="terminal.run",
                args={"command": command, "cwd": args.cwd},
                reason="User requested a terminal command from the CLI.",
            )
        )
        _print_agent_response(response.status, response.message, response.data)
        return 0 if response.status in {"ok", "approval_required"} else 1

    return 2


def command_memory(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    manager = MemoryManager(paths.memory)
    manager.bootstrap()

    if args.memory_command == "path":
        print(paths.memory)
        return 0
    if args.memory_command == "list":
        for item in manager.list_files():
            print(item)
        return 0
    if args.memory_command == "read":
        print(manager.read(args.path))
        return 0
    if args.memory_command == "open":
        _open_path(paths.memory)
        return 0
    return 2


def command_approvals(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    store = AuditStore(paths.audit_db)

    if args.approvals_command == "approve":
        core = build_agent_core()
        response = core.approve_and_execute(args.approval_id)
        _print_agent_response(response.status, response.message, response.data)
        return 0 if response.status == "ok" else 1

    if args.approvals_command == "deny":
        changed = store.set_approval_status(args.approval_id, "denied")
        print("Approval updated." if changed else "No pending approval found.")
        return 0 if changed else 1

    approvals = store.list_approvals(status="pending", limit=50)
    if not approvals:
        print("No pending approvals.")
        return 0
    _print_rows(
        ["ID", "Created", "Tool", "Risk", "Reason"],
        [
            [
                str(approval["id"]),
                approval["created_at"],
                approval["tool"] or "-",
                str(approval["risk"]),
                approval["reason"] or "-",
            ]
            for approval in approvals
        ],
    )
    return 0


def command_telegram(args: argparse.Namespace) -> int:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    telegram = _telegram_config_section(config)

    if args.telegram_command == "status":
        settings = settings_from_config(config)
        token_env = settings.bot_token_env
        print("Telegram interface")
        print(f"Enabled: {'yes' if settings.enabled else 'no'}")
        print(
            "Allowed user IDs: "
            + (
                ", ".join(str(item) for item in sorted(settings.allowed_user_ids))
                if settings.allowed_user_ids
                else "-"
            )
        )
        print(f"Bot token env: {token_env}")
        print(f"Bot token available: {'yes' if telegram_token_available(settings) else 'no'}")
        print(f"Polling timeout: {settings.polling_timeout_seconds}s")
        return 0

    if args.telegram_command == "enable":
        telegram["enabled"] = True
        save_config(config, config_path)
        print("Telegram interface enabled.")
        print("Only allowlisted Telegram user IDs can use it.")
        return 0

    if args.telegram_command == "disable":
        telegram["enabled"] = False
        save_config(config, config_path)
        print("Telegram interface disabled.")
        return 0

    if args.telegram_command == "allow":
        allowed = _telegram_allowed_user_ids(telegram)
        allowed.add(int(args.user_id))
        telegram["allowed_user_ids"] = sorted(allowed)
        save_config(config, config_path)
        print(f"Telegram user allowed: {args.user_id}")
        return 0

    if args.telegram_command == "remove":
        allowed = _telegram_allowed_user_ids(telegram)
        allowed.discard(int(args.user_id))
        telegram["allowed_user_ids"] = sorted(allowed)
        save_config(config, config_path)
        print(f"Telegram user removed: {args.user_id}")
        return 0

    if args.telegram_command == "run":
        spawned: list[tuple[str, subprocess.Popen[Any]]] = []
        try:
            settings = settings_from_config(config)
            if not load_telegram_token(settings):
                raise TelegramConfigError(
                    f"Missing Telegram bot token. Load it with /telegram token <bot_token> "
                    f"or export {settings.bot_token_env} before running."
                )
            if str(config.get("llm", {}).get("provider", "ollama")).lower() in {"ollama", "local"}:
                ollama_process = _ensure_ollama(config, verbose=False)
                if ollama_process is not None:
                    spawned.append(("Ollama", ollama_process))
            interface = TelegramInterface.from_current_config()
            print("Telegram polling started.")
            print("Press Ctrl+C to stop.")
            interface.run_polling(once=args.once, timeout_seconds=args.timeout)
            return 0
        except TelegramConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("")
            print("Telegram polling stopped.")
            return 0
        finally:
            _stop_processes(spawned)

    return 2


def _print_rows(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def _print_chat_header(config: dict[str, Any]) -> None:
    llm = config.get("llm", {})
    setup = config.get("setup", {})
    agent_name = str(setup.get("agent_name") or "DMD Agent")
    user_name = str(setup.get("user_name") or "").strip()
    _print_title(f"{agent_name} Terminal Chat")
    if user_name:
        print(f"User: {user_name}")
    print(f"Model: {llm.get('model', '-')}")
    print("Type a message and press Enter.")
    print("Commands: /panel, /help, /telegram, /permissions, /model fast --pull, /exit")
    print("")
    if str(llm.get("model", "")).strip().lower() == "dumb":
        print("The current model looks invalid.")
        print("Fix it with: /model fast --pull")
        print("")


def _handle_chat_command(
    text: str,
    core: Any,
    background_services: list[BackgroundService],
) -> str:
    if not text.startswith("/"):
        return "unhandled"
    parts = text.split()
    command = parts[0].lower()
    args = parts[1:]

    if command in {"/exit", "/quit", "/q"}:
        return "exit"
    if command in {"/help", "/h"}:
        if args:
            _print_help_topic(args[0])
        else:
            _print_chat_help()
        return "handled"
    if command == "/panel":
        _print_control_panel()
        return "handled"
    if command == "/back":
        _print_control_panel()
        return "handled"
    if command in {"/telegram", "/telegram-setup"}:
        _handle_telegram_panel_command(args, background_services)
        return "handled"
    if command == "/status":
        command_status()
        return "handled"
    if command == "/doctor":
        command_doctor(json_output=False, check_network=True)
        return "handled"
    if command == "/setup":
        updated_config = _run_terminal_onboarding(AppPaths.default())
        _replace_core_config(core, updated_config)
        return "handled"
    if command == "/logs":
        _print_recent_logs()
        return "handled"
    if command == "/open":
        _open_url("http://127.0.0.1:5174")
        return "handled"
    if command == "/model":
        command_model(argparse.Namespace(args=args, pull=False))
        return "handled"
    if command == "/memory":
        _print_memory_files()
        return "handled"
    if command == "/read":
        if not args:
            print("Usage: /read long-term/profile.md")
        else:
            _print_memory_file(args[0])
        return "handled"
    if command == "/tools":
        if args:
            _handle_tool_panel_command(args, core)
            return "handled"
        command_tools(argparse.Namespace(tools_command="list", tool=None))
        return "handled"
    if command == "/tool":
        _handle_tool_panel_command(args, core)
        return "handled"
    if command in {"/permissions", "/permission"}:
        _handle_permissions_panel_command(args, core)
        return "handled"
    if command == "/approvals":
        command_approvals(argparse.Namespace(approvals_command="list", approval_id=None))
        return "handled"
    if command == "/approve":
        if not args or not args[0].isdigit():
            print("Usage: /approve 3")
        else:
            response = core.approve_and_execute(int(args[0]))
            _print_chat_response(response.status, response.message, response.data)
        return "handled"
    if command == "/deny":
        if not args or not args[0].isdigit():
            print("Usage: /deny 3")
        else:
            changed = AuditStore(AppPaths.default().audit_db).set_approval_status(int(args[0]), "denied")
            print("Approval denied." if changed else "No pending approval found.")
        return "handled"
    if command == "/clear":
        print("\033c", end="")
        return "handled"

    print(f"Unknown command: {command}")
    print("Type /help for available commands.")
    return "handled"


def _replace_core_config(core: Any, config: dict[str, Any]) -> None:
    runtime_context = getattr(core, "runtime_context", None)
    current_config = getattr(runtime_context, "config", None)
    if isinstance(current_config, dict):
        current_config.clear()
        current_config.update(config)


def _refresh_core_from_disk(core: Any) -> None:
    paths = AppPaths.default()
    config = load_config(write_default_config(paths.config))
    _replace_core_config(core, config)
    if hasattr(core, "permission_context"):
        context = permission_context_from_config(config)
        if bool(getattr(core, "_cloud_context_approved", False)):
            context = replace(context, cloud_context_approved=True)
        core.permission_context = context


def _print_control_panel() -> None:
    paths = AppPaths.default()
    config = load_config(write_default_config(paths.config))
    setup = config.get("setup", {})
    llm = config.get("llm", {})
    telegram = settings_from_config(config)
    _print_title("Terminal Control Panel")
    _print_rows(
        ["Area", "Current", "Open"],
        [
            [
                "Identity",
                f"{setup.get('agent_name', 'DMD Agent')} / {setup.get('user_name') or 'no user name'}",
                "/setup",
            ],
            [
                "Model",
                f"{llm.get('provider', '-')} / {llm.get('model', '-')}",
                "/model",
            ],
            [
                "Telegram",
                "enabled" if telegram.enabled else "disabled",
                "/telegram",
            ],
            [
                "Permissions",
                f"{len(config.get('permissions', {}).get('granted', []))} granted",
                "/permissions",
            ],
            ["Approvals", "pending actions", "/approvals"],
            ["Logs", "recent audit events", "/logs"],
        ],
    )
    print("")
    print("Use /back to return to this panel from a submenu.")


def _start_telegram_background_if_ready(
    config: dict[str, Any],
    background_services: list[BackgroundService],
    *,
    explicit: bool,
) -> bool:
    if _background_service_running(background_services, "Telegram"):
        if explicit:
            print("Telegram: already running in background.")
        return True

    settings = settings_from_config(config)
    if not settings.enabled:
        if explicit:
            print("Telegram: disabled. Use /telegram setup or /telegram enable.")
        return False
    if not settings.allowed_user_ids:
        if explicit:
            print("Telegram: no allowed user IDs. Use /telegram setup or /telegram allow <id>.")
        return False
    if not load_telegram_token(settings):
        if explicit:
            print(
                "Missing Telegram bot token. Use /telegram token <bot_token> "
                f"or export {settings.bot_token_env} before starting."
            )
        else:
            print(f"Telegram: enabled, but {settings.bot_token_env} is not loaded.")
            print("Use /telegram token <bot_token> once, or export it before start web.")
        return False

    try:
        interface = TelegramInterface.from_current_config()
    except TelegramConfigError as exc:
        if explicit:
            print(str(exc))
        return False

    stop_event = threading.Event()

    def run() -> None:
        try:
            interface.run_polling(timeout_seconds=2, stop_event=stop_event)
        except TelegramError as exc:
            print("")
            print(f"Telegram stopped: {exc}")
        except Exception as exc:  # pragma: no cover - defensive background guard
            print("")
            print(f"Telegram stopped unexpectedly: {exc}")

    thread = threading.Thread(target=run, name="dmdcore-telegram", daemon=True)
    thread.start()
    background_services.append(
        BackgroundService(name="Telegram", thread=thread, stop_event=stop_event)
    )
    print("Telegram: running in background.")
    return True


def _background_service_running(
    background_services: list[BackgroundService],
    name: str,
) -> bool:
    return any(service.name == name and service.thread.is_alive() for service in background_services)


def _stop_background_services(background_services: list[BackgroundService]) -> None:
    for service in background_services:
        if service.thread.is_alive():
            print(f"Stopping {service.name}...")
            service.stop_event.set()
    for service in background_services:
        if service.thread.is_alive():
            service.thread.join(timeout=3)


def _handle_telegram_panel_command(
    args: list[str],
    background_services: list[BackgroundService],
) -> None:
    action = args[0].lower() if args else "menu"
    rest = args[1:]
    if action in {"menu", "help"}:
        _print_telegram_panel()
        return
    if action in {"setup", "wizard"}:
        _run_telegram_setup_wizard(background_services)
        return
    if action == "status":
        command_telegram(argparse.Namespace(telegram_command="status"))
        return
    if action == "enable":
        command_telegram(argparse.Namespace(telegram_command="enable"))
        return
    if action == "disable":
        command_telegram(argparse.Namespace(telegram_command="disable"))
        return
    if action == "allow":
        if not rest or not rest[0].isdigit():
            print("Usage: /telegram allow <telegram_user_id>")
            return
        command_telegram(argparse.Namespace(telegram_command="allow", user_id=int(rest[0])))
        return
    if action == "remove":
        if not rest or not rest[0].isdigit():
            print("Usage: /telegram remove <telegram_user_id>")
            return
        command_telegram(argparse.Namespace(telegram_command="remove", user_id=int(rest[0])))
        return
    if action == "token":
        if not rest:
            print("Usage: /telegram token <bot_token>")
            print("The token is loaded into this process environment only, never config.yaml.")
            return
        config = load_config(write_default_config(AppPaths.default().config))
        settings = settings_from_config(config)
        storage = store_telegram_token(settings, " ".join(rest).strip())
        print(f"Telegram token loaded into {storage} as {settings.bot_token_env}.")
        _start_telegram_background_if_ready(config, background_services, explicit=False)
        return
    if action in {"run", "start"}:
        config = load_config(write_default_config(AppPaths.default().config))
        _start_telegram_background_if_ready(config, background_services, explicit=True)
        return
    if action == "once":
        command_telegram(argparse.Namespace(telegram_command="run", once=True, timeout=10))
        return
    print(f"Unknown Telegram command: {action}")
    _print_telegram_panel()


def _print_telegram_panel() -> None:
    _print_title("Telegram Panel")
    command_telegram(argparse.Namespace(telegram_command="status"))
    print("")
    _print_rows(
        ["Command", "Meaning"],
        [
            ["/telegram setup", "Guided setup inside this chat."],
            ["/telegram token <token>", "Load BotFather token for this terminal only."],
            ["/telegram allow <id>", "Allow your Telegram user ID."],
            ["/telegram enable", "Enable Telegram access."],
            ["/telegram disable", "Disable Telegram access."],
            ["/telegram run", "Start polling in this same terminal panel."],
            ["/telegram once", "Poll once, useful while discovering /id."],
            ["/telegram status", "Show current Telegram state."],
            ["/back", "Return to the main panel."],
        ],
    )
    print("")
    print("Security: the bot token value is never saved in config.yaml and is lost on process restart.")


def _run_telegram_setup_wizard(background_services: list[BackgroundService]) -> None:
    paths = AppPaths.default()
    paths.ensure()
    config_path = write_default_config(paths.config)
    config = load_config(config_path)
    settings = settings_from_config(config)
    token_env = settings.bot_token_env

    _print_title("Telegram Setup")
    print("This setup stays inside the current terminal chat.")
    print("Token values are loaded into this process environment only, never config.yaml.")
    print("")

    if load_telegram_token(settings):
        print(f"Bot token: available in {token_env}")
    else:
        if sys.stdin.isatty():
            token = getpass.getpass(f"Paste BotFather token for {token_env} (hidden): ").strip()
        else:
            token = input(f"Paste BotFather token for {token_env}: ").strip()
        if not token:
            print("No token loaded. Use /telegram token <bot_token> or run /telegram setup again.")
            return
        storage = store_telegram_token(settings, token)
        print(f"Bot token loaded into {storage}.")

    raw_user_id = _ask_text(
        "Telegram user ID (paste it, or press Enter to poll /id once)",
        "",
    )
    if not raw_user_id:
        print("Open your bot in Telegram, send /id, then press Enter here.")
        input("Press Enter after sending /id: ")
        try:
            interface = TelegramInterface.from_current_config()
            interface.run_polling(once=True, timeout_seconds=10)
            print("If the bot replied with an ID in Telegram, paste it below.")
        except (TelegramConfigError, TelegramError) as exc:
            print(f"Telegram polling failed: {exc}")
            return
        raw_user_id = _ask_text("Telegram user ID", "")

    if not raw_user_id.isdigit():
        print("Telegram user ID must be a number.")
        return

    telegram = _telegram_config_section(config)
    allowed = _telegram_allowed_user_ids(telegram)
    allowed.add(int(raw_user_id))
    telegram["allowed_user_ids"] = sorted(allowed)
    telegram["enabled"] = True
    save_config(config, config_path)
    print("")
    print(_style("Telegram enabled.", "1;32"))
    print(f"Allowed user: {raw_user_id}")
    print("")

    start_now = _ask_choice(
        "Start Telegram polling now in this same panel?",
        [("yes", "Yes, start now"), ("no", "No, I will start it later")],
        default="yes",
    )
    if start_now == "yes":
        _start_telegram_background_if_ready(config, background_services, explicit=True)
    else:
        print("Start later with: /telegram run")


def _handle_permissions_panel_command(args: list[str], core: Any) -> None:
    action = args[0].lower() if args else "menu"
    if action in {"menu", "help", "list"}:
        _print_permissions_panel()
        return
    if action in {"grant", "revoke"}:
        if len(args) < 2:
            print(f"Usage: /permission {action} <permission>")
            return
        permission_target = _normalize_tool_alias(args[1])
        command_permissions(argparse.Namespace(permissions_command=action, permission=permission_target))
        _refresh_core_from_disk(core)
        return
    print(f"Unknown permission command: {action}")
    _print_permissions_panel()


def _normalize_tool_alias(value: str) -> str:
    normalized = value.strip()
    aliases = {
        "calendar_event": "calendar.create_event",
        "calendar.event": "calendar.create_event",
        "calendar.create": "calendar.create_event",
        "calendar_create_event": "calendar.create_event",
        "reminder": "calendar.create_event",
        "telegram": "telegram",
    }
    return aliases.get(normalized, normalized)


def _permissions_for_target(target: str, manifests: dict[str, Any]) -> list[str]:
    normalized = _normalize_tool_alias(target)
    manifest = manifests.get(normalized)
    if manifest is not None:
        if manifest.permissions:
            return list(manifest.permissions)
        print(f"Tool {normalized} does not require extra permissions.")
        return [normalized]
    return [normalized]


def _print_permissions_panel() -> None:
    paths = AppPaths.default()
    config = load_config(write_default_config(paths.config))
    granted = sorted(config.get("permissions", {}).get("granted", []))
    known_permissions = sorted(
        {
            permission
            for manifest in load_builtin_manifests().values()
            for permission in manifest.permissions
        }
    )
    _print_title("Permissions Panel")
    print("Granted permissions: " + (", ".join(granted) if granted else "-"))
    print("")
    _print_rows(
        ["Command", "Meaning"],
        [
            ["/tools", "List all tools, risk levels, and enabled state."],
            ["/tool enable <tool>", "Enable a tool."],
            ["/tool disable <tool>", "Disable a tool."],
            ["/permission grant <permission>", "Grant a permission."],
            ["/permission revoke <permission>", "Revoke a permission."],
            ["/approvals", "List pending risky actions."],
            ["/back", "Return to the main panel."],
        ],
    )
    print("")
    print("Common permissions: " + (", ".join(known_permissions[:12]) if known_permissions else "-"))
    print("Risky tools still require /approve even when enabled.")


def _handle_tool_panel_command(args: list[str], core: Any) -> None:
    if len(args) < 2 or args[0].lower() not in {"enable", "disable"}:
        if len(args) == 1 and "." in args[0]:
            action, raw_tool = args[0].split(".", 1)
            args = [action, raw_tool]
        else:
            print("Usage: /tool enable <tool> or /tool disable <tool>")
            print("Use /tools to list tool names.")
            return
    if len(args) < 2 or args[0].lower() not in {"enable", "disable"}:
        print("Usage: /tool enable <tool> or /tool disable <tool>")
        print("Use /tools to list tool names.")
        return
    tool = _normalize_tool_alias(args[1])
    command_tools(
        argparse.Namespace(
            tools_command=args[0].lower(),
            tool=tool,
        )
    )
    _refresh_core_from_disk(core)
    if args[0].lower() == "enable":
        manifests = load_builtin_manifests()
        manifest = manifests.get(tool)
        if manifest and manifest.permissions:
            print("Required permission: " + ", ".join(manifest.permissions))
            print(f"Grant it with: /permission grant {tool}")


def _print_chat_help() -> None:
    _print_title("Chat Commands")
    rows = [
        ["/panel", "Show the main terminal control panel."],
        ["/help", "Show this help."],
        ["/help telegram", "How to set up a Telegram bot."],
        ["/telegram", "Telegram setup and controls."],
        ["/back", "Return to the main control panel."],
        ["/exit", "Close terminal chat."],
        ["/model", "Show current model and presets."],
        ["/model fast --pull", "Use the default local model and pull it."],
        ["/model light --pull", "Use a smaller local model and pull it."],
        ["/doctor", "Check local health and security."],
        ["/setup", "Run first-time setup again."],
        ["/memory", "List local memory files."],
        ["/read long-term/profile.md", "Read a memory file."],
        ["/tools", "List tools and risk levels."],
        ["/tool enable <tool>", "Enable a tool in this chat."],
        ["/permissions", "Show access and permission controls."],
        ["/permission grant <permission>", "Grant a permission in this chat."],
        ["/approvals", "List pending approvals."],
        ["/logs", "Show recent local audit events."],
        ["/approve 3", "Approve and execute pending request 3."],
        ["/deny 3", "Deny pending request 3."],
        ["/open", "Open the browser dashboard."],
    ]
    _print_rows(["Command", "Meaning"], rows)
    print("")
    print("Help topics: telegram, models, memory, approvals, logs, troubleshooting")


def _print_help_topic(topic: str) -> None:
    normalized = topic.lower().strip()
    if normalized in {"telegram", "tg", "bot"}:
        _print_telegram_setup_guide()
        return
    if normalized in {"models", "model"}:
        _print_title("Model Help")
        print("Use local Ollama models by default:")
        print("  /model fast --pull")
        print("  /model light --pull")
        print("")
        print("From the shell:")
        print("  start model fast --pull")
        print("  start model use qwen3:14b --pull")
        return
    if normalized in {"memory", "remember"}:
        _print_title("Memory Help")
        print("Memory is stored as local Markdown files.")
        print("Commands:")
        print("  /memory")
        print("  /read long-term/profile.md")
        print("Writes require approval when routed through tools.")
        return
    if normalized in {"approvals", "approval"}:
        _print_title("Approvals Help")
        print("Risky actions create pending approvals.")
        print("Commands:")
        print("  /approvals")
        print("  /approve <id>")
        print("  /deny <id>")
        return
    if normalized in {"permissions", "permission", "access"}:
        _print_permissions_panel()
        return
    if normalized in {"logs", "log"}:
        _print_title("Logs Help")
        print("Use /logs to view recent audit events.")
        print("Telegram requests are recorded as telegram.request events.")
        return
    if normalized in {"troubleshooting", "trouble", "doctor"}:
        _print_title("Troubleshooting Help")
        print("Run /doctor inside chat or `start doctor` in the shell.")
        print("Common model fix:")
        print("  /model fast --pull")
        return
    print(f"Unknown help topic: {topic}")
    print("Try: /help telegram, /help models, /help memory, /help approvals")


def _is_telegram_setup_request(text: str) -> bool:
    normalized = text.lower()
    telegram_words = {"telegram", "телеграм"}
    setup_words = {
        "setup",
        "set up",
        "configure",
        "install",
        "create",
        "connect",
        "настрой",
        "настрои",
        "настроим",
        "настроя",
        "създай",
        "свържи",
        "пусни",
    }
    return any(word in normalized for word in telegram_words) and any(
        word in normalized for word in setup_words
    )


def _print_telegram_setup_guide() -> None:
    _print_title("Telegram Bot Setup")
    print("Goal: connect a private Telegram bot to this local agent.")
    print("")
    print("Security rules:")
    print("- Telegram is disabled by default.")
    print("- Only allowed Telegram user IDs can use it.")
    print("- The bot token is loaded into the current process environment, not config.yaml.")
    print("- Risky actions still require approval.")
    print("")
    print("Step 1: create a bot in Telegram")
    print("1. Open Telegram and search for BotFather.")
    print("2. Send /newbot.")
    print("3. Choose a display name and username.")
    print("4. Copy the token BotFather gives you.")
    print("")
    print("Step 2: load the token once")
    print("  /telegram token 123456:your-token")
    print("  # or: export DMDCORE_TELEGRAM_BOT_TOKEN=\"123456:your-token\"")
    print("")
    print("Step 3: find your Telegram user ID")
    print("1. Run:")
    print("  start telegram run")
    print("2. In Telegram, send /id to your bot.")
    print("3. Copy the user ID it returns.")
    print("")
    print("Step 4: allow your user ID and enable Telegram")
    print("  start telegram allow <telegram_user_id>")
    print("  start telegram enable")
    print("")
    print("Step 5: run Telegram polling")
    print("  start telegram run")
    print("")
    print("Check status any time:")
    print("  start telegram status")
    print("")
    print("Inside Telegram you can use:")
    print("  /id")
    print("  /help")
    print("  /approvals")
    print("  /approve <id>")
    print("  /deny <id>")


def _print_chat_response(
    status: str,
    message: str,
    data: dict[str, Any] | None,
) -> None:
    if status == "ok":
        print(_style("agent> ", "1;32") + message)
        if message in {
            "Saved to memory.",
            "Saved to local calendar store. Active reminder notifications are not implemented yet.",
        }:
            return
        _print_chat_data(data)
        return
    if status == "approval_required":
        approval_id = None if data is None else data.get("approval_id")
        print(_style("approval required> ", "1;33") + message)
        if approval_id is not None:
            print(f"Approve with: /approve {approval_id}")
            print(f"Deny with:    /deny {approval_id}")
        return
    if status == "denied":
        print(_style("blocked> ", "1;31") + message)
        missing = [] if data is None else data.get("missing_permissions", [])
        if missing:
            print("Missing permissions: " + ", ".join(str(item) for item in missing))
        return
    if status == "error":
        print(_style("error> ", "1;31") + message)
        if "Cannot connect to Ollama" in message:
            print("Fix:")
            print("  /model fast --pull")
            print("  or run: dmdcore model fast --pull")
        return
    print(_style(f"{status}> ", "1;36") + message)
    _print_chat_data(data)


def _print_chat_data(data: dict[str, Any] | None) -> None:
    if not data:
        return
    if set(data.keys()) == {"planner"}:
        return
    if "files" in data and isinstance(data["files"], list):
        files = data["files"]
        if not files:
            print("No memory files yet.")
            return
        print("Memory files:")
        for item in files:
            print(f"- {item}")
        return
    if "tools" in data and isinstance(data["tools"], list):
        _print_rows(
            ["Tool", "Risk", "Enabled", "Approval"],
            [
                [
                    str(tool.get("name", "-")),
                    str(tool.get("risk", "-")),
                    _yes_no(bool(tool.get("enabled", False))),
                    _yes_no(bool(tool.get("approval_required", False))),
                ]
                for tool in data["tools"]
                if isinstance(tool, dict)
            ],
        )
        return
    if "stdout" in data or "stderr" in data:
        print(f"Return code: {data.get('returncode', '-')}")
        stdout = str(data.get("stdout", "")).strip()
        stderr = str(data.get("stderr", "")).strip()
        if stdout:
            print(stdout)
        if stderr:
            print(_style(stderr, "31"))
        return
    print(json.dumps(data, indent=2, sort_keys=True))


def _print_memory_files() -> None:
    paths = AppPaths.default()
    manager = MemoryManager(paths.memory)
    manager.bootstrap()
    files = manager.list_files()
    if not files:
        print("No memory files yet.")
        return
    print("Memory files:")
    for item in files:
        print(f"- {item}")


def _print_memory_file(path: str) -> None:
    manager = MemoryManager(AppPaths.default().memory)
    try:
        print(manager.read(path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot read memory file: {exc}")


def _print_recent_logs(limit: int = 20) -> None:
    store = AuditStore(AppPaths.default().audit_db)
    events = store.list_recent_events(limit=limit)
    if not events:
        print("No audit events yet.")
        return
    _print_rows(
        ["Time", "Event", "Tool", "Status"],
        [
            [
                str(event.get("created_at", "-")),
                str(event.get("event_type", "-")),
                str(event.get("tool") or "-"),
                str(event.get("result_status") or "recorded"),
            ]
            for event in events
        ],
    )


def _ensure_ollama(config: dict[str, Any], *, verbose: bool) -> subprocess.Popen[Any] | None:
    llm = config.get("llm", {})
    base_url = str(llm.get("base_url", "http://localhost:11434")).rstrip("/")
    if _http_ok(f"{base_url}/api/tags"):
        print("Ollama: running")
        return None
    if shutil.which("ollama") is None:
        print("Ollama: not found. Install it from https://ollama.com/download")
        return None
    print("Ollama: starting")
    process = _start_process(["ollama", "serve"], cwd=None, verbose=verbose)
    if _wait_http(f"{base_url}/api/tags", timeout_seconds=12):
        print("Ollama: ready")
    else:
        print("Ollama: started, but not ready yet.")
    return process


def _ensure_api(
    *,
    host: str,
    port: int,
    verbose: bool,
    start_telegram: bool = True,
) -> subprocess.Popen[Any] | None:
    if _http_ok(f"http://{host}:{port}/health"):
        print(f"API: already running on {host}:{port}")
        return None
    print(f"API: starting on {host}:{port}")
    env = os.environ if start_telegram else {**os.environ, "DMDCORE_NO_TELEGRAM": "1"}
    process = _start_process(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dmdcore.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=_repo_root(),
        verbose=verbose,
        env=env,
    )
    if not _wait_http(f"http://{host}:{port}/health", timeout_seconds=15):
        raise RuntimeError("API did not become ready. Run `dmdcore doctor` for details.")
    print("API: ready")
    return process


def _ensure_dashboard(
    *,
    host: str,
    ui_port: int,
    api_url: str,
    verbose: bool,
) -> subprocess.Popen[Any] | None:
    if _http_ok(f"http://{host}:{ui_port}/"):
        print(f"Dashboard: already running on {host}:{ui_port}")
        return None
    frontend_dir = _frontend_dir()
    if not frontend_dir.exists():
        raise RuntimeError(f"Dashboard source not found: {frontend_dir}")
    if shutil.which("npm") is None:
        raise RuntimeError("npm is not installed. Install Node.js, then run `dmdcore start` again.")
    if not (frontend_dir / "node_modules").exists():
        print("Dashboard: installing frontend packages")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Frontend package install failed.") from exc
    print(f"Dashboard: starting on {host}:{ui_port}")
    env = {
        **os.environ,
        "DMDCORE_API_URL": api_url,
    }
    process = _start_process(
        ["npm", "run", "dev", "--", "--host", host, "--port", str(ui_port)],
        cwd=frontend_dir,
        env=env,
        verbose=verbose,
    )
    if not _wait_http(f"http://{host}:{ui_port}/", timeout_seconds=20):
        raise RuntimeError("Dashboard did not become ready.")
    print("Dashboard: ready")
    return process


def _start_process(
    command: list[str],
    *,
    cwd: Path | None,
    verbose: bool,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    output = None if verbose else subprocess.DEVNULL
    return subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        env=env,
        stdout=output,
        stderr=output,
    )


def _stop_processes(processes: list[tuple[str, subprocess.Popen[Any]]]) -> None:
    for name, process in reversed(processes):
        if process.poll() is None:
            print(f"Stopping {name}...")
            process.terminate()
    deadline = time.monotonic() + 5
    for _name, process in reversed(processes):
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return 200 <= int(response.status) < 500
    except (OSError, URLError):
        return False


def _http_json(url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            raw = response.read().decode("utf-8")
    except (OSError, URLError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _wait_http(url: str, *, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _http_ok(url):
            return True
        time.sleep(0.3)
    return False


def _pull_ollama_model(model: str) -> int:
    if not model:
        print("No model configured.", file=sys.stderr)
        return 1
    if shutil.which("ollama") is None:
        print("Ollama is not installed. Install it from https://ollama.com/download", file=sys.stderr)
        return 1
    print(f"Pulling Ollama model: {model}")
    completed = subprocess.run(["ollama", "pull", model], check=False)
    if completed.returncode == 0:
        print(f"Model ready: {model}")
    return int(completed.returncode)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _frontend_dir() -> Path:
    return _repo_root() / "frontend"


def _print_doctor_report(checks: list[Any]) -> None:
    summary = doctor_summary(checks)
    _print_title("DMD Agent Doctor")
    print(
        f"{_style(str(summary['ok']), '32')} ok, "
        f"{_style(str(summary['warn']), '33')} warnings, "
        f"{_style(str(summary['fail']), '31')} failures"
    )
    print("")
    _print_rows(
        ["Status", "Area", "Check", "Hint"],
        [
            [
                _status_badge(check.status),
                check.area,
                check.message,
                check.hint or "-",
            ]
            for check in checks
        ],
    )


def _print_title(title: str) -> None:
    print(_style(title, "1;36"))


def _status_badge(status: str) -> str:
    if status == "ok":
        return _style("OK", "32")
    if status == "warn":
        return _style("WARN", "33")
    if status == "fail":
        return _style("FAIL", "31")
    return status.upper()


def _style(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _print_model_hint(llm: dict[str, Any]) -> None:
    provider = str(llm.get("provider", "ollama")).lower()
    model = str(llm.get("model", ""))
    if provider in {"ollama", "local"}:
        print(f"Install it with: ollama pull {model}")
        return
    api_key_env = llm.get("api_key_env")
    if api_key_env:
        print(f"Before use, export {api_key_env}=<api-key> in the shell environment.")
    else:
        print("Before use, start the configured local OpenAI-compatible server.")


def _default_provider_base_url(provider: str) -> str:
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


def _default_provider_api_key_env(provider: str) -> str | None:
    if provider == "openai":
        return "DMDCORE_OPENAI_API_KEY"
    if provider == "openrouter":
        return "DMDCORE_OPENROUTER_API_KEY"
    if provider == "deepseek":
        return "DMDCORE_DEEPSEEK_API_KEY"
    if provider in {"ollama", "local", "lmstudio", "vllm"}:
        return None
    return "DMDCORE_OPENAI_COMPATIBLE_API_KEY"


def _looks_like_openai_chat_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4"))


def _looks_like_deepseek_model(model: str) -> bool:
    return model.lower().startswith("deepseek-")


def _looks_like_cloud_model(model: str) -> bool:
    normalized = model.lower()
    return _looks_like_openai_chat_model(normalized) or _looks_like_deepseek_model(normalized) or "/" in normalized


def _terminal_config_section(config: dict[str, Any]) -> dict[str, Any]:
    terminal = config.setdefault("terminal", {})
    terminal.setdefault("enabled", False)
    terminal.setdefault("mode", "allowlist")
    terminal.setdefault("workspace_only", True)
    terminal.setdefault("workspace_root", "")
    terminal.setdefault("timeout_seconds", 30)
    terminal.setdefault("max_output_chars", 20000)
    terminal.setdefault("auto_approve_allowlisted", False)
    terminal.setdefault(
        "allowed_commands",
        [
            ["pwd"],
            ["ls"],
            ["git", "status"],
            ["git", "diff"],
            ["cat", "README.md"],
            ["cat", "readme.md"],
            ["npm", "test"],
            ["pytest"],
        ],
    )
    return terminal


def _terminal_workspace_root(terminal: dict[str, Any]) -> str:
    raw_root = terminal.get("workspace_root")
    if isinstance(raw_root, str) and raw_root.strip():
        return str(Path(raw_root).expanduser())
    return str(AppPaths.default().workspace)


def _terminal_allowed_commands(terminal: dict[str, Any]) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    raw_commands = terminal.setdefault("allowed_commands", [])
    if isinstance(raw_commands, list):
        for command in raw_commands:
            if isinstance(command, list) and command:
                commands.append(tuple(str(part) for part in command))
    return commands


def _terminal_command_from_args(raw: list[str]) -> list[str]:
    parts = list(raw)
    if parts and parts[0] == "--":
        parts = parts[1:]
    return [part for part in parts if part]


def _validate_terminal_allowlist_command(command: tuple[str, ...]) -> None:
    TerminalPolicy(enabled=True, allowed_commands=(command,)).validate(list(command))


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    else:
        print(path)


def _open_url(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", url], check=False)
    else:
        print(url)


def _print_agent_response(
    status: str,
    message: str,
    data: dict[str, Any] | None,
) -> None:
    print(f"[{status}] {message}")
    if data:
        print(json.dumps(data, indent=2, sort_keys=True))


def dump_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def _telegram_config_section(config: dict[str, Any]) -> dict[str, Any]:
    telegram = config.setdefault("interfaces", {}).setdefault("telegram", {})
    telegram.setdefault("enabled", False)
    telegram.setdefault("allowed_user_ids", [])
    telegram.setdefault("bot_token_env", DEFAULT_TOKEN_ENV)
    telegram.setdefault("polling_timeout_seconds", 30)
    return telegram


def _telegram_allowed_user_ids(telegram: dict[str, Any]) -> set[int]:
    allowed: set[int] = set()
    raw_allowed = telegram.setdefault("allowed_user_ids", [])
    if isinstance(raw_allowed, list):
        for item in raw_allowed:
            try:
                allowed.add(int(item))
            except (TypeError, ValueError):
                continue
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
