from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditStore
from dmdagent4all.config import load_config, save_config, write_default_config
from dmdagent4all.memory import MemoryManager
from dmdagent4all.model_presets import (
    MAC_MINI_RECOMMENDATIONS,
    MODEL_MODES,
    detect_memory_gb,
    recommend_for_memory,
)
from dmdagent4all.runtime import build_agent_core
from dmdagent4all.tools import load_builtin_manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dmdagent",
        description="DMD Agent 4 All local control center.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("status", help="Show local agent status.")
    subcommands.add_parser("wizard", help="Run the local install wizard.")
    subcommands.add_parser("init", help="Create local data directories and config.")
    chat_parser = subcommands.add_parser("chat", help="Send a message to the local agent.")
    chat_parser.add_argument("message", nargs="*", help="Message text. Omit for interactive mode.")

    models_parser = subcommands.add_parser("models", help="Manage local model settings.")
    models_subcommands = models_parser.add_subparsers(dest="models_command", required=True)
    models_subcommands.add_parser("list", help="List recommended model modes.")
    models_set_mode = models_subcommands.add_parser("set-mode", help="Set a recommended model mode.")
    models_set_mode.add_argument("mode", choices=[mode.key for mode in MODEL_MODES])
    models_set = models_subcommands.add_parser("set", help="Set an explicit Ollama model name.")
    models_set.add_argument("model")

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

    args = parser.parse_args(argv)

    if args.command == "init":
        return command_init()
    if args.command == "status":
        return command_status()
    if args.command == "wizard":
        return command_wizard()
    if args.command == "serve":
        return command_serve(args.host, args.port, args.reload)
    if args.command == "chat":
        return command_chat(" ".join(args.message))
    if args.command == "models":
        return command_models(args)
    if args.command == "tools":
        return command_tools(args)
    if args.command == "permissions":
        return command_permissions(args)
    if args.command == "memory":
        return command_memory(args)
    if args.command == "approvals":
        return command_approvals(args)

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
    print("DMD Agent 4 All status")
    print(f"Data directory: {paths.root}")
    print(f"Config: {config_path}")
    print(f"Memory: {paths.memory}")
    print(f"Workspace: {paths.workspace}")
    print(f"Audit DB: {paths.audit_db}")
    print(f"LLM provider: {config['llm']['provider']}")
    print(f"LLM model: {config['llm']['model']}")
    print(f"UI language: {config['interfaces']['ui_language']}")
    print(f"Assistant response language: {config['llm']['response_language']}")
    print(f"Terminal enabled: {config['terminal']['enabled']}")
    print(f"Browser enabled: {config['browser']['enabled']}")
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
    print("- Start the local API: dmdagent serve")
    return 0


def command_serve(host: str, port: int, reload: bool) -> int:
    import uvicorn

    uvicorn.run(
        "dmdagent4all.server:app",
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def command_chat(message: str) -> int:
    core = build_agent_core()
    if message.strip():
        response = core.handle_text(message)
        _print_agent_response(response.status, response.message, response.data)
        return 0 if response.status in {"ok", "approval_required", "denied"} else 1

    print("DMD Agent 4 All chat")
    print("Type /exit to quit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print("")
            return 0
        if line.strip() in {"/exit", "exit", "quit"}:
            return 0
        if not line.strip():
            continue
        response = core.handle_text(line)
        _print_agent_response(response.status, response.message, response.data)


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
        print(f"Current model: {config['llm'].get('model', '-')}")
        print(f"Current planner model: {config['llm'].get('planner_model') or config['llm'].get('model', '-')}")
        return 0

    if args.models_command == "set-mode":
        selected = next(mode for mode in MODEL_MODES if mode.key == args.mode)
        config.setdefault("llm", {})["mode"] = selected.key
        config["llm"]["model"] = selected.default_model
        config["llm"]["planner_model"] = selected.default_model
        save_config(config, config_path)
        print(f"Model mode set to {selected.label}.")
        print(f"Model: {selected.default_model}")
        print(f"Planner model: {selected.default_model}")
        print(f"Install it with: ollama pull {selected.default_model}")
        return 0

    if args.models_command == "set":
        config.setdefault("llm", {})["model"] = args.model
        config["llm"]["planner_model"] = args.model
        config["llm"]["mode"] = "custom"
        save_config(config, config_path)
        print(f"Model set to {args.model}.")
        print(f"Planner model set to {args.model}.")
        print(f"Install it with: ollama pull {args.model}")
        return 0

    return 2


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

    if args.permissions_command == "list":
        if not permissions:
            print("No permissions granted.")
            return 0
        for permission in sorted(permissions):
            print(permission)
        return 0

    if args.permissions_command == "grant":
        permissions.add(args.permission)
        print(f"Permission granted: {args.permission}")
    elif args.permissions_command == "revoke":
        permissions.discard(args.permission)
        print(f"Permission revoked: {args.permission}")

    config["permissions"]["granted"] = sorted(permissions)
    save_config(config, config_path)
    return 0


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
        changed = store.set_approval_status(args.approval_id, "approved")
        print("Approval updated." if changed else "No pending approval found.")
        return 0 if changed else 1

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


def _print_rows(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    else:
        print(path)


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


if __name__ == "__main__":
    raise SystemExit(main())
