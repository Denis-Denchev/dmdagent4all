from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from dmdagent4all.config import save_config
from dmdagent4all.memory import MemoryManager
from dmdagent4all.sandbox import TerminalPolicy, run_workspace_command
from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.browser_automation import browser_click, browser_fill_form, browser_submit
from dmdagent4all.tools.email_connector import make_email_handler
from dmdagent4all.tools.reminders import complete_reminder, create_reminder, list_reminders
from dmdagent4all.tools.registry import ToolRegistry, load_builtin_manifests
from dmdagent4all.tools.web import browser_extract_text, browser_open, browser_scrape_markdown
from dmdagent4all.workspace import WorkspaceManager


def build_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry(load_builtin_manifests())
    registry.register_handler("developer.context", _developer_context)
    registry.register_handler("system.list_enabled_tools", _list_enabled_tools)
    registry.register_handler("memory.list", _memory_list)
    registry.register_handler("memory.read", _memory_read)
    registry.register_handler("memory.write", _memory_write)
    registry.register_handler("memory.organize_long_term", _memory_organize_long_term)
    registry.register_handler("profile.update", _profile_update)
    registry.register_handler("files.read", _files_read)
    registry.register_handler("files.write", _files_write)
    registry.register_handler("files.delete", _files_delete)
    registry.register_handler("workspace.switch", _workspace_switch)
    registry.register_handler("workspace.status", _workspace_status)
    registry.register_handler("reminders.create", create_reminder)
    registry.register_handler("reminders.list", list_reminders)
    registry.register_handler("reminders.complete", complete_reminder)
    registry.register_handler("terminal.run", _terminal_run)
    registry.register_handler("browser.open", browser_open)
    registry.register_handler("browser.extract_text", browser_extract_text)
    registry.register_handler("browser.scrape_markdown", browser_scrape_markdown)
    registry.register_handler("browser.click", browser_click)
    registry.register_handler("browser.fill_form", browser_fill_form)
    registry.register_handler("browser.submit", browser_submit)
    registry.register_handler("calendar.create_event", _calendar_create_event)
    registry.register_handler("calendar.delete_event", _calendar_delete_event)
    registry.register_handler("calendar.find_free_slots", _calendar_find_free_slots)
    registry.register_handler("calendar.today", _calendar_today)
    registry.register_handler("calendar.update_event", _calendar_update_event)
    registry.register_handler("calendar.week", _calendar_week)
    for provider in {"gmail", "outlook"}:
        for action in {
            "archive",
            "create_draft",
            "read_thread",
            "reply_draft",
            "search",
            "send_draft",
            "summarize_inbox",
        }:
            registry.register_handler(f"{provider}.{action}", make_email_handler(provider, action))
    registry.register_handler("gmail.label", make_email_handler("gmail", "label"))
    return registry


def _list_enabled_tools(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del args
    tool_overrides = context.config.get("tools", {})
    manifests = load_builtin_manifests()
    tools = []
    for name, manifest in manifests.items():
        override = tool_overrides.get(name, {})
        enabled = bool(override.get("enabled", manifest.default_enabled))
        tools.append(
            {
                "name": name,
                "risk": int(manifest.risk),
                "enabled": enabled,
                "approval_required": manifest.approval_required,
                "permissions": list(manifest.permissions),
            }
        )
    return {"tools": tools}


def _memory_list(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    manager = MemoryManager(context.memory_root)
    manager.bootstrap()
    return {"files": manager.list_files()}


def _memory_read(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path", "long-term/profile.md"))
    manager = MemoryManager(context.memory_root)
    return {"path": path, "content": manager.read(path)}


def _memory_write(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    body = str(args.get("body") or "")
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")
    if path.lower() == "auto":
        path = ""
    if not path or not body:
        title = str(args.get("title") or "").strip()
        if not body:
            raise ValueError("memory.write requires body.")
        path = _memory_auto_path(title=title, body=body, args=args)
    metadata = dict(metadata or {})
    scope = _memory_scope(args, metadata)
    metadata["memory_scope"] = scope
    if "ttl_hours" in args and "ttl_hours" not in metadata:
        metadata["ttl_hours"] = args["ttl_hours"]
    if scope == "short-term" and not path.startswith("short-term/"):
        path = f"short-term/{path}"
    manager = MemoryManager(context.memory_root)
    written = manager.write(path, body, metadata=metadata)
    return {"path": str(written.relative_to(context.memory_root.resolve()))}


def _memory_organize_long_term(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    source = str(args.get("source_markdown") or "").strip()
    if len(source) < 100:
        raise ValueError("memory.organize_long_term requires a substantial source_markdown value.")
    today = str(args.get("today") or datetime.now().date().isoformat())
    source = redact_text(source)
    files = _long_term_memory_files(source, today=today)
    manager = MemoryManager(context.memory_root)
    written: list[str] = []
    summaries: dict[str, str] = {}
    for relative_path, body in files.items():
        target = manager.write(
            relative_path,
            body,
            metadata={
                "type": "organized_long_term_memory",
                "memory_scope": "long-term",
                "source": "bulk_memory_organizer",
                "confidence": "medium",
            },
        )
        path = str(target.relative_to(context.memory_root.resolve()))
        written.append(path)
        summaries[path] = _first_content_sentence(body)
    tree = _tree_from_paths(written)
    high_priority = [
        "owner/profile.md",
        "owner/preferences.md",
        "projects/primary-product/failover.md",
        "projects/primary-product/infrastructure.md",
        "projects/dmd-agent-4-all/security-model.md",
        "homelab/security.md",
        "security/incident-response.md",
        "assistant-behavior/emergency-mode.md",
    ]
    always = [path for path in high_priority if path in written]
    return {
        "files": written,
        "tree": tree,
        "summary": summaries,
        "ambiguities": _memory_ambiguities(source),
        "retrieval": {
            "always": always,
            "on_demand": [path for path in written if path not in set(always)],
        },
    }


def _profile_update(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    setup = context.config.setdefault("setup", {})
    llm = context.config.setdefault("llm", {})
    changed: dict[str, str] = {}
    for key in {"agent_name", "user_name", "nickname", "nickname_bg", "preferred_language"}:
        value = _clean_profile_value(args.get(key))
        if value is not None:
            setup[key] = value
            changed[key] = value
    response_language = _clean_profile_value(args.get("response_language"))
    if response_language is not None:
        llm["response_language"] = response_language
        changed["response_language"] = response_language
    if not changed:
        raise ValueError("profile.update requires at least one profile field.")
    setup.setdefault("agent_name", "DMD Agent")
    setup.setdefault("user_name", "")
    setup.setdefault("preferred_language", "auto")
    setup["completed"] = True
    if context.config_path is not None:
        save_config(context.config, context.config_path)
    _write_profile_memory(context)
    return {"changed": changed}


def _clean_profile_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().strip(" .,!?:;\"'")
    if not cleaned:
        return None
    if "\n" in cleaned or len(cleaned) > 120:
        raise ValueError("Profile values must be single-line strings up to 120 characters.")
    return cleaned


def _write_profile_memory(context: ToolRuntimeContext) -> None:
    setup = context.config.get("setup", {})
    MemoryManager(context.memory_root).write(
        "long-term/profile.md",
        "\n".join(
            [
                f"User name: {setup.get('user_name') or 'not set'}",
                f"Preferred nickname: {setup.get('nickname') or 'not set'}",
                f"Bulgarian nickname: {setup.get('nickname_bg') or setup.get('nickname') or 'not set'}",
                f"Assistant name: {setup.get('agent_name') or 'DMD Agent'}",
                f"Preferred response language: {setup.get('preferred_language') or 'auto'}",
            ]
        ),
        metadata={
            "type": "profile",
            "memory_scope": "long-term",
            "source": "profile.update",
            "confidence": "high",
        },
    )


def _files_read(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("files.read requires path.")
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    resolved = manager.validate_user_path(path)
    if not resolved.is_file():
        raise ValueError("files.read requires a file path.")
    body = resolved.read_bytes()
    max_bytes = int(context.config.get("files", {}).get("max_read_bytes", 200_000))
    truncated = len(body) > max_bytes
    if truncated:
        body = body[:max_bytes]
    return {
        "path": str(resolved),
        "content": redact_text(body.decode("utf-8", errors="replace")),
        "truncated": truncated,
    }


def _files_write(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("files.write requires path.")
    content = str(args.get("content") or "")
    overwrite = bool(args.get("overwrite", False))
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    resolved = manager.validate_user_path(path, allow_missing=True)
    existed = resolved.exists()
    if existed and resolved.is_dir():
        raise ValueError("files.write requires a file path, not a directory.")
    if existed and not overwrite:
        raise ValueError("File already exists. Set overwrite=true after explicit approval to replace it.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return {
        "path": str(resolved),
        "bytes": len(content.encode("utf-8")),
        "overwritten": existed and overwrite,
    }


def _files_delete(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("files.delete requires path.")
    recursive = bool(args.get("recursive", False))
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    resolved = manager.validate_user_path(path)
    if resolved.is_dir():
        if recursive:
            shutil.rmtree(resolved)
        else:
            resolved.rmdir()
        return {"path": str(resolved), "deleted": True, "kind": "directory"}
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    resolved.unlink()
    return {"path": str(resolved), "deleted": True, "kind": "file"}


def _developer_context(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    raw_workspace = args.get("workspace") or str(manager.current_workspace)
    workspace = manager.validate_workspace(str(raw_workspace))
    max_files = _bounded_int(args.get("max_files"), default=160, minimum=20, maximum=500)
    max_preview_bytes = _bounded_int(
        args.get("max_preview_bytes"),
        default=12_000,
        minimum=1_000,
        maximum=50_000,
    )

    files, truncated = _developer_file_tree(workspace, manager, max_files=max_files)
    focus_files = _developer_focus_files(
        args.get("focus_paths"),
        workspace=workspace,
        manager=manager,
        max_preview_bytes=max_preview_bytes,
    )
    return {
        "workspace": manager.workspace_info(),
        "scan_root": str(workspace),
        "files": files,
        "file_count": len(files),
        "truncated": truncated,
        "focus_files": focus_files,
        "available_actions": [
            {
                "tool": "files.read",
                "purpose": "Read non-secret files inside allowed workspace roots.",
            },
            {
                "tool": "files.write",
                "purpose": "Create or overwrite text files after explicit approval.",
            },
            {
                "tool": "terminal.run",
                "purpose": "Run allowlisted or policy-approved workspace commands after approval when required.",
            },
            {
                "tool": "workspace.switch",
                "purpose": "Change the active coding workspace after path validation.",
            },
        ],
        "safety": [
            "Secret files and blocked system paths are excluded.",
            "File writes and terminal execution remain approval/policy gated.",
            "Destructive database statements remain blocked by backend policy.",
        ],
    }


def _developer_file_tree(
    workspace: Path,
    manager: WorkspaceManager,
    *,
    max_files: int,
) -> tuple[list[dict[str, Any]], bool]:
    excluded_dirs = {
        ".git",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
    files: list[dict[str, Any]] = []
    for current_root, dirnames, filenames in os.walk(workspace):
        root_path = Path(current_root)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            path = root_path / dirname
            if dirname in excluded_dirs or manager.is_secret_path(path):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = root_path / filename
            if manager.is_secret_path(path):
                continue
            if len(files) >= max_files:
                return files, True
            try:
                relative = path.relative_to(workspace).as_posix()
                size = path.stat().st_size
            except OSError:
                continue
            files.append(
                {
                    "path": relative,
                    "size": size,
                    "previewable": _looks_previewable_code_file(path),
                }
            )
    return files, False


def _developer_focus_files(
    raw_focus_paths: Any,
    *,
    workspace: Path,
    manager: WorkspaceManager,
    max_preview_bytes: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_focus_paths, list):
        return []
    previews: list[dict[str, Any]] = []
    for raw_path in raw_focus_paths[:10]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        resolved = manager.validate_user_path(raw_path, base=workspace)
        if not resolved.is_file():
            continue
        try:
            relative = resolved.relative_to(workspace).as_posix()
        except ValueError:
            relative = str(resolved)
        if not _looks_previewable_code_file(resolved):
            previews.append({"path": relative, "previewable": False})
            continue
        body = resolved.read_bytes()
        truncated = len(body) > max_preview_bytes
        if truncated:
            body = body[:max_preview_bytes]
        previews.append(
            {
                "path": relative,
                "previewable": True,
                "truncated": truncated,
                "content": redact_text(body.decode("utf-8", errors="replace")),
            }
        )
    return previews


def _looks_previewable_code_file(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile", "README", "LICENSE"}:
        return True
    return path.suffix.casefold() in {
        ".css",
        ".csv",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _workspace_switch(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("workspace.switch requires path.")
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    target = manager.validate_workspace(path)
    target.mkdir(parents=True, exist_ok=True)
    workspace = context.config.setdefault("workspace", {})
    workspace["current_path"] = str(target)
    workspace.setdefault("default_path", str(manager.default_workspace))
    context.config.setdefault("terminal", {})["workspace_root"] = str(target)
    if context.config_path is not None:
        save_config(context.config, context.config_path)
    return {
        "current_workspace": str(target),
        "default_workspace": str(manager.default_workspace),
        "allowed_roots": [str(root) for root in manager.allowed_roots],
    }


def _workspace_status(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    return manager.workspace_info()


def _memory_scope(args: dict[str, Any], metadata: dict[str, Any]) -> str:
    raw_scope = (
        args.get("memory_scope")
        or args.get("scope")
        or metadata.get("memory_scope")
        or metadata.get("scope")
        or "long-term"
    )
    scope = str(raw_scope).strip().lower().replace("_", "-")
    return "short-term" if scope in {"short", "short-term", "temporary", "temp"} else "long-term"


def _memory_auto_path(*, title: str, body: str, args: dict[str, Any]) -> str:
    scope = _memory_scope(args, dict(args.get("metadata") or {}))
    raw_title = title or _first_meaningful_line(body) or "memory"
    slug = _slugify(raw_title) or "memory"
    prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = "short-term" if scope == "short-term" else "long-term/notes"
    return f"{folder}/{prefix}-{slug}.md"


def _first_meaningful_line(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip().strip("# ").strip()
        if stripped:
            return stripped
    return ""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.lower(), flags=re.UNICODE).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:80].strip("-")


def _long_term_memory_files(source: str, *, today: str) -> dict[str, str]:
    sections = _markdown_sections(source)

    def section(*names: str) -> str:
        values = [sections.get(name, "").strip() for name in names if sections.get(name, "").strip()]
        return "\n\n".join(values).strip()

    owner = section("Owner / User")
    motivation = section("Personal / Motivation")
    business = section("Main Business / Brand", "Business / Brand")
    primary_product = section("Project: Primary Product", "Project: Product")
    product_strategy = section("Primary Product Strategy", "Product Strategy")
    product_stack = section("Primary Product Stack", "Product Stack")
    product_features = section("Primary Product Features", "Product Features")
    product_deployment = section("Primary Product Deployment Pattern", "Product Deployment Pattern")
    product_failover = section("Primary Product High Availability / Failover Architecture", "Product High Availability / Failover Architecture")
    dmd_agent = section("Project: DMD Agent 4 All")
    homelab = section("Homelab Infrastructure")
    primary = section("Primary Server: homelab")
    backup = section("Backup Server: homelab2")
    older_infra = section("Older / Alternative Infrastructure Notes")
    immich = section("Immich")
    other_services = section("Other Homelab Services")
    npm = section("Nginx Proxy Manager")
    homepage = section("Homepage")
    grafana = section("Grafana")
    monitoring = section("Monitoring Services", "Monitoring")
    adguard = section("AdGuard Home")
    plex = section("Plex")
    paperless = section("Paperless NGX")
    security = section("Security Preferences / Rules")
    github = section("GitHub / Open Source Preferences")
    assistant = section("Preferred Assistant Behavior")
    commands = section(
        "Important Commands / Patterns",
        "Check Docker containers",
        "Check Docker disk usage",
        "Safe Docker cleanup",
        "Check UFW",
        "Check SSH effective config",
        "Check active SSH sessions",
        "Check failed services",
        "Check logs",
        "Check zombie processes",
        "Check Tailscale",
    )
    priorities = section("Current Priorities")

    del priorities
    files: dict[str, str] = {}

    def add(
        path: str,
        title: str,
        purpose: str,
        content: str,
        *,
        important_rules: list[str] | None = None,
    ) -> None:
        if not _has_real_memory_content(content):
            return
        files[path] = _memory_doc(title, purpose, today, content, important_rules=important_rules)

    add(
            "owner/profile.md",
            "Owner Profile",
            "Stable identity, background, work context and high-level owner facts.",
            owner,
        )
    add(
            "owner/preferences.md",
            "Owner Preferences",
            "Communication, work style and assistant behavior preferences for the owner.",
            _filter_lines(source, ["preferred", "language", "style", "tone", "assistant"]),
        )
    add(
            "owner/motivation.md",
            "Motivation",
            "Long-term motivation, personal context and why stability matters.",
            motivation,
        )
    add(
            "business/brand.md",
            "Brand",
            "Business and brand memory.",
            business,
        )
    add(
            "business/domains-and-email.md",
            "Domains And Email",
            "Domains, email providers and mailbox planning.",
            _filter_lines(business, ["domain", "email", "mailbox", "@"]),
        )
    add(
            "business/positioning.md",
            "Positioning",
            "Brand positioning, tone and business focus.",
            _filter_lines(business, ["position", "tone", "focus", "automation", "business"]),
        )
    add(
            "projects/primary-product/overview.md",
            "Primary Product Overview",
            "High-level product and business context.",
            primary_product,
        )
    add(
            "projects/primary-product/product-strategy.md",
            "Primary Product Strategy",
            "Strategy, monetization and validation priorities.",
            product_strategy,
        )
    add(
            "projects/primary-product/stack.md",
            "Primary Product Stack",
            "Backend, frontend, database and infrastructure stack.",
            product_stack,
        )
    add(
            "projects/primary-product/infrastructure.md",
            "Primary Product Infrastructure",
            "Production infrastructure, paths, containers, Cloudflare and replication context.",
            "\n\n".join([product_stack, product_failover]),
            important_rules=[
                "Do not change IP addresses, container names, paths, tunnel names or replication roles unless the owner confirms.",
                "Backup cloudflared should stay stopped except during failover.",
            ],
        )
    add(
            "projects/primary-product/failover.md",
            "Primary Product Failover",
            "Canonical high availability and failover procedure.",
            product_failover,
            important_rules=[
                "Avoid split-brain at all costs.",
                "Confirm the primary is truly down before promoting standby.",
                "After failover, old primary must not write as primary until carefully re-synced.",
            ],
        )
    add(
            "projects/primary-product/deployment.md",
            "Primary Product Deployment",
            "Deployment workflow and commands.",
            _commands_to_code_blocks(product_deployment),
        )
    add(
            "projects/primary-product/features.md",
            "Primary Product Features",
            "Known product features and integrations.",
            product_features,
        )
    add(
            "projects/dmd-agent-4-all/overview.md",
            "DMD Agent 4 All Overview",
            "High-level memory for the local-first AI control center project.",
            dmd_agent,
        )
    add(
            "projects/dmd-agent-4-all/current-features.md",
            "DMD Agent 4 All Current Features",
            "Current practical capabilities and usage scenarios.",
            _filter_lines(dmd_agent, ["feature", "current", "scenario", "reminder", "telegram", "dashboard", "terminal"]),
        )
    add(
            "projects/dmd-agent-4-all/security-model.md",
            "DMD Agent 4 All Security Model",
            "Security philosophy and constraints for the agent project.",
            _filter_lines(dmd_agent, ["security", "local-first", "allowlist", "approval", "audit", "secret", "dangerous"]),
            important_rules=[
                "No full dangerous shell by default.",
                "Risky actions require approvals.",
                "Secrets should never be committed or placed in model context.",
            ],
        )
    add(
            "projects/dmd-agent-4-all/open-source-strategy.md",
            "DMD Agent 4 All Open Source Strategy",
            "Open-source positioning and launch model.",
            _filter_lines(dmd_agent, ["open-source", "repository", "license", "contribution", "pull request", "launch"]),
        )
    add(
            "projects/dmd-agent-4-all/roadmap.md",
            "DMD Agent 4 All Roadmap",
            "Recommended files, branch model and launch next actions.",
            _filter_lines(dmd_agent, ["recommended", "branch", "main", "dev", "roadmap", "release"]),
        )
    add(
            "homelab/overview.md",
            "Homelab Overview",
            "High-level memory for the owner's homelab.",
            "\n\n".join([homelab, older_infra]),
        )
    add(
            "homelab/homelab-primary.md",
            "Primary Homelab Server",
            "Current primary server role, IPs, paths, services and ports.",
            primary,
        )
    add(
            "homelab/homelab2-backup.md",
            "Backup Homelab2 Server",
            "Backup server role, standby database and failover target facts.",
            backup,
            important_rules=["Do not promote homelab2 unless the primary is confirmed down."],
        )
    add(
            "homelab/services.md",
            "Homelab Services",
            "Service inventory across the homelab.",
            "\n\n".join(
                [
                    _filter_lines(primary, ["container", "service", "port"]),
                    other_services,
                    npm,
                    homepage,
                    grafana,
                    monitoring,
                    adguard,
                    plex,
                    paperless,
                ]
            ),
        )
    add(
            "homelab/networking.md",
            "Homelab Networking",
            "LAN, Tailscale, ports and exposure model.",
            _filter_lines(source, ["LAN IP", "Tailscale IP", "port", "Cloudflare", "Tailscale", "NPM", "proxy"]),
        )
    add(
            "homelab/security.md",
            "Homelab Security",
            "Security posture and hardening rules for homelab services.",
            "\n\n".join([_filter_lines(primary, ["Security note", "SSH", "UFW", "password", "public"]), security]),
            important_rules=[
                "Do not expose admin panels publicly.",
                "Do not disable SSH password login until key login is confirmed working.",
                "Do not delete Docker volumes unless explicitly confirmed.",
            ],
        )
    add(
            "homelab/commands.md",
            "Homelab Commands",
            "Reusable diagnostic and maintenance commands.",
            _commands_to_code_blocks(commands),
        )
    add(
            "services/immich.md",
            "Immich",
            "Immich deployment, update lessons, containers and risks.",
            immich,
            important_rules=["Do not run docker compose down -v for Immich.", "Back up the database before updates."],
        )
    add(
            "services/nginx-proxy-manager.md",
            "Nginx Proxy Manager",
            "NPM container, internal hostnames and admin exposure rules.",
            npm,
            important_rules=["Port 81 admin should stay LAN/Tailscale only."],
        )
    add(
            "services/grafana-monitoring.md",
            "Grafana And Monitoring",
            "Grafana, Prometheus, node-exporter and cAdvisor notes.",
            "\n\n".join([grafana, monitoring]),
            important_rules=["Do not expose Prometheus, node-exporter or cAdvisor publicly."],
        )
    add(
            "services/adguard.md",
            "AdGuard Home",
            "AdGuard container, config path, ports and DNS exposure rules.",
            adguard,
            important_rules=["Avoid exposing DNS publicly unless intentionally configured."],
        )
    add("services/plex.md", "Plex", "Plex media service notes.", plex)
    add("services/paperless.md", "Paperless NGX", "Paperless NGX and companion services.", paperless)
    add(
            "security/ssh-hardening.md",
            "SSH Hardening",
            "Desired SSH security settings and current lockout warning.",
            _filter_lines(security + "\n" + primary, ["SSH", "PasswordAuthentication", "PermitRootLogin", "PubkeyAuthentication", "key"]),
            important_rules=["Do not disable password login until key-based login works."],
        )
    add(
            "security/firewall-rules.md",
            "Firewall Rules",
            "UFW and network exposure preferences.",
            _filter_lines(security + "\n" + primary, ["UFW", "port", "LAN", "tailscale", "public", "Anywhere"]),
        )
    add(
            "security/secrets-policy.md",
            "Secrets Policy",
            "Rules for secrets, env files, GitHub and model context.",
            _filter_lines(security + "\n" + dmd_agent, ["secret", "token", "API key", ".env", "GitHub", "commit"]),
            important_rules=["Never write secrets, tokens, API keys, private keys or passwords into memory."],
        )
    add(
            "security/incident-response.md",
            "Incident Response",
            "Emergency operating mode, failover and incident response rules.",
            _filter_lines(source, ["incident", "Emergency", "failover", "down", "burned", "recovery", "verify", "split-brain"]),
            important_rules=["During incidents, recovery comes before explanation.", "Group commands by machine."],
        )
    add(
            "github/open-source-workflow.md",
            "Open Source Workflow",
            "DMD Agent open-source contribution workflow.",
            github,
        )
    add(
            "github/branch-strategy.md",
            "Branch Strategy",
            "Recommended branch model for DMD Agent 4 All.",
            _filter_lines(github + "\n" + dmd_agent, ["branch", "main", "dev", "feature", "owner-dev"]),
        )
    add(
            "github/repo-protection.md",
            "Repository Protection",
            "Rulesets, protected branches, CI and CODEOWNERS.",
            _filter_lines(github + "\n" + dmd_agent, ["protected", "ruleset", "CI", "CODEOWNERS", "SECURITY", "direct write"]),
        )
    add(
            "github/contribution-model.md",
            "Contribution Model",
            "Contributor expectations and community files.",
            _filter_lines(github + "\n" + dmd_agent, ["contribution", "pull request", "fork", "CODE_OF_CONDUCT", "issue", "PR"]),
        )
    add(
            "assistant-behavior/response-style.md",
            "Response Style",
            "How the assistant should communicate with the owner.",
            assistant,
        )
    add(
            "assistant-behavior/safety-rules.md",
            "Assistant Safety Rules",
            "Operational safety rules for DevOps, commands and destructive actions.",
            _filter_lines(assistant + "\n" + security, ["warn", "destructive", "down -v", "rm -rf", "force push", "DB", "volume"]),
        )
    add(
            "assistant-behavior/emergency-mode.md",
            "Emergency Mode",
            "How to respond during live incidents and primary product failover events.",
            _filter_lines(source, ["Emergency response style", "senior SRE", "failover", "downtime", "split-brain", "recovery"]),
            important_rules=["Exact recovery commands only during incidents.", "Verify DB role before promotion."],
        )
    files["README.md"] = _memory_index(files, today=today)
    return files


def _markdown_sections(source: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "Root"
    sections[current] = []
    for line in _normalize_inline_markdown_headings(source).splitlines():
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if match:
            current = match.group(1).strip().strip("# ")
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _normalize_inline_markdown_headings(source: str) -> str:
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+---\s+", "\n---\n", text)
    heading_titles = [
        r"Long-Term Memory[^\n#]*",
        r"Long-term Memory[^\n#]*",
        r"Owner / User",
        r"Personal / Motivation",
        r"Main Business / Brand",
        r"Business / Brand",
        r"Project: Primary Product",
        r"Project: Product",
        r"Primary Product Strategy",
        r"Product Strategy",
        r"Primary Product Stack",
        r"Product Stack",
        r"Primary Product Features",
        r"Product Features",
        r"Primary Product Deployment Pattern",
        r"Product Deployment Pattern",
        r"Primary Product High Availability / Failover Architecture",
        r"Product High Availability / Failover Architecture",
        r"Project: DMD Agent 4 All",
        r"Homelab Infrastructure",
        r"Primary Server: homelab",
        r"Backup Server: homelab2",
        r"Older / Alternative Infrastructure Notes",
        r"Immich",
        r"Other Homelab Services",
        r"Nginx Proxy Manager",
        r"Homepage",
        r"Grafana",
        r"Monitoring Services",
        r"AdGuard Home",
        r"Plex",
        r"Paperless NGX",
        r"Security Preferences / Rules",
        r"GitHub / Open Source Preferences",
        r"Preferred Assistant Behavior",
        r"Important Commands / Patterns",
        r"Check Docker containers",
        r"Check Docker disk usage",
        r"Safe Docker cleanup",
        r"Check UFW",
        r"Check SSH effective config",
        r"Check active SSH sessions",
        r"Check failed services",
        r"Check logs",
        r"Check zombie processes",
        r"Check Tailscale",
        r"Current Priorities",
    ]
    pattern = re.compile(r"#{1,3}\s+(?:" + "|".join(heading_titles) + r")", flags=re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        heading = re.sub(r"\s+", " ", match.group(0).strip())
        return f"\n{heading}\n"

    return pattern.sub(replace, text).strip()


def _memory_doc(
    title: str,
    purpose: str,
    today: str,
    content: str,
    *,
    important_rules: list[str] | None = None,
) -> str:
    body = _normalize_memory_notes(content)
    lines = [
        f"# {title}",
        "",
        f"Purpose: {purpose}",
        f"Last updated: {today}",
        "",
    ]
    if important_rules:
        lines.extend(["## Important Rules", "", *[f"- {rule}" for rule in important_rules], ""])
    lines.extend(["## Notes", "", body.rstrip()])
    return "\n".join(lines).rstrip() + "\n"


def _has_real_memory_content(content: str) -> bool:
    return bool(_normalize_memory_notes(content).strip())


def _normalize_memory_notes(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    text = re.sub(r"\s+\d+\.\s+", lambda match: f"\n{match.group(0).strip()} ", text)
    text = re.sub(r"\s+-\s+", "\n- ", text)
    text = re.sub(
        r"\s+(Name:|Location:|Role:|Core strategy:|90-day execution priorities:|Early traction / contacts:|Backend:|Frontend:|Infrastructure:|Important paths:|Important containers:|Database:|API:|Cloudflare:|Primary server:|Backup server:|Replication:|Code/uploads sync:|Backup API:|Failover:|Failover order:|Emergency response style:|Goal:|Positioning:|Current practical features:|Useful real scenarios:|Security philosophy:|Desired open-source model:|Recommended branch model:|Recommended repository files:|Recommended license:|Potential launch message:|Known system:|Important services / containers on homelab:|Important ports observed:|Security notes:|Normal state:|Known files/directories:|Immich containers:|Immich port:|Current updated version observed:|Important Immich lesson:|Known issue:|Services:|General rules:|SSH hardening desired:|Correct model:|Recommended launch plan:)\s+",
        r"\n\1\n",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _filter_lines(text: str, keywords: list[str]) -> str:
    lowered_keywords = [keyword.casefold() for keyword in keywords]
    selected: list[str] = []
    active_heading = ""
    for line in _normalize_memory_notes(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            active_heading = stripped
        haystack = f"{active_heading}\n{line}".casefold()
        if any(keyword in haystack for keyword in lowered_keywords):
            selected.append(line)
    return "\n".join(selected).strip()


def _commands_to_code_blocks(text: str) -> str:
    lines: list[str] = []
    in_block = False
    for raw_line in _normalize_memory_notes(text).splitlines():
        stripped = raw_line.strip()
        looks_command = bool(
            re.match(r"^(cd|git|docker|sudo|ssh|curl|systemctl|journalctl|tailscale|ps|who|w|ss|make)\b", stripped)
            or stripped.startswith("/")
            or "SELECT " in stripped
        )
        if looks_command and not in_block:
            lines.append("```bash")
            in_block = True
        if not looks_command and in_block:
            lines.append("```")
            in_block = False
        lines.append(raw_line)
    if in_block:
        lines.append("```")
    return "\n".join(lines).strip()


def _memory_index(files: dict[str, str], *, today: str) -> str:
    high_priority = {
        "owner/profile.md",
        "owner/preferences.md",
        "projects/primary-product/failover.md",
        "projects/primary-product/infrastructure.md",
        "projects/dmd-agent-4-all/security-model.md",
        "homelab/security.md",
        "security/incident-response.md",
        "assistant-behavior/emergency-mode.md",
    }
    lines = [
        "# Memory Index",
        "",
        "Purpose: Index for modular long-term memory files.",
        f"Last updated: {today}",
        "",
        "## High Priority For Retrieval",
        "",
        *[f"- {path}" for path in sorted(path for path in high_priority if path in files)],
        "",
        "## Files",
        "",
    ]
    for path, body in sorted(files.items()):
        if path == "README.md":
            continue
        lines.append(f"- `{path}` — {_first_content_sentence(body)}")
    return "\n".join(lines).rstrip() + "\n"


def _tree_from_paths(paths: list[str]) -> str:
    root: dict[str, Any] = {}
    for path in sorted(paths):
        node = root
        for part in path.split("/"):
            node = node.setdefault(part, {})

    def render(node: dict[str, Any], prefix: str = "") -> list[str]:
        lines: list[str] = []
        items = sorted(node.items())
        for index, (name, child) in enumerate(items):
            connector = "└── " if index == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if child:
                extension = "    " if index == len(items) - 1 else "│   "
                lines.extend(render(child, prefix + extension))
        return lines

    return "\n".join(["memory", *render(root)])


def _first_content_sentence(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("Purpose:"):
            return stripped.removeprefix("Purpose:").strip()
    for line in body.splitlines():
        stripped = line.strip().strip("# ")
        if stripped:
            return stripped[:160]
    return "No summary available."


def _memory_ambiguities(source: str) -> list[str]:
    ambiguities: list[str] = []
    if "Older / Alternative Infrastructure Notes" in source:
        ambiguities.append(
            "Older / Alternative Infrastructure Notes were preserved separately; prefer the newer homelab primary/standby architecture unless the owner says they reverted."
        )
    if "SSH password login was still active" in source and "Do not disable SSH password login" in source:
        ambiguities.append(
            "SSH hardening is desired, but password login must not be disabled until key-based login is confirmed working."
        )
    if "cloudflared should be active on primary" in source and "Backup cloudflared should remain stopped" in source:
        ambiguities.append(
            "Cloudflared should normally run only on the active primary; backup tunnel is for failover."
        )
    return ambiguities


def _terminal_run(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    raw_command = args.get("command")
    if not isinstance(raw_command, list):
        raise ValueError("terminal.run requires command as a string array")
    command = [str(part) for part in raw_command]
    cwd = args.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("terminal.run cwd must be a string when provided")

    result = run_workspace_command(
        command,
        workspace=_terminal_workspace_root(context),
        policy=TerminalPolicy.from_config(context.config),
        cwd=cwd,
    )
    return {
        **result,
        "stdout": redact_text(str(result.get("stdout", ""))),
        "stderr": redact_text(str(result.get("stderr", ""))),
    }


def _terminal_workspace_root(context: ToolRuntimeContext) -> Path:
    manager = WorkspaceManager.from_config(context.config, fallback_workspace=context.workspace_root)
    return manager.validate_workspace(manager.current_workspace)


def _browser_not_implemented(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del context
    url = args.get("url")
    if isinstance(url, str) and url:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("browser tools require a valid http or https URL")
        url = parsed.geturl()
    return {
        "status": "not_implemented",
        "message": (
            "Browser sandbox execution is not implemented yet. "
            "The request passed tool and permission policy, but no browser session was started."
        ),
        "url": url,
    }


def _gmail_not_configured(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del args, context
    return {
        "status": "connector_not_configured",
        "connector": "gmail",
        "message": (
            "Gmail OAuth is not configured in this build. Enablement and permissions "
            "are enforced, but real Gmail API access still requires connector setup."
        ),
    }


def _calendar_create_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    context.workspace_root.mkdir(parents=True, exist_ok=True)
    title = (
        args.get("title")
        or args.get("summary")
        or args.get("name")
        or args.get("description")
        or "Untitled event"
    )
    event = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "start": args.get("start") or args.get("start_time") or args.get("datetime"),
        "end": args.get("end") or args.get("end_time"),
        "reminder": args.get("reminder") or args.get("remind_at") or args.get("reminder_offset"),
        "raw_args": args,
    }
    calendar_file = context.workspace_root / "local_calendar_events.jsonl"
    with calendar_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "status": "saved_local",
        "message": (
            "Saved to the local calendar store. External calendar sync and active "
            "notification delivery are not implemented yet."
        ),
        "event": event,
        "path": str(calendar_file),
    }


def _calendar_update_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    event_id = str(args.get("id") or args.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("calendar.update_event requires id.")
    events = _read_local_calendar_events(context)
    updated_event: dict[str, Any] | None = None
    for event in events:
        if str(event.get("id") or "") != event_id:
            continue
        for source_key, target_key in {
            "title": "title",
            "summary": "title",
            "start": "start",
            "start_time": "start",
            "end": "end",
            "end_time": "end",
            "reminder": "reminder",
            "remind_at": "reminder",
        }.items():
            if source_key in args:
                event[target_key] = args[source_key]
        event["updated_at"] = datetime.now(timezone.utc).isoformat()
        event["raw_update_args"] = args
        updated_event = event
        break
    if updated_event is None:
        raise ValueError(f"Calendar event not found: {event_id}")
    _write_local_calendar_events(context, events)
    return {
        "status": "updated_local",
        "message": "Updated the local calendar store. External calendar sync is not implemented yet.",
        "event": updated_event,
        "path": str(_calendar_store_path(context)),
    }


def _calendar_delete_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    event_id = str(args.get("id") or args.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("calendar.delete_event requires id.")
    events = _read_local_calendar_events(context)
    remaining = [event for event in events if str(event.get("id") or "") != event_id]
    if len(remaining) == len(events):
        raise ValueError(f"Calendar event not found: {event_id}")
    _write_local_calendar_events(context, remaining)
    return {
        "status": "deleted_local",
        "message": "Deleted the event from the local calendar store. External calendar sync is not implemented yet.",
        "event_id": event_id,
        "path": str(_calendar_store_path(context)),
    }


def _calendar_find_free_slots(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    start = _parse_datetime_arg(args.get("start")) or datetime.now().astimezone()
    end = _parse_datetime_arg(args.get("end")) or (start + timedelta(days=7))
    if end <= start:
        raise ValueError("calendar.find_free_slots end must be after start.")
    duration_minutes = _bounded_int(args.get("duration_minutes"), default=30, minimum=5, maximum=480)
    limit = _bounded_int(args.get("limit"), default=20, minimum=1, maximum=100)
    workday_start = _parse_time_arg(args.get("workday_start"), default=time(9, 0))
    workday_end = _parse_time_arg(args.get("workday_end"), default=time(17, 0))
    if workday_end <= workday_start:
        raise ValueError("workday_end must be after workday_start.")

    busy_ranges = sorted(
        range_
        for range_ in (_event_datetime_range(event) for event in _read_local_calendar_events(context))
        if range_ is not None and range_[1] > start and range_[0] < end
    )
    slots: list[dict[str, str]] = []
    current_day = start.date()
    while len(slots) < limit and current_day <= end.date():
        window_start = datetime.combine(current_day, workday_start, tzinfo=start.tzinfo)
        window_end = datetime.combine(current_day, workday_end, tzinfo=start.tzinfo)
        cursor = max(window_start, start)
        day_end = min(window_end, end)
        if cursor < day_end:
            for busy_start, busy_end in busy_ranges:
                if busy_end <= cursor or busy_start >= day_end:
                    continue
                if _minutes_between(cursor, busy_start) >= duration_minutes:
                    slots.append({"start": cursor.isoformat(), "end": busy_start.isoformat()})
                    if len(slots) >= limit:
                        break
                cursor = max(cursor, busy_end)
            if len(slots) >= limit:
                break
            if _minutes_between(cursor, day_end) >= duration_minutes:
                slots.append({"start": cursor.isoformat(), "end": day_end.isoformat()})
        current_day += timedelta(days=1)

    return {
        "slots": slots[:limit],
        "source": "local_calendar_store",
        "duration_minutes": duration_minutes,
    }


def _calendar_today(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    now = datetime.now().astimezone()
    events = [
        event
        for event in _read_local_calendar_events(context)
        if _event_start_date(event) == now.date()
    ]
    return {"events": events, "source": "local_calendar_store"}


def _calendar_week(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    now = datetime.now().astimezone()
    today = now.date()
    events = []
    for event in _read_local_calendar_events(context):
        start_date = _event_start_date(event)
        if start_date is not None and 0 <= (start_date - today).days <= 6:
            events.append(event)
    return {"events": events, "source": "local_calendar_store"}


def _read_local_calendar_events(context: ToolRuntimeContext) -> list[dict[str, Any]]:
    calendar_file = _calendar_store_path(context)
    if not calendar_file.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in calendar_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            events.append(loaded)
    return events


def _write_local_calendar_events(context: ToolRuntimeContext, events: list[dict[str, Any]]) -> None:
    calendar_file = _calendar_store_path(context)
    calendar_file.parent.mkdir(parents=True, exist_ok=True)
    with calendar_file.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _calendar_store_path(context: ToolRuntimeContext):
    context.workspace_root.mkdir(parents=True, exist_ok=True)
    return context.workspace_root / "local_calendar_events.jsonl"


def _event_start_date(event: dict[str, Any]):
    raw = event.get("start")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone().date()
    except ValueError:
        return None


def _event_datetime_range(event: dict[str, Any]):
    start = _parse_datetime_arg(event.get("start"))
    if start is None:
        return None
    end = _parse_datetime_arg(event.get("end")) or (start + timedelta(hours=1))
    if end <= start:
        end = start + timedelta(hours=1)
    return start, end


def _parse_datetime_arg(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone()


def _parse_time_arg(value: Any, *, default: time) -> time:
    if value is None or str(value).strip() == "":
        return default
    raw = str(value).strip()
    try:
        parsed = time.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {raw}") from exc
    return parsed.replace(tzinfo=None)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Value must be between {minimum} and {maximum}.")
    return parsed


def _minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60
