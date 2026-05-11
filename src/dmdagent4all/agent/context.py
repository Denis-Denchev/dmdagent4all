from __future__ import annotations

from pathlib import Path
from typing import Any

from dmdagent4all.audit import AuditStore
from dmdagent4all.agent.runtime_state import runtime_state_snapshot
from dmdagent4all.autonomy import local_dev_autonomy_enabled
from dmdagent4all.memory import MemoryManager
from dmdagent4all.permissions import PermissionContext, ToolManifest
from dmdagent4all.sandbox import TerminalPolicy
from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.storage import downloads_root_from_config
from dmdagent4all.workspace import WorkspaceManager


_LOCAL_DEV_AUTONOMY_ENABLED_TOOLS = {
    "browser.extract_text",
    "browser.open",
    "browser.scrape_markdown",
    "developer.context",
    "files.list",
    "files.mkdir",
    "files.read",
    "files.write",
    "files.write_many",
    "memory.list",
    "memory.read",
    "project.scaffold_one_page_app",
    "system.list_enabled_tools",
    "terminal.run",
    "workspace.status",
}


class AgentContextProvider:
    def __init__(
        self,
        *,
        runtime_context: ToolRuntimeContext,
        manifests: dict[str, ToolManifest],
        permission_context: PermissionContext,
        audit_store: AuditStore,
    ) -> None:
        self.runtime_context = runtime_context
        self.manifests = manifests
        self.permission_context = permission_context
        self.audit_store = audit_store

    def snapshot(
        self,
        *,
        planner_active: bool,
        last_denial_reason: str = "",
        last_tool_result_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.runtime_context.config
        manager = WorkspaceManager.from_config(
            config,
            fallback_workspace=self.runtime_context.workspace_root,
        )
        llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
        setup = config.get("setup", {}) if isinstance(config.get("setup"), dict) else {}
        tools = _tool_registry_summary(self.manifests, config, self.permission_context)
        memory_files = _memory_index(self.runtime_context.memory_root)
        autonomy_active = local_dev_autonomy_enabled(config)
        downloads_root = downloads_root_from_config(config, default_root=self.runtime_context.workspace_root)
        runtime_state = runtime_state_snapshot(
            runtime_context=self.runtime_context,
            manager=manager,
            downloads_root=downloads_root,
            memory_files=memory_files,
            enabled_tools=list(tools["enabled"]),
            disabled_tools=list(tools["disabled"]),
            planner_active=planner_active,
        )
        return {
            "agent_name": str(setup.get("agent_name") or "DMD Agent"),
            "autonomy": {
                "local_dev_autonomy": autonomy_active,
                "enabled": autonomy_active,
                "mode": "LOCAL_DEV_AUTONOMY" if autonomy_active else "default",
                "safe_auto_approval_scope": (
                    [
                        "files.list",
                        "files.mkdir",
                        "files.read",
                        "files.write",
                        "files.write_many",
                        "browser.open",
                        "browser.extract_text",
                        "browser.scrape_markdown",
                        "memory.list",
                        "memory.read",
                        "project.scaffold_one_page_app",
                        "workspace.status",
                        "system.list_enabled_tools",
                        "terminal.run (readonly commands only)",
                        "terminal.run (validated workspace scripts)",
                    ]
                    if autonomy_active
                    else []
                ),
                "hard_safety_source_of_truth": "backend",
            },
            "runtime": {
                "provider": str(llm.get("provider") or ""),
                "model": str(llm.get("model") or ""),
                "planner_model": str(llm.get("planner_model") or llm.get("model") or ""),
                "planner_active": bool(planner_active),
                "response_language": str(llm.get("response_language") or "auto"),
            },
            "runtime_state": runtime_state,
            "workspace": {
                "current": str(manager.current_workspace),
                "default": str(manager.default_workspace),
                "allowed_roots": [str(root) for root in manager.allowed_roots],
                "blocked_paths": [str(path) for path in manager.blocked_paths],
                "mounted_roots": _mounted_roots(config, manager, self.runtime_context.workspace_root),
                "downloads_path": str(downloads_root),
            },
            "memory": {
                "root": str(self.runtime_context.memory_root.resolve()),
                "locations": [str(self.runtime_context.memory_root.resolve())],
                "files": memory_files,
                "file_count": len(memory_files),
            },
            "tools": tools,
            "permissions": {
                "granted": sorted(self.permission_context.granted_permissions),
                "approval_risk_threshold": self.permission_context.approval_risk_threshold,
                "cloud_model_active": self.permission_context.cloud_model_active,
                "cloud_context_approved": self.permission_context.cloud_context_approved,
            },
            "terminal": _terminal_summary(config),
            "config": {
                "path": str(self.runtime_context.config_path or ""),
                "sections": _config_section_summary(config),
                "locations": [str(self.runtime_context.config_path)] if self.runtime_context.config_path else [],
            },
            "config_sections": _config_section_summary(config),
            "email": _email_summary(config, tools["enabled"]),
            "approvals": {
                "pending": _pending_approvals(self.audit_store),
            },
            "recent_activity": {
                "last_denial_reason": redact_text(last_denial_reason) if last_denial_reason else "",
                "last_tool_result": last_tool_result_summary or {},
            },
            "security_invariants": [
                "Backend safety remains source of truth for workspace/path validation.",
                "Backend blocks .env and secret/system paths.",
                "Backend blocks destructive SQL.",
                "Backend requires approval for configured risky actions, file delete, external send, and emergency-stop state.",
            ],
        }


def _tool_registry_summary(
    manifests: dict[str, ToolManifest],
    config: dict[str, Any],
    permission_context: PermissionContext,
) -> dict[str, Any]:
    overrides = config.get("tools", {})
    if not isinstance(overrides, dict):
        overrides = {}
    tools: list[dict[str, Any]] = []
    enabled: list[str] = []
    disabled: list[str] = []
    for name, manifest in sorted(manifests.items()):
        override = overrides.get(name, {})
        if not isinstance(override, dict):
            override = {}
        is_enabled = (
            manifest.default_enabled
            or manifest.name in permission_context.enabled_tools
            or bool(override.get("enabled", False))
        )
        if manifest.name in permission_context.disabled_tools:
            is_enabled = False
        if override.get("enabled") is False:
            is_enabled = False
        if local_dev_autonomy_enabled(config) and manifest.name in _LOCAL_DEV_AUTONOMY_ENABLED_TOOLS:
            is_enabled = True
        if is_enabled:
            enabled.append(name)
        else:
            disabled.append(name)
        tools.append(
            {
                "name": name,
                "description": manifest.description,
                "risk": int(manifest.risk),
                "enabled": is_enabled,
                "approval_required": manifest.approval_required,
                "permissions": list(manifest.permissions),
                "args": manifest.argument_schema,
            }
        )
    return {
        "count": len(tools),
        "enabled": enabled,
        "disabled": disabled,
        "registry": tools,
    }


def _memory_index(memory_root: Path, *, limit: int = 120) -> list[str]:
    manager = MemoryManager(memory_root)
    try:
        manager.bootstrap()
        return manager.list_files()[:limit]
    except (OSError, ValueError):
        return []


def _mounted_roots(
    config: dict[str, Any],
    manager: WorkspaceManager,
    workspace_root: Path,
) -> list[dict[str, Any]]:
    roots = [
        {
            "label": "workspace",
            "path": str(manager.current_workspace),
            "allowed": _is_inside_allowed_roots(manager.current_workspace, manager),
        }
    ]
    downloads_root = downloads_root_from_config(config, default_root=workspace_root)
    roots.append(
        {
            "label": "downloads",
            "path": str(downloads_root),
            "allowed": _is_inside_allowed_roots(downloads_root, manager),
            "custom": bool(str(config.get("storage", {}).get("downloads_root") or "").strip())
            if isinstance(config.get("storage"), dict)
            else False,
        }
    )
    workspace_config = config.get("workspace", {})
    configured_mounts = workspace_config.get("mounts") if isinstance(workspace_config, dict) else None
    if isinstance(configured_mounts, list):
        for index, item in enumerate(configured_mounts):
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            try:
                path = manager.resolve_user_path(raw_path)
            except (OSError, ValueError):
                continue
            roots.append(
                {
                    "label": str(item.get("label") or f"mount_{index}"),
                    "path": str(path),
                    "allowed": _is_inside_allowed_roots(path, manager),
                }
            )
    return roots


def _is_inside_allowed_roots(path: Path, manager: WorkspaceManager) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in manager.allowed_roots)


def _terminal_summary(config: dict[str, Any]) -> dict[str, Any]:
    terminal = config.get("terminal", {})
    if not isinstance(terminal, dict):
        terminal = {}
    try:
        policy = TerminalPolicy.from_config(config)
        allowed_commands = [" ".join(command) for command in policy.allowed_commands]
        effective_enabled = policy.enabled
    except (TypeError, ValueError):
        allowed_commands = []
        effective_enabled = bool(terminal.get("enabled", False))
    return {
        "enabled": effective_enabled,
        "configured_enabled": bool(terminal.get("enabled", False)),
        "mode": str(terminal.get("mode") or "allowlist"),
        "workspace_only": bool(terminal.get("workspace_only", True)),
        "auto_approve_allowlisted": bool(terminal.get("auto_approve_allowlisted", False)),
        "local_dev_readonly_relaxation": local_dev_autonomy_enabled(config),
        "local_dev_workspace_script_execution": local_dev_autonomy_enabled(config),
        "allowed_commands": allowed_commands[:80],
    }


def _config_section_summary(config: dict[str, Any]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for section, value in sorted(config.items()):
        if isinstance(value, dict):
            summary[str(section)] = sorted(str(key) for key in value.keys())
        else:
            summary[str(section)] = []
    return summary


def _email_summary(config: dict[str, Any], enabled_tools: list[str]) -> dict[str, Any]:
    email = config.get("email", {})
    if not isinstance(email, dict):
        email = {}
    summary: dict[str, Any] = {}
    for provider in ("gmail", "outlook"):
        provider_config = email.get(provider, {})
        if not isinstance(provider_config, dict):
            provider_config = {}
        summary[provider] = {
            "enabled": bool(provider_config.get("enabled", False)),
            "auth_method": str(provider_config.get("auth_method") or ""),
            "tools_enabled": sorted(tool for tool in enabled_tools if tool.startswith(f"{provider}.")),
        }
    return summary


def _pending_approvals(audit_store: AuditStore, *, limit: int = 10) -> list[dict[str, Any]]:
    try:
        approvals = audit_store.list_approvals(status="pending", limit=limit)
    except (OSError, ValueError):
        return []
    return [
        {
            "id": approval.get("id"),
            "tool": approval.get("tool"),
            "risk": approval.get("risk"),
            "reason": redact_text(str(approval.get("reason") or "")),
            "args": _safe_args_summary(approval.get("args") or {}),
        }
        for approval in approvals
    ]


def _safe_args_summary(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    safe_keys = {
        "command",
        "content_type",
        "draft_id",
        "filename",
        "format",
        "limit",
        "mailbox",
        "mode",
        "path",
        "provider",
        "query",
        "recursive",
        "status",
        "subject",
        "thread_id",
        "to",
        "url",
    }
    summarized: dict[str, Any] = {}
    for key, value in args.items():
        if key in safe_keys:
            summarized[key] = _redact_value(value)
        elif key.endswith("_from_previous_step") or key.startswith("path_from_"):
            summarized[key] = value
    return summarized


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)[:500]
    if isinstance(value, list):
        return [_redact_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in list(value.items())[:20]}
    return value
