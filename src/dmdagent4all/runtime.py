from __future__ import annotations

from typing import Any

from dmdagent4all.agent import AgentCore
from dmdagent4all.agent.planner import LLMPlanner
from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditStore
from dmdagent4all.config import load_config, write_default_config
from dmdagent4all.llm.cloud_providers import CloudProviderStub
from dmdagent4all.llm.ollama_provider import OllamaProvider
from dmdagent4all.permissions import PermissionContext, PermissionEngine
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.base import ToolRuntimeContext


def build_agent_core() -> AgentCore:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    config = load_config(paths.config)
    registry = build_builtin_registry()
    provider = _provider_from_config(config)
    return AgentCore(
        permission_engine=PermissionEngine(registry.manifests),
        tool_registry=registry,
        permission_context=permission_context_from_config(config),
        runtime_context=ToolRuntimeContext(
            memory_root=paths.memory,
            workspace_root=paths.workspace,
            config=config,
        ),
        audit_store=AuditStore(paths.audit_db),
        planner=LLMPlanner(provider),
    )


def permission_context_from_config(config: dict[str, Any]) -> PermissionContext:
    permissions = config.get("permissions", {})
    tool_overrides = config.get("tools", {})
    enabled_tools = {
        name
        for name, value in tool_overrides.items()
        if isinstance(value, dict) and value.get("enabled") is True
    }
    disabled_tools = {
        name
        for name, value in tool_overrides.items()
        if isinstance(value, dict) and value.get("enabled") is False
    }
    return PermissionContext(
        granted_permissions=frozenset(permissions.get("granted", [])),
        enabled_tools=frozenset(enabled_tools),
        disabled_tools=frozenset(disabled_tools),
        cloud_model_active=config.get("llm", {}).get("provider") not in {"ollama", "local"},
        approval_risk_threshold=int(permissions.get("approval_required_at_risk", 3)),
    )


def _provider_from_config(config: dict[str, Any]):
    llm = config.get("llm", {})
    provider = str(llm.get("provider", "ollama")).lower()
    model = str(llm.get("model", "qwen3:8b"))
    planner_model = llm.get("planner_model") or model
    if provider in {"ollama", "local"}:
        return OllamaProvider(
            model=str(planner_model),
            base_url=str(llm.get("base_url", "http://localhost:11434")),
        )
    return CloudProviderStub(provider_name=provider, model=str(planner_model))
