from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dmdcore.agent import AgentCore
from dmdcore.agent.planner import LLMPlanner
from dmdcore.app_paths import AppPaths
from dmdcore.audit import AuditStore
from dmdcore.config import load_config, write_default_config
from dmdcore.llm.cloud_providers import CloudProviderStub
from dmdcore.llm.ollama_provider import OllamaProvider
from dmdcore.llm.openai_compatible_provider import OpenAICompatibleProvider
from dmdcore.permissions import PermissionContext, PermissionEngine
from dmdcore.tools import build_builtin_registry
from dmdcore.tools.base import ToolRuntimeContext


def build_agent_core(
    *,
    planner_enabled: bool = True,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> AgentCore:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    config = load_config(paths.config)
    registry = build_builtin_registry()
    provider = _provider_from_config(config) if planner_enabled else None
    return AgentCore(
        permission_engine=PermissionEngine(registry.manifests),
        tool_registry=registry,
        permission_context=permission_context_from_config(config),
        runtime_context=ToolRuntimeContext(
            memory_root=paths.memory,
            workspace_root=paths.workspace,
            config=config,
            config_path=paths.config,
        ),
        audit_store=AuditStore(paths.audit_db),
        planner=LLMPlanner(provider) if provider is not None else None,
        event_sink=event_sink,
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
    if provider in {"openai", "deepseek", "openai-compatible", "openrouter", "lmstudio", "vllm"}:
        return OpenAICompatibleProvider(
            model=str(planner_model),
            base_url=_openai_compatible_base_url(provider, llm),
            api_key_env=_openai_compatible_api_key_env(provider, llm),
            provider_name=provider,
        )
    return CloudProviderStub(provider_name=provider, model=str(planner_model))


def _openai_compatible_base_url(provider: str, llm: dict[str, Any]) -> str:
    configured = llm.get("base_url")
    if configured:
        return str(configured)
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "lmstudio":
        return "http://localhost:1234/v1"
    if provider == "vllm":
        return "http://localhost:8000/v1"
    return "https://api.openai.com/v1"


def _openai_compatible_api_key_env(provider: str, llm: dict[str, Any]) -> str | None:
    configured = llm.get("api_key_env")
    if configured:
        return str(configured)
    if provider == "openai":
        return "DMDCORE_OPENAI_API_KEY"
    if provider == "deepseek":
        return "DMDCORE_DEEPSEEK_API_KEY"
    if provider == "openrouter":
        return "DMDCORE_OPENROUTER_API_KEY"
    if provider in {"lmstudio", "vllm"}:
        return None
    return "DMDCORE_OPENAI_COMPATIBLE_API_KEY"
