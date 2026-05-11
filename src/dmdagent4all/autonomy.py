from __future__ import annotations

import logging
import os
from typing import Any


LOGGER = logging.getLogger("dmdagent4all.local_dev_autonomy")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FULL_MODE_VALUES = {"full", "full_llm_first", "full-autonomy", "autonomy", "local_dev"}


def local_dev_autonomy_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return the effective autonomy-mode state for this local runtime.

    The environment variable remains the strongest local override. The UI writes
    the same mode into runtime.autonomy so a running local server can be switched
    without a restart. This function only reports orchestration mode; backend
    security policy remains the source of truth for each tool execution.
    """
    if not isinstance(config, dict):
        return os.environ.get("LOCAL_DEV_AUTONOMY", "").strip().casefold() in _TRUE_VALUES
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    autonomy = runtime.get("autonomy", {})
    if not isinstance(autonomy, dict):
        autonomy = config.get("autonomy", {})
    if not isinstance(autonomy, dict):
        return False
    if isinstance(autonomy.get("enabled"), bool):
        return bool(autonomy.get("enabled"))
    if os.environ.get("LOCAL_DEV_AUTONOMY", "").strip().casefold() in _TRUE_VALUES:
        return True
    return str(autonomy.get("mode") or "").strip().casefold() in _FULL_MODE_VALUES


def log_local_dev_autonomy(event: str, config: dict[str, Any] | None = None, **metadata: Any) -> None:
    if not local_dev_autonomy_enabled(config):
        return
    safe_metadata = {str(key): _compact_value(value) for key, value in metadata.items()}
    LOGGER.info("LOCAL_DEV_AUTONOMY %s %s", event, safe_metadata)


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "...[truncated]"
    if isinstance(value, list | tuple):
        return [_compact_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item)
            for key, item in list(value.items())[:20]
        }
    return value
