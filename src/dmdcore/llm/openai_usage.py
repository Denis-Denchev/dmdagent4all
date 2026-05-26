from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dmdcore.app_paths import AppPaths
from dmdcore.config import load_config


DEFAULT_OPENAI_API_KEY_ENV = "DMDCORE_OPENAI_API_KEY"


@dataclass(frozen=True)
class OpenAIPrice:
    input_per_1m: float
    output_per_1m: float


DEFAULT_PRICES: dict[str, OpenAIPrice] = {
    "gpt-4o": OpenAIPrice(input_per_1m=5.0, output_per_1m=15.0),
    "gpt-4o-mini": OpenAIPrice(input_per_1m=0.15, output_per_1m=0.6),
    "gpt-4.1": OpenAIPrice(input_per_1m=2.0, output_per_1m=8.0),
    "gpt-4.1-mini": OpenAIPrice(input_per_1m=0.4, output_per_1m=1.6),
    "gpt-4.1-nano": OpenAIPrice(input_per_1m=0.1, output_per_1m=0.4),
    "o3-mini": OpenAIPrice(input_per_1m=1.1, output_per_1m=4.4),
}

OPENAI_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS openai_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  total_tokens INTEGER NOT NULL,
  estimated_cost_usd REAL NOT NULL
);
"""


def check_openai_budget_available() -> None:
    config = load_config(AppPaths.default().config)
    usage_config = _openai_usage_config(config)
    limit = _as_float(usage_config.get("limit_usd"))
    if limit is None or limit <= 0:
        return
    summary = openai_usage_summary(AppPaths.default().audit_db, config=config)
    if float(summary["estimated_cost_usd"]) >= limit:
        raise RuntimeError(
            "OpenAI local project budget limit reached. Increase the local limit in the dashboard or reset usage."
        )


def record_openai_usage(
    *,
    provider: str,
    model: str,
    usage: dict[str, Any],
    audit_db: Path | None = None,
) -> dict[str, Any]:
    prompt_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens
    cost = estimate_openai_cost(model, prompt_tokens, completion_tokens)
    db = audit_db or AppPaths.default().audit_db
    _ensure_openai_usage_table(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            """
            INSERT INTO openai_usage (
              created_at, provider, model, prompt_tokens, completion_tokens,
              total_tokens, estimated_cost_usd
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
            ),
        )
    return {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }


def openai_usage_summary(
    audit_db: Path | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    db = audit_db or AppPaths.default().audit_db
    _ensure_openai_usage_table(db)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0),
                   COALESCE(SUM(completion_tokens), 0),
                   COALESCE(SUM(total_tokens), 0),
                   COALESCE(SUM(estimated_cost_usd), 0)
            FROM openai_usage
            """
        ).fetchone()
    loaded_config = config or load_config(AppPaths.default().config)
    usage_config = _openai_usage_config(loaded_config)
    limit = _as_float(usage_config.get("limit_usd"))
    spent = float(row[4] or 0.0)
    return {
        "requests": int(row[0] or 0),
        "prompt_tokens": int(row[1] or 0),
        "completion_tokens": int(row[2] or 0),
        "total_tokens": int(row[3] or 0),
        "estimated_cost_usd": spent,
        "limit_usd": limit,
        "remaining_usd": None if limit is None else max(0.0, limit - spent),
        "limit_reached": bool(limit is not None and limit > 0 and spent >= limit),
    }


def reset_openai_usage(audit_db: Path | None = None) -> None:
    db = audit_db or AppPaths.default().audit_db
    _ensure_openai_usage_table(db)
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM openai_usage")


def estimate_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _price_for_model(model)
    if price is None:
        return 0.0
    return round(
        (prompt_tokens / 1_000_000.0) * price.input_per_1m
        + (completion_tokens / 1_000_000.0) * price.output_per_1m,
        8,
    )


def _price_for_model(model: str) -> OpenAIPrice | None:
    normalized = model.lower()
    for key, price in sorted(DEFAULT_PRICES.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized == key or normalized.startswith(f"{key}-"):
            return price
    return None


def _openai_usage_config(config: dict[str, Any]) -> dict[str, Any]:
    usage = config.get("openai_usage", {})
    return usage if isinstance(usage, dict) else {}


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _ensure_openai_usage_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(OPENAI_USAGE_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
