"""Optional LangSmith tracing helpers.

This module keeps tracing optional: if LangSmith is unavailable or env vars are not set,
all helpers become no-ops and the agent runs offline without errors.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def langsmith_enabled() -> bool:
    """Return True when env vars indicate LangSmith tracing should be active."""
    return bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover - optional dependency/runtime
    _traceable = None


def traceable(name: str) -> Callable[[F], F]:
    """Return a LangSmith traceable decorator when configured, else a no-op decorator."""

    def decorator(func: F) -> F:
        if _traceable is not None and langsmith_enabled():
            wrapped = _traceable(name=name, run_type="chain")(func)
            return cast(F, wrapped)
        return func

    return decorator


def invocation_config(user_query: str) -> dict[str, Any]:
    """Build LangGraph invoke config with trace metadata when tracing is enabled."""
    if not langsmith_enabled():
        return {}
    return {
        "run_name": "finchat-agent-query",
        "metadata": {"user_query": user_query, "app": "finchat_agent"},
        "tags": ["finchat", "offline-mvp"],
    }
