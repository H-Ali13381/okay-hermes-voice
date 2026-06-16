"""One router model call through an already resolved client."""
from __future__ import annotations

from typing import Any

from ..interaction_types import InteractionRouterConfig, RouterDecision
from .router_messages import build_router_messages


def classify_with_client(
    client: Any,
    model: str,
    transcript: str,
    cfg: InteractionRouterConfig,
) -> RouterDecision:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_router_messages(transcript),
        "temperature": 0,
        "max_tokens": 180,
        "response_format": {"type": "json_object"},
        "timeout": cfg.router_timeout_seconds,
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError as exc:
        if "timeout" not in str(exc):
            return RouterDecision(brief_reason=f"router_call_failed:{type(exc).__name__}")
        kwargs.pop("timeout", None)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as retry_exc:
            return RouterDecision(brief_reason=f"router_call_failed:{type(retry_exc).__name__}")
    except Exception as exc:
        return RouterDecision(brief_reason=f"router_call_failed:{type(exc).__name__}")

    try:
        raw = response.choices[0].message.content
    except Exception as exc:
        return RouterDecision(brief_reason=f"router_response_invalid:{type(exc).__name__}")
    return RouterDecision.parse(raw)


__all__ = ["classify_with_client"]
