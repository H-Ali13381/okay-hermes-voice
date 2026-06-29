"""Fast small-model answering for simple safe voice requests."""
from __future__ import annotations

import importlib
from typing import Optional

from ..interaction_types import InteractionRouterConfig
from .small_model_messages import build_small_model_messages


def answer_with_small_model(transcript: str, cfg: InteractionRouterConfig) -> Optional[str]:
    try:
        module = importlib.import_module("agent.auxiliary_client")
        resolve_provider_client = module.resolve_provider_client
        client, model = resolve_provider_client(cfg.small_model_provider, model=cfg.small_model_model)
    except Exception:
        return None
    if client is None or not model:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_small_model_messages(transcript),
            temperature=0.2,
            max_tokens=300,
            timeout=cfg.small_model_timeout_seconds,
        )
    except TypeError as exc:
        if "timeout" not in str(exc):
            return None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_small_model_messages(transcript),
                temperature=0.2,
                max_tokens=300,
            )
        except Exception:
            return None
    except Exception:
        return None
    try:
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        return None


__all__ = ["answer_with_small_model"]
