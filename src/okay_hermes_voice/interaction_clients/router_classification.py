"""Provider resolution and router classification entrypoint."""
from __future__ import annotations

import importlib

from ..interaction_types import InteractionRouterConfig, RouterDecision
from .router_call import classify_with_client


def classify_request(transcript: str, cfg: InteractionRouterConfig) -> RouterDecision:
    if not cfg.router_enabled:
        return RouterDecision(brief_reason="router_disabled")
    try:
        module = importlib.import_module("agent.auxiliary_client")
        resolve_provider_client = module.resolve_provider_client
        client, model = resolve_provider_client(cfg.router_provider, model=cfg.router_model)
    except Exception as exc:
        return RouterDecision(brief_reason=f"provider_resolution_failed:{type(exc).__name__}")
    if client is None or not model:
        return RouterDecision(brief_reason="provider_resolution_returned_none")
    return classify_with_client(client, model, transcript, cfg)


__all__ = ["classify_request"]
