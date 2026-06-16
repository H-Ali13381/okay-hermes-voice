"""Provider resolution and router classification entrypoint."""
from __future__ import annotations

from ..interaction_types import InteractionRouterConfig, RouterDecision
from .router_call import classify_with_client
from .router_client_cache import router_client_cache


def classify_request(transcript: str, cfg: InteractionRouterConfig) -> RouterDecision:
    if not cfg.router_enabled:
        return RouterDecision(brief_reason="router_disabled")
    cached = router_client_cache.get(cfg)
    if cached is not None:
        return classify_with_client(cached.client, cached.resolved_model, transcript, cfg)
    try:
        client, model = router_client_cache.resolve(cfg)
    except Exception as exc:
        return RouterDecision(brief_reason=f"provider_resolution_failed:{type(exc).__name__}")
    if client is None or not model:
        return RouterDecision(brief_reason="provider_resolution_returned_none")
    router_client_cache.store(cfg, client, model)
    return classify_with_client(client, model, transcript, cfg)


__all__ = ["classify_request"]
