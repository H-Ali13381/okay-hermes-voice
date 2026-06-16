"""Warmup helpers for the provider-backed interaction router."""
from __future__ import annotations

from ..interaction_types import InteractionRouterConfig
from .router_client_cache import router_client_cache


def clear_prewarmed_router() -> None:
    """Clear cached router client state, primarily for tests and config reloads."""
    router_client_cache.clear()


def prewarm_interaction_router(cfg: InteractionRouterConfig) -> bool:
    """Resolve and cache the router provider client before the first voice turn."""
    return router_client_cache.prewarm(cfg)


__all__ = ["clear_prewarmed_router", "prewarm_interaction_router"]
