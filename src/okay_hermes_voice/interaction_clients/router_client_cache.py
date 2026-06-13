"""Process-local cache for resolved interaction-router provider clients."""
from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..interaction_types import InteractionRouterConfig


@dataclass(frozen=True)
class CachedRouterClient:
    """Resolved router provider client keyed by the daemon router config."""

    provider: str
    requested_model: str
    resolved_model: str
    client: Any
    cached_at: float

    def matches(self, cfg: InteractionRouterConfig) -> bool:
        return (
            cfg.router_enabled
            and self.provider == cfg.router_provider
            and self.requested_model == cfg.router_model
        )


class RouterClientCache:
    """Cache the expensive provider-client resolution step for voice routing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: CachedRouterClient | None = None

    def clear(self) -> None:
        with self._lock:
            self._cached = None

    def get(self, cfg: InteractionRouterConfig) -> CachedRouterClient | None:
        with self._lock:
            cached = self._cached
        if cached is not None and cached.matches(cfg):
            return cached
        return None

    def resolve(self, cfg: InteractionRouterConfig) -> tuple[Any | None, str | None]:
        module = importlib.import_module("agent.auxiliary_client")
        resolve_provider_client = module.resolve_provider_client
        return resolve_provider_client(cfg.router_provider, model=cfg.router_model)

    def store(
        self,
        cfg: InteractionRouterConfig,
        client: Any,
        resolved_model: str,
    ) -> CachedRouterClient:
        cached = CachedRouterClient(
            provider=cfg.router_provider,
            requested_model=cfg.router_model,
            resolved_model=resolved_model,
            client=client,
            cached_at=time.monotonic(),
        )
        with self._lock:
            self._cached = cached
        return cached

    def prewarm(self, cfg: InteractionRouterConfig) -> bool:
        if not cfg.router_enabled:
            self.clear()
            return False
        client, model = self.resolve(cfg)
        if client is None or not model:
            self.clear()
            return False
        self.store(cfg, client, model)
        return True


router_client_cache = RouterClientCache()

__all__ = ["CachedRouterClient", "RouterClientCache", "router_client_cache"]
