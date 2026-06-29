"""Translate daemon config to interaction-router config."""
from __future__ import annotations

from typing import Any, Dict

from ..daemon_config import HERMES_HOME
from ..interaction_router import InteractionRouterConfig


def interaction_router_config_from_daemon_config(cfg: Dict[str, Any]) -> InteractionRouterConfig:
    """Translate daemon config keys into the standalone router config."""
    return InteractionRouterConfig.from_mapping({
        "router_enabled": cfg.get("interaction_router_enabled", True),
        "router_provider": cfg.get("interaction_router_provider", "openrouter"),
        "router_model": cfg.get("interaction_router_model", "google/gemini-2.5-flash-lite"),
        "router_timeout_seconds": cfg.get("interaction_router_timeout_seconds", 1.5),
        "router_min_confidence": cfg.get("interaction_router_min_confidence", 0.70),
        "small_model_enabled": cfg.get("interaction_router_small_model_enabled", True),
        "small_model_provider": cfg.get("interaction_router_small_model_provider", "openrouter"),
        "small_model_model": cfg.get("interaction_router_small_model_model", "google/gemini-2.5-flash-lite"),
        "small_model_timeout_seconds": cfg.get("interaction_router_small_model_timeout_seconds", 4.0),
        "ack_cache_enabled": cfg.get("interaction_router_ack_cache_enabled", True),
        "ack_cache_dir": cfg.get("interaction_router_ack_cache_dir", str(HERMES_HOME / "wakeword" / "ack_cache")),
    })


__all__ = ["interaction_router_config_from_daemon_config"]
